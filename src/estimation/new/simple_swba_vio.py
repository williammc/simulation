"""
Simplified Sliding Window Bundle Adjustment VIO estimator.

Core functionality:
- predict(): Store preintegrated IMU data between keyframes
- update(): Form constraints from IMU and visual measurements, then optimize
- Selective window of keyframes with marginalization
"""

from typing import List, Optional, Dict, Tuple, Any
import numpy as np
import logging
import json
from pathlib import Path
from pydantic import Field

from src.estimation.base_estimator import (
    BaseEstimator, 
    EstimatorConfig, 
    EstimatorState,
    EstimatorResult
)
from src.estimation.interfaces import (
    ProcessedVisualFrame,
    PreprocessedIMUData,
    VisualMeasurement
)
from src.common.data_structures import (
    Pose, Map, Landmark, Trajectory, TrajectoryState
)
from src.common.config import EstimatorType

logger = logging.getLogger(__name__)


class SimpleSWBAConfig(EstimatorConfig):
    """Simplified SWBA configuration."""
    
    # Window parameters
    window_size: int = Field(10, ge=3, le=20, description="Number of keyframes in window")
    keyframe_spacing: int = Field(5, ge=1, description="Create keyframe every N frames")
    
    # Optimization parameters
    max_iterations: int = Field(10, ge=1, le=50, description="Max optimization iterations")
    convergence_threshold: float = Field(1e-4, gt=0, description="Convergence threshold")
    damping_factor: float = Field(0.01, gt=0, description="Levenberg-Marquardt damping")
    
    # Visual measurement parameters
    min_measurements: int = Field(5, ge=1, description="Minimum measurements for update")
    ideal_coord_weight: float = Field(10.0, gt=0, description="Weight for ideal coordinates")
    pixel_coord_weight: float = Field(1.0, gt=0, description="Weight for pixel coordinates")
    
    def __init__(self, **data):
        """Initialize with SWBA estimator type."""
        super().__init__(**data)
        self.estimator_type = EstimatorType.SWBA


class SimpleSWBAVIO(BaseEstimator):
    """
    Simplified Sliding Window Bundle Adjustment VIO.
    
    Key simplifications:
    1. predict() just stores preintegrated IMU data
    2. update() decides on keyframes and triggers optimization
    3. Optimization uses precomputed Jacobians from measurements
    4. Simple marginalization of oldest keyframe
    """
    
    def __init__(self, config: SimpleSWBAConfig):
        """Initialize simplified SWBA VIO."""
        super().__init__(config, imu_calibration=None, camera_calibration=None)
        self.config: SimpleSWBAConfig = config
        
        # State
        self.current_pose: Optional[Pose] = None
        self.current_velocity: np.ndarray = np.zeros(3)
        self.current_bias = {'accel': np.zeros(3), 'gyro': np.zeros(3)}
        
        # All poses (for complete trajectory)
        self.all_poses: List[Pose] = []
        self.pnp_poses: List[Pose] = []  # PnP solved poses
        self.ground_truth_poses: List[Pose] = []  # Ground truth for comparison
        
        # Keyframe data
        self.keyframes: List[Tuple[Pose, ProcessedVisualFrame]] = []
        self.keyframe_ids: List[int] = []
        
        # IMU preintegration storage
        # Key: (from_kf_id, to_kf_id), Value: PreprocessedIMUData
        self.imu_constraints: Dict[Tuple[int, int], PreprocessedIMUData] = {}
        self.pending_imu: Optional[PreprocessedIMUData] = None
        
        # Landmarks
        self.landmarks: Dict[int, np.ndarray] = {}  # id -> position
        
        # Counters
        self.frame_count = 0
        self.keyframe_count = 0
        
        # Debug settings
        self.debug_dir = Path("debug_output")
        self.debug_enabled = True
        
    def initialize(self, initial_pose: Pose, initial_covariance: Optional[np.ndarray] = None, initial_velocity: Optional[np.ndarray] = None):
        """Initialize with first pose and optional velocity."""
        self.current_pose = initial_pose
        self.current_velocity = initial_velocity if initial_velocity is not None else np.zeros(3)
        self.current_state = EstimatorState(
            timestamp=initial_pose.timestamp,
            robot_pose=initial_pose,
            robot_velocity=self.current_velocity,
            robot_covariance=initial_covariance if initial_covariance else np.eye(15) * 0.1
        )
        
        # Add initial pose to all_poses
        self.all_poses.append(Pose(
            timestamp=initial_pose.timestamp,
            position=initial_pose.position.copy(),
            rotation_matrix=initial_pose.rotation_matrix.copy()
        ))
        
        logger.info("SimpleSWBAVIO initialized")
        
    def predict(self, imu_data: Any, dt: float):
        """
        Prediction step - store preintegrated IMU and propagate state.
        
        Args:
            imu_data: Preintegrated IMU measurements
            dt: Time delta (already in preintegration)
        """
        if self.current_pose is None:
            logger.warning("Not initialized")
            return
            
        # Store preintegrated IMU for later optimization
        self.pending_imu = imu_data
        
        # Propagate state using preintegrated measurements
        R = self.current_pose.rotation_matrix
        gravity = np.array([0, 0, -9.81])
        
        # Get rotation matrix - handle both attribute names
        if hasattr(imu_data, 'rotation_matrix'):
            delta_R = imu_data.rotation_matrix
        elif hasattr(imu_data, 'delta_rotation'):
            delta_R = imu_data.delta_rotation
        else:
            logger.warning("No rotation data in IMU preintegration")
            return
        
        # Standard VIO prediction equations
        new_R = R @ delta_R
        new_v = self.current_velocity + gravity * dt + R @ imu_data.delta_velocity
        new_p = self.current_pose.position + self.current_velocity * dt + \
                0.5 * gravity * dt**2 + R @ imu_data.delta_position
        
        # Update state
        self.current_pose = Pose(
            timestamp=self.current_pose.timestamp + dt,
            position=new_p,
            rotation_matrix=new_R
        )
        self.current_velocity = new_v
        
        # Store all poses for complete trajectory
        self.all_poses.append(Pose(
            timestamp=self.current_pose.timestamp,
            position=self.current_pose.position.copy(),
            rotation_matrix=self.current_pose.rotation_matrix.copy()
        ))
        
        self.total_predictions += 1
        
    def update(self, visual_frame: Any, landmarks: Optional[Map] = None):
        """
        Update step - process ALL frames and decide on keyframes for optimization.
        
        Args:
            visual_frame: Processed visual measurements with Jacobians
            landmarks: Optional landmark map for initialization
        """
        if self.current_pose is None:
            logger.warning("Not initialized")
            return
            
        self.frame_count += 1
        
        # IMPORTANT: Process every frame, not just keyframes
        # This ensures we track poses for all camera frames
        
        # Note: pending_imu is kept until we create a keyframe to store the constraint
        
        # Store current pose for this frame (already updated by predict())
        # This happens for EVERY frame
        current_frame_pose = Pose(
            timestamp=self.current_pose.timestamp,
            position=self.current_pose.position.copy(),
            rotation_matrix=self.current_pose.rotation_matrix.copy()
        )
        
        # Don't add duplicate poses (predict already added it)
        # Check if this pose was already added by predict()
        if (not self.all_poses or 
            self.all_poses[-1].timestamp != current_frame_pose.timestamp):
            self.all_poses.append(current_frame_pose)
        
        # Decide if this should be a keyframe for optimization
        is_keyframe = (self.frame_count % self.config.keyframe_spacing == 0) or \
                      len(self.keyframes) == 0
        
        logger.debug(f"Frame {self.frame_count}: is_keyframe={is_keyframe}, total_keyframes={len(self.keyframes)}")
        
        if is_keyframe:
            # Store keyframe (create new Pose object instead of copy)
            pose_copy = Pose(
                timestamp=self.current_pose.timestamp,
                position=self.current_pose.position.copy(),
                rotation_matrix=self.current_pose.rotation_matrix.copy()
            )
            self.keyframes.append((pose_copy, visual_frame))
            self.keyframe_ids.append(self.keyframe_count)
            
            # Store IMU constraint if we have pending IMU data
            if self.pending_imu and len(self.keyframe_ids) > 1:
                prev_id = self.keyframe_ids[-2]
                curr_id = self.keyframe_ids[-1]
                self.imu_constraints[(prev_id, curr_id)] = self.pending_imu
                self.pending_imu = None
            
            # Initialize new landmarks
            self._initialize_landmarks(visual_frame)
            
            # Maintain window size
            if len(self.keyframes) > self.config.window_size:
                self._marginalize_oldest()
            
            # Optimize if we have enough keyframes
            if len(self.keyframes) >= 3:
                self._optimize()
            
            self.keyframe_count += 1
            logger.debug(f"Created keyframe {self.keyframe_count}")
        
        self.total_updates += 1
        
    def optimize(self) -> bool:
        """Public optimize method for compatibility."""
        return self._optimize()
        
    def _optimize(self) -> bool:
        """
        Bundle adjustment optimization over sliding window.
        
        Forms constraints from:
        1. IMU preintegration between consecutive keyframes
        2. Visual reprojection to ideal coordinates
        """
        if len(self.keyframes) < 2:
            return True
            
        num_kf = len(self.keyframes)
        num_lm = len(self.landmarks)
        
        # State vector: [kf_positions..., landmark_positions...]
        # Simplified: only optimize positions (3 DOF each)
        state_dim = num_kf * 3 + num_lm * 3
        state = np.zeros(state_dim)
        
        # Initialize state
        for i, (pose, _) in enumerate(self.keyframes):
            state[i*3:(i+1)*3] = pose.position
            
        lm_ids = list(self.landmarks.keys())
        lm_idx_map = {lid: idx for idx, lid in enumerate(lm_ids)}
        for idx, lid in enumerate(lm_ids):
            state[num_kf*3 + idx*3:num_kf*3 + (idx+1)*3] = self.landmarks[lid]
        
        # Gauss-Newton optimization
        for iteration in range(self.config.max_iterations):
            residuals = []
            jacobian_rows = []
            
            # 1. IMU constraints
            for (from_id, to_id), imu_data in self.imu_constraints.items():
                if from_id in self.keyframe_ids and to_id in self.keyframe_ids:
                    from_idx = self.keyframe_ids.index(from_id)
                    to_idx = self.keyframe_ids.index(to_id)
                    
                    if from_idx < num_kf and to_idx < num_kf:
                        # IMU residual: position consistency
                        # Simplified: r = p_j - (p_i + predicted_delta_p)
                        p_i = state[from_idx*3:(from_idx+1)*3]
                        p_j = state[to_idx*3:(to_idx+1)*3]
                        
                        # Get rotation from stored keyframe (not optimizing rotation here)
                        R_i = self.keyframes[from_idx][0].rotation_matrix
                        dt = imu_data.dt if hasattr(imu_data, 'dt') else 0.1
                        gravity = np.array([0, 0, -9.81])
                        
                        # Predicted position change
                        predicted_p_j = p_i + self.current_velocity * dt + \
                                       0.5 * gravity * dt**2 + R_i @ imu_data.delta_position
                        
                        # Residual
                        r_imu = p_j - predicted_p_j
                        residuals.extend(r_imu * 10.0)  # Weight IMU constraints higher
                        
                        # Jacobian (simplified)
                        J_row = np.zeros((3, state_dim))
                        J_row[:, from_idx*3:(from_idx+1)*3] = -np.eye(3) * 10.0
                        J_row[:, to_idx*3:(to_idx+1)*3] = np.eye(3) * 10.0
                        jacobian_rows.append(J_row)
            
            # 2. Visual constraints
            for kf_idx, (pose, frame) in enumerate(self.keyframes):
                for meas in frame.measurements:
                    if meas.landmark_id not in lm_idx_map or not meas.is_valid:
                        continue
                        
                    lm_idx = lm_idx_map[meas.landmark_id]
                    
                    # Use ideal coordinates if available
                    if meas.has_ideal_coordinates and meas.ideal_residual is not None:
                        r = meas.ideal_residual
                        J_pose = meas.ideal_jacobian_wrt_pose[:, :3] if meas.ideal_jacobian_wrt_pose is not None else np.zeros((2, 3))
                        J_lm = meas.ideal_jacobian_wrt_landmark if meas.ideal_jacobian_wrt_landmark is not None else np.zeros((2, 3))
                        weight = self.config.ideal_coord_weight
                    else:
                        r = meas.residual
                        J_pose = meas.jacobian_wrt_pose[:, :3] if meas.jacobian_wrt_pose is not None else np.zeros((2, 3))
                        J_lm = meas.jacobian_wrt_landmark if meas.jacobian_wrt_landmark is not None else np.zeros((2, 3))
                        weight = self.config.pixel_coord_weight
                    
                    # Apply robust weight from preprocessing
                    weight *= meas.robust_weight
                    
                    residuals.extend(r * weight)
                    
                    # Build Jacobian row
                    J_row = np.zeros((2, state_dim))
                    J_row[:, kf_idx*3:(kf_idx+1)*3] = J_pose * weight
                    J_row[:, num_kf*3 + lm_idx*3:num_kf*3 + (lm_idx+1)*3] = J_lm * weight
                    jacobian_rows.append(J_row)
            
            if len(residuals) == 0:
                break
                
            # Stack and solve
            r = np.array(residuals)
            J = np.vstack(jacobian_rows)
            
            # Levenberg-Marquardt
            H = J.T @ J + self.config.damping_factor * np.eye(state_dim)
            g = -J.T @ r
            
            try:
                delta = np.linalg.solve(H, g)
                
                # Line search for step size
                alpha = 1.0
                cost_before = 0.5 * np.dot(r, r)
                
                # Try full step
                state_new = state + alpha * delta
                
                # Accept update
                state = state_new
                
                # Check convergence
                if np.linalg.norm(delta) < self.config.convergence_threshold:
                    logger.debug(f"Converged at iteration {iteration}")
                    break
                    
            except np.linalg.LinAlgError:
                logger.warning("Singular matrix in optimization")
                break
        
        # Update estimates
        for i in range(num_kf):
            self.keyframes[i][0].position[:] = state[i*3:(i+1)*3]
            
        for idx, lid in enumerate(lm_ids):
            self.landmarks[lid] = state[num_kf*3 + idx*3:num_kf*3 + (idx+1)*3].copy()
            
        # Update current pose to match last keyframe
        if self.keyframes:
            self.current_pose.position[:] = self.keyframes[-1][0].position
            
        self.total_iterations += iteration + 1
        return True
        
    def _initialize_landmarks(self, frame: ProcessedVisualFrame):
        """Initialize new landmarks from measurements."""
        default_depth = 5.0
        
        for meas in frame.measurements:
            if meas.landmark_id not in self.landmarks:
                # Use bearing vector or ideal coordinates
                if meas.bearing_vector is not None:
                    p_cam = meas.bearing_vector * default_depth
                elif meas.observed_ideal is not None:
                    x, y = meas.observed_ideal
                    p_cam = np.array([x * default_depth, y * default_depth, default_depth])
                else:
                    # Fallback to pixel with assumed intrinsics
                    fx = fy = 500.0
                    cx, cy = 320.0, 240.0
                    u, v = meas.observed_pixel
                    x = (u - cx) / fx * default_depth
                    y = (v - cy) / fy * default_depth
                    p_cam = np.array([x, y, default_depth])
                
                # Transform to world
                R = self.current_pose.rotation_matrix
                t = self.current_pose.position
                p_world = R @ p_cam + t
                
                self.landmarks[meas.landmark_id] = p_world
                
    def _marginalize_oldest(self):
        """Remove oldest keyframe and associated constraints."""
        if len(self.keyframes) <= self.config.window_size:
            return
            
        # Remove oldest
        old_kf = self.keyframes.pop(0)
        old_id = self.keyframe_ids.pop(0)
        
        # Remove associated IMU constraints
        keys_to_remove = [k for k in self.imu_constraints.keys() if old_id in k]
        for k in keys_to_remove:
            del self.imu_constraints[k]
            
        logger.debug(f"Marginalized keyframe {old_id}")
    
    def _dump_debug_data(self):
        """Dump debug data to JSON files."""
        try:
            # Create debug directory
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Dump estimated poses
            estimated_data = {
                'all_poses': [
                    {
                        'timestamp': p.timestamp,
                        'position': p.position.tolist(),
                        'rotation': p.rotation_matrix.tolist()
                    } for p in self.all_poses
                ],
                'keyframe_poses': [
                    {
                        'timestamp': p.timestamp,
                        'position': p.position.tolist(),
                        'rotation': p.rotation_matrix.tolist()
                    } for p, _ in self.keyframes
                ],
                'num_frames': self.frame_count,
                'num_keyframes': len(self.keyframes),
                'num_landmarks': len(self.landmarks)
            }
            
            with open(self.debug_dir / 'simple_swba_poses.json', 'w') as f:
                json.dump(estimated_data, f, indent=2)
            
            # Dump PnP poses if available
            if self.pnp_poses:
                pnp_data = {
                    'pnp_poses': [
                        {
                            'timestamp': p.timestamp,
                            'position': p.position.tolist(),
                            'rotation': p.rotation_matrix.tolist()
                        } for p in self.pnp_poses
                    ]
                }
                with open(self.debug_dir / 'pnp_poses.json', 'w') as f:
                    json.dump(pnp_data, f, indent=2)
            
            # Dump landmarks
            landmark_data = {
                'landmarks': {
                    str(lid): pos.tolist() for lid, pos in self.landmarks.items()
                }
            }
            with open(self.debug_dir / 'simple_swba_landmarks.json', 'w') as f:
                json.dump(landmark_data, f, indent=2)
            
            # Dump ground truth if available
            if self.ground_truth_poses:
                gt_data = {
                    'ground_truth_poses': [
                        {
                            'timestamp': p.timestamp,
                            'position': p.position.tolist(),
                            'rotation': p.rotation_matrix.tolist()
                        } for p in self.ground_truth_poses
                    ]
                }
                with open(self.debug_dir / 'ground_truth_poses.json', 'w') as f:
                    json.dump(gt_data, f, indent=2)
            
            logger.info(f"Debug data dumped to {self.debug_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to dump debug data: {e}")
        
    def set_ground_truth(self, gt_trajectory: Any):
        """Store ground truth trajectory for comparison."""
        if hasattr(gt_trajectory, 'states'):
            self.ground_truth_poses = [state.pose for state in gt_trajectory.states]
            logger.info(f"Stored {len(self.ground_truth_poses)} ground truth poses")
    
    def marginalize(self):
        """Public marginalize for compatibility."""
        self._marginalize_oldest()
        
    def get_state_vector(self) -> np.ndarray:
        """Get current state vector."""
        if self.current_pose is None:
            return np.zeros(21)
            
        state = np.zeros(21)
        state[0:3] = self.current_pose.position
        state[3:12] = self.current_pose.rotation_matrix.flatten()
        state[12:15] = self.current_velocity
        state[15:18] = self.current_bias['accel']
        state[18:21] = self.current_bias['gyro']
        return state
        
    def get_covariance_matrix(self) -> Optional[np.ndarray]:
        """Get covariance (simplified)."""
        return np.eye(15) * 0.1 if self.current_state else None
    
    def get_result(self) -> EstimatorResult:
        """Get estimation result."""
        # Dump debug data if enabled
        if self.debug_enabled:
            self._dump_debug_data()
        
        # Build trajectory from ALL poses (not just keyframes)
        trajectory = Trajectory()
        
        # Use all_poses if available, otherwise use keyframes
        poses_to_use = self.all_poses if self.all_poses else [pose for pose, _ in self.keyframes]
        
        # If still no poses, add current pose
        if not poses_to_use and self.current_pose:
            poses_to_use = [self.current_pose]
        
        for pose in poses_to_use:
            state = TrajectoryState(pose=pose)
            trajectory.add_state(state)
        
        # Build landmark map
        landmark_map = Map()
        for lid, position in self.landmarks.items():
            landmark = Landmark(
                id=lid,
                position=position,
                descriptor=None
            )
            landmark_map.add_landmark(landmark)
        
        # Get states history from all poses
        states = []
        for pose in poses_to_use:
            state = EstimatorState(
                timestamp=pose.timestamp,
                robot_pose=pose,
                robot_velocity=self.current_velocity,
                robot_covariance=self.get_covariance_matrix()
            )
            states.append(state)
        
        logger.info(f"Returning result with {len(poses_to_use)} poses, {len(self.keyframes)} keyframes, {len(self.landmarks)} landmarks")
        
        return EstimatorResult(
            trajectory=trajectory,
            landmarks=landmark_map,
            states=states if states else [self.current_state] if self.current_state else [],
            runtime_ms=0.0,
            iterations=self.total_iterations,
            converged=True,
            final_cost=0.0,
            metadata={
                'estimator_type': 'simple_swba',
                'num_keyframes': len(self.keyframes),
                'num_all_poses': len(self.all_poses),
                'num_landmarks': len(self.landmarks)
            }
        )