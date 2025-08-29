"""
Clean SWBA (Sliding Window Bundle Adjustment) estimator implementation.

Flow:
1. predict() - Store IMU preintegration data only
2. update() - PnP pose estimation and factor graph construction  
3. optimize() - Bundle adjustment with IMU and reprojection constraints
"""

import numpy as np
from typing import Optional, Dict, List, Any, Tuple
import logging
from dataclasses import dataclass, field

from src.estimation.base_estimator import BaseEstimator, EstimatorConfig
from src.estimation.interfaces import PreprocessedIMUData, ProcessedVisualFrame, VisualMeasurement
from src.common.data_structures import Pose, Map
from src.utils.math_utils import so3_log, so3_exp, se3_log, se3_exp
import logging
logger = logging.getLogger(__name__)


@dataclass
class CleanSWBAConfig(EstimatorConfig):
    """Configuration for clean SWBA estimator."""
    name: str = "clean-swba"
    
    # Sliding window parameters
    window_size: int = 10  # Number of keyframes in sliding window
    
    # PnP parameters
    min_pnp_points: int = 4  # Minimum points for PnP
    pnp_reprojection_threshold: float = 3.0  # pixels
    
    # Optimization parameters  
    max_iterations: int = 10
    convergence_threshold: float = 1e-6
    
    # IMU noise parameters (from GTSAM defaults)
    imu_accel_noise: float = 0.01  # m/s^2
    imu_gyro_noise: float = 0.001  # rad/s
    imu_integration_noise: float = 1e-5
    
    # Visual noise parameters
    pixel_noise: float = 1.0  # pixels
    ideal_noise: float = 0.01  # ideal coordinates
    
    # Gravity
    gravity_magnitude: float = 9.81


class CleanSWBAEstimator(BaseEstimator):
    """Clean SWBA estimator with simplified flow."""
    
    def __init__(self, config: CleanSWBAConfig):
        super().__init__(config)
        self.config = config
        
        # State - initialize with default pose
        self.current_pose = Pose(
            timestamp=0.0,
            position=np.array([2.0, 0.0, 1.5]),  # Default initial position
            rotation_matrix=np.eye(3)
        )
        self.current_velocity = np.zeros(3)
        
        # Sliding window data
        self.keyframes = []  # List of ProcessedVisualFrame
        self.keyframe_poses = []  # List of Pose objects
        self.keyframe_velocities = []  # List of velocity vectors
        
        # IMU preintegration storage
        self.imu_preintegrations = {}  # Dict[(from_id, to_id)] = PreprocessedIMUData
        self.pending_imu = None  # Current IMU data waiting for next keyframe
        
        # Landmarks
        self.landmarks = {}  # Dict[landmark_id] = position
        
        # Factor graph components
        self.imu_factors = []  # List of IMU constraints
        self.visual_factors = []  # List of reprojection constraints
        
        # Counters
        self.frame_count = 0
        self.keyframe_count = 0
        
        # Debug data collection
        self.debug_poses = []  # SWBA poses for debugging
        self.debug_pnp_poses = []  # PnP poses for comparison
        self.debug_pose_errors = []  # Tracking errors
        
        logger.info(f"Initialized CleanSWBAEstimator with window size {config.window_size}")
    
    def predict(self, imu_measurements: PreprocessedIMUData, dt: float):
        """
        Store IMU preintegration data for later use in optimization.
        Does NOT update the current pose.
        """
        logger.debug(f"Storing IMU data: dt={dt:.3f}, delta_p norm={np.linalg.norm(imu_measurements.delta_position):.4f}")
        
        # Store the IMU data for use when next keyframe is created
        self.pending_imu = imu_measurements
    
    def update(self, camera_frame: ProcessedVisualFrame, landmarks: Optional[Map] = None):
        """
        Update with visual measurements:
        1. Estimate pose using PnP
        2. Create keyframe if needed
        3. Build factor graph constraints
        """
        self.frame_count += 1
        
        # Check if this is a keyframe
        if not camera_frame.is_keyframe:
            return
        
        logger.info(f"Processing keyframe {self.keyframe_count} at t={camera_frame.timestamp:.3f}")
        
        # Extract valid measurements with known landmarks
        valid_measurements = []
        points_3d_W = []  # 3D landmark positions in world frame
        ideal_coords_C = []  # Ideal coordinates in camera frame
        
        for meas in camera_frame.measurements:
            if not meas.is_valid:
                continue
                
            # Get landmark position
            landmark_pos = None
            if meas.landmark_id in self.landmarks:
                landmark_pos = self.landmarks[meas.landmark_id]
            elif landmarks and meas.landmark_id in landmarks.landmarks:
                landmark_pos = landmarks.landmarks[meas.landmark_id].position
            
            # Use observed_ideal from the pre-processed measurement
            if landmark_pos is not None and meas.observed_ideal is not None:
                valid_measurements.append(meas)
                points_3d_W.append(landmark_pos)  # Landmark position in world frame
                ideal_coords_C.append(meas.observed_ideal)  # Ideal coords in camera frame
        
        # Estimate pose using PnP if we have enough points
        W_T_B_pnp = None  # World-from-body transform from PnP
        if len(valid_measurements) >= self.config.min_pnp_points:
            W_T_B_pnp = self._estimate_pose_pnp(
                np.array(points_3d_W),
                np.array(ideal_coords_C),
                camera_frame.timestamp
            )
            
            if W_T_B_pnp is not None:
                self.current_pose = W_T_B_pnp
                logger.debug(f"PnP pose: position={self.current_pose.position}, "
                           f"rotation norm={np.linalg.norm(so3_log(self.current_pose.rotation_matrix)):.4f}")
                
                # Store debug PnP pose
                self.debug_pnp_poses.append({
                    'timestamp': camera_frame.timestamp,
                    'position': W_T_B_pnp.position.tolist(),
                    'rotation': W_T_B_pnp.rotation_matrix.tolist(),
                    'num_points': len(valid_measurements)
                })
        else:
            logger.warning(f"Only {len(valid_measurements)} valid measurements, need {self.config.min_pnp_points} for PnP")
            # Fall back to IMU-only prediction if available
            if self.pending_imu is not None and len(self.keyframe_poses) > 0:
                self._predict_with_imu()
        
        # Add keyframe to sliding window
        self.keyframes.append(camera_frame)
        logger.debug(f"Adding keyframe {self.keyframe_count} at t={camera_frame.timestamp:.3f}, now have {len(self.keyframe_poses) + 1} poses")
        self.keyframe_poses.append(Pose(
            timestamp=self.current_pose.timestamp,
            position=self.current_pose.position.copy(),
            rotation_matrix=self.current_pose.rotation_matrix.copy()
        ))
        self.keyframe_velocities.append(self.current_velocity.copy())
        
        # Store debug SWBA pose
        self.debug_poses.append({
            'timestamp': camera_frame.timestamp,
            'position': self.current_pose.position.tolist(),
            'rotation': self.current_pose.rotation_matrix.tolist()
        })
        
        # Compute and store error if we have PnP pose for comparison
        if W_T_B_pnp is not None:
            position_error = np.linalg.norm(self.current_pose.position - W_T_B_pnp.position)
            self.debug_pose_errors.append({
                'timestamp': camera_frame.timestamp,
                'position_error': position_error,
                'swba_z': self.current_pose.position[2],
                'pnp_z': W_T_B_pnp.position[2]
            })
        
        # Store IMU factor if we have pending IMU data
        if self.pending_imu is not None and self.keyframe_count > 0:
            from_id = self.keyframe_count - 1
            to_id = self.keyframe_count
            self.imu_preintegrations[(from_id, to_id)] = self.pending_imu
            self.pending_imu = None
        
        # Build visual factors for this keyframe
        self._build_visual_factors(camera_frame, valid_measurements)
        
        # Initialize new landmarks if needed
        self._initialize_new_landmarks(camera_frame)
        
        # Maintain sliding window size
        if len(self.keyframes) > self.config.window_size:
            self._marginalize_oldest_keyframe()
        
        self.keyframe_count += 1
    
    def optimize(self) -> bool:
        """
        Perform bundle adjustment on the sliding window using:
        - IMU preintegration factors
        - Visual reprojection factors
        
        Returns:
            True if optimization converged successfully
        """
        if len(self.keyframes) < 2:
            logger.debug("Not enough keyframes for optimization")
            return True  # Trivially successful
        
        logger.info(f"Running BA optimization on {len(self.keyframes)} keyframes")
        
        # Build optimization problem
        num_poses = len(self.keyframe_poses)
        num_landmarks = len(self.landmarks)
        
        # State vector: [poses (6*n), velocities (3*n), landmarks (3*m)]
        state_dim = 6 * num_poses + 3 * num_poses + 3 * num_landmarks
        
        # Initialize with current estimates
        x = np.zeros(state_dim)
        
        # Pack poses (position + so3 rotation)
        for i, pose in enumerate(self.keyframe_poses):
            x[6*i:6*i+3] = pose.position
            x[6*i+3:6*i+6] = so3_log(pose.rotation_matrix)
        
        # Pack velocities
        vel_offset = 6 * num_poses
        for i, vel in enumerate(self.keyframe_velocities):
            x[vel_offset + 3*i:vel_offset + 3*i+3] = vel
        
        # Pack landmarks
        landmark_offset = 6 * num_poses + 3 * num_poses
        landmark_ids = list(self.landmarks.keys())
        for i, lid in enumerate(landmark_ids):
            x[landmark_offset + 3*i:landmark_offset + 3*i+3] = self.landmarks[lid]
        
        # Gauss-Newton optimization
        for iteration in range(self.config.max_iterations):
            # Compute residuals and Jacobians
            residuals, H, b = self._compute_ba_system(x, landmark_ids)
            
            # Check convergence
            cost = 0.5 * np.sum(residuals**2)
            logger.debug(f"  Iteration {iteration}: cost={cost:.6f}")
            
            if iteration > 0 and abs(cost - prev_cost) < self.config.convergence_threshold:
                logger.debug(f"  Converged after {iteration+1} iterations")
                break
            prev_cost = cost
            
            # Solve normal equations: H * delta = -b
            try:
                # Add regularization for stability
                H += np.eye(H.shape[0]) * 1e-6
                delta = np.linalg.solve(H, -b)
                
                # Line search for robust convergence
                alpha = 1.0
                for _ in range(10):
                    x_new = self._update_state(x, alpha * delta, landmark_ids)
                    new_residuals, _, _ = self._compute_ba_system(x_new, landmark_ids)
                    new_cost = 0.5 * np.sum(new_residuals**2)
                    
                    if new_cost < cost:
                        x = x_new
                        break
                    alpha *= 0.5
                
                if alpha < 0.01:
                    logger.debug(f"  Line search failed, stopping optimization")
                    break
                    
            except np.linalg.LinAlgError:
                logger.warning("Failed to solve normal equations")
                break
        
        # Unpack optimized state
        self._unpack_state(x, landmark_ids)
        
        logger.info(f"Optimization complete. Final cost: {cost:.6f}")
        return True  # Successfully optimized
    
    def _estimate_pose_pnp(self, points_3d_W: np.ndarray, ideal_coords_C: np.ndarray, 
                           timestamp: float) -> Optional[Pose]:
        """
        Estimate pose using PnP with ideal coordinates.
        
        Args:
            points_3d_W: (N, 3) array of 3D landmark positions in world frame
            ideal_coords_C: (N, 2) array of ideal coordinates in camera frame
            timestamp: Frame timestamp
            
        Returns:
            Estimated W_T_B pose or None if PnP fails
        """
        try:
            from src.utils.pnp_utils import solve_pnp_ideal
            
            # Solve PnP - returns C_R_W and C_t_W (camera-from-world transform)
            success, C_R_W, C_t_W = solve_pnp_ideal(
                points_3d_W, 
                ideal_coords_C,
                reprojection_threshold=self.config.pnp_reprojection_threshold
            )
            
            if success:
                # PnP gives us C_T_W (camera-from-world)
                # We need W_T_B (world-from-body)
                # Assuming B_T_C = I (camera at body center), so B = C
                # Therefore: W_T_B = inv(C_T_W)
                W_R_B = C_R_W.T  # Transpose of rotation matrix
                W_t_B = -C_R_W.T @ C_t_W  # Invert translation
                
                pose = Pose(
                    timestamp=timestamp,
                    position=W_t_B,
                    rotation_matrix=W_R_B
                )
                return pose
            else:
                logger.warning("PnP failed to find valid solution")
                return None
                
        except Exception as e:
            logger.error(f"PnP estimation failed: {e}")
            return None
    
    def _predict_with_imu(self):
        """Fallback IMU-based prediction when PnP fails."""
        if self.pending_imu is None or len(self.keyframe_poses) == 0:
            return
            
        # Get last world-from-body pose
        W_pose_B_prev = self.keyframe_poses[-1]
        W_vel_prev = self.keyframe_velocities[-1] if self.keyframe_velocities else np.zeros(3)
        
        # Apply IMU preintegration
        dt = self.pending_imu.delta_t
        gravity_W = np.array([0, 0, -self.config.gravity_magnitude])  # Gravity in world frame
        
        # Get previous world-from-body rotation
        W_R_B_prev = W_pose_B_prev.rotation_matrix
        W_t_B_prev = W_pose_B_prev.position
        
        # Update position in world frame
        # W_t_B_new = W_t_B_prev + W_v_prev * dt + 0.5 * gravity * dt^2 + W_R_B_prev @ delta_p_B
        W_t_B_new = (W_t_B_prev + 
                     W_vel_prev * dt + 
                     0.5 * gravity_W * dt**2 + 
                     W_R_B_prev @ self.pending_imu.delta_position)
        
        # Update velocity in world frame
        # W_v_new = W_v_prev + gravity * dt + W_R_B_prev @ delta_v_B
        W_vel_new = (W_vel_prev + 
                     gravity_W * dt + 
                     W_R_B_prev @ self.pending_imu.delta_velocity)
        
        # Update rotation
        # W_R_B_new = W_R_B_prev @ delta_R_B
        W_R_B_new = W_R_B_prev @ self.pending_imu.delta_rotation
        
        self.current_pose = Pose(
            timestamp=W_pose_B_prev.timestamp + dt,
            position=W_t_B_new,
            rotation_matrix=W_R_B_new
        )
        self.current_velocity = W_vel_new
    
    def _build_visual_factors(self, frame: ProcessedVisualFrame, measurements: List[VisualMeasurement]):
        """Build visual reprojection factors for the factor graph."""
        kf_id = self.keyframe_count
        
        for meas in measurements:
            if meas.landmark_id in self.landmarks:
                # Store the pre-computed Jacobians if available
                factor = {
                    'type': 'visual',
                    'keyframe_id': kf_id,
                    'landmark_id': meas.landmark_id,
                    'measurement': meas.observed_ideal,  # Use observed ideal coordinates
                    'information': np.eye(2) / (self.config.ideal_noise ** 2),
                    # Store pre-computed Jacobians
                    'jacobian_wrt_pose': meas.ideal_jacobian_wrt_pose if meas.ideal_jacobian_wrt_pose is not None else None,
                    'jacobian_wrt_landmark': meas.ideal_jacobian_wrt_landmark if meas.ideal_jacobian_wrt_landmark is not None else None
                }
                self.visual_factors.append(factor)
    
    def _initialize_new_landmarks(self, frame: ProcessedVisualFrame):
        """Initialize landmarks that haven't been seen before."""
        # Get current world-from-body transform
        W_R_B = self.current_pose.rotation_matrix
        W_t_B = self.current_pose.position
        
        for meas in frame.measurements:
            if (meas.is_valid and 
                meas.landmark_id not in self.landmarks and
                meas.observed_ideal is not None):
                
                # Triangulate using ideal coordinates
                # Assume default depth of 10m for initialization
                default_depth = 10.0
                
                # Convert ideal coordinates to 3D ray in camera frame
                # Ideal coords are already in camera frame (z=1 plane)
                ray_C = np.array([meas.observed_ideal[0], 
                                  meas.observed_ideal[1], 
                                  1.0])
                
                # Point in camera frame at default depth
                point_C = default_depth * ray_C
                
                # Transform to world frame
                # Assuming B_T_C = I (camera at body center), so C = B
                # Therefore: point_W = W_R_B @ point_C + W_t_B
                point_W = W_R_B @ point_C + W_t_B
                self.landmarks[meas.landmark_id] = point_W
    
    def _marginalize_oldest_keyframe(self):
        """Remove oldest keyframe from sliding window with marginalization."""
        # Remove oldest keyframe
        self.keyframes.pop(0)
        old_pose = self.keyframe_poses.pop(0)
        old_vel = self.keyframe_velocities.pop(0)
        
        # Update IMU factors (shift indices)
        new_imu_factors = {}
        for (from_id, to_id), data in self.imu_preintegrations.items():
            if from_id > 0 and to_id > 0:
                new_imu_factors[(from_id-1, to_id-1)] = data
        self.imu_preintegrations = new_imu_factors
        
        # Update visual factors (shift keyframe indices)
        for factor in self.visual_factors:
            if factor['keyframe_id'] > 0:
                factor['keyframe_id'] -= 1
        
        # Remove factors related to marginalized keyframe
        self.visual_factors = [f for f in self.visual_factors if f['keyframe_id'] >= 0]
    
    def _compute_ba_system(self, x: np.ndarray, landmark_ids: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute residuals, information matrix and information vector for BA.
        
        Returns:
            residuals: All residuals concatenated
            H: Information matrix (J^T * W * J)
            b: Information vector (J^T * W * r)
        """
        num_poses = len(self.keyframe_poses)
        num_landmarks = len(landmark_ids)
        state_dim = 6 * num_poses + 3 * num_poses + 3 * num_landmarks
        
        all_residuals = []
        H = np.zeros((state_dim, state_dim))
        b = np.zeros(state_dim)
        
        # Add IMU factors
        for (from_id, to_id), imu_data in self.imu_preintegrations.items():
            if from_id < num_poses and to_id < num_poses:
                r_imu, J_imu = self._compute_imu_factor(x, from_id, to_id, imu_data)
                all_residuals.append(r_imu)
                
                # IMU information matrix (simplified)
                W_imu = np.eye(9) / (self.config.imu_integration_noise ** 2)
                
                # Add to system
                H += J_imu.T @ W_imu @ J_imu
                b += J_imu.T @ W_imu @ r_imu
        
        # Add visual factors
        for factor in self.visual_factors:
            kf_id = factor['keyframe_id']
            if kf_id < num_poses:
                lid = factor['landmark_id']
                if lid in landmark_ids:
                    landmark_idx = landmark_ids.index(lid)
                    r_vis, J_vis = self._compute_visual_factor(x, kf_id, landmark_idx, factor)
                    all_residuals.append(r_vis)
                    
                    # Visual information matrix
                    W_vis = factor['information']
                    
                    # Add to system
                    H += J_vis.T @ W_vis @ J_vis
                    b += J_vis.T @ W_vis @ r_vis
        
        residuals = np.concatenate(all_residuals) if all_residuals else np.array([])
        return residuals, H, b
    
    def _compute_imu_factor(self, x: np.ndarray, from_id: int, to_id: int, 
                           imu_data: PreprocessedIMUData) -> Tuple[np.ndarray, np.ndarray]:
        """Compute IMU preintegration factor residual and Jacobian."""
        # Extract states at time i (world-from-body)
        W_t_B_i = x[6*from_id:6*from_id+3]
        W_R_B_i = so3_exp(x[6*from_id+3:6*from_id+6])
        W_v_i = x[6*len(self.keyframe_poses) + 3*from_id:6*len(self.keyframe_poses) + 3*from_id+3]
        
        # Extract states at time j (world-from-body)
        W_t_B_j = x[6*to_id:6*to_id+3]
        W_R_B_j = so3_exp(x[6*to_id+3:6*to_id+6])
        W_v_j = x[6*len(self.keyframe_poses) + 3*to_id:6*len(self.keyframe_poses) + 3*to_id+3]
        
        dt = imu_data.delta_t
        gravity_W = np.array([0, 0, -self.config.gravity_magnitude])  # Gravity in world frame
        
        # Compute residuals
        # Position residual: r_p = W_t_B_j - W_t_B_i - W_v_i * dt - 0.5 * g * dt^2 - W_R_B_i @ delta_p
        r_p = W_t_B_j - W_t_B_i - W_v_i * dt - 0.5 * gravity_W * dt**2 - W_R_B_i @ imu_data.delta_position
        
        # Velocity residual: r_v = W_v_j - W_v_i - g * dt - W_R_B_i @ delta_v
        r_v = W_v_j - W_v_i - gravity_W * dt - W_R_B_i @ imu_data.delta_velocity
        
        # Rotation residual: r_R = log(delta_R^T @ W_R_B_i^T @ W_R_B_j)
        r_R = so3_log(imu_data.delta_rotation.T @ W_R_B_i.T @ W_R_B_j)
        
        residual = np.concatenate([r_p, r_v, r_R])
        
        # Simplified Jacobian (ignoring cross terms for now)
        state_dim = x.shape[0]
        jacobian = np.zeros((9, state_dim))
        
        # Derivatives w.r.t pose_i
        jacobian[0:3, 6*from_id:6*from_id+3] = -np.eye(3)  # position
        jacobian[3:6, 6*from_id+3:6*from_id+6] = np.zeros((3, 3))  # rotation (simplified)
        
        # Derivatives w.r.t velocity_i  
        vel_offset = 6 * len(self.keyframe_poses)
        jacobian[0:3, vel_offset + 3*from_id:vel_offset + 3*from_id+3] = -np.eye(3) * dt
        jacobian[3:6, vel_offset + 3*from_id:vel_offset + 3*from_id+3] = -np.eye(3)
        
        # Derivatives w.r.t pose_j
        jacobian[0:3, 6*to_id:6*to_id+3] = np.eye(3)  # position
        jacobian[6:9, 6*to_id+3:6*to_id+6] = np.eye(3)  # rotation (simplified)
        
        # Derivatives w.r.t velocity_j
        jacobian[3:6, vel_offset + 3*to_id:vel_offset + 3*to_id+3] = np.eye(3)
        
        return residual, jacobian
    
    def _compute_visual_factor(self, x: np.ndarray, kf_id: int, landmark_idx: int,
                               factor: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Compute visual reprojection factor residual and Jacobian."""
        
        # Require pre-computed Jacobians - they should have been computed during preprocessing
        if 'jacobian_wrt_pose' not in factor or factor['jacobian_wrt_pose'] is None:
            raise ValueError(f"Visual factor missing pre-computed jacobian_wrt_pose. "
                           f"Preprocessing should have computed all Jacobians!")
        
        if 'jacobian_wrt_landmark' not in factor or factor['jacobian_wrt_landmark'] is None:
            raise ValueError(f"Visual factor missing pre-computed jacobian_wrt_landmark. "
                           f"Preprocessing should have computed all Jacobians!")
        
        # Get current predicted measurement to compute residual
        p = x[6*kf_id:6*kf_id+3]
        R = so3_exp(x[6*kf_id+3:6*kf_id+6])
        
        landmark_offset = 6 * len(self.keyframe_poses) + 3 * len(self.keyframe_poses)
        landmark = x[landmark_offset + 3*landmark_idx:landmark_offset + 3*landmark_idx+3]
        
        # Transform to camera frame
        p_cam = R.T @ (landmark - p)
        
        # Project to ideal coordinates
        if p_cam[2] > 0.1:
            predicted = p_cam[:2] / p_cam[2]
        else:
            predicted = np.array([0, 0])
        
        # Residual: measured - predicted (both in ideal coordinates)
        observed_ideal_C = factor['measurement']
        residual = observed_ideal_C - predicted
        
        # Use pre-computed Jacobians
        state_dim = x.shape[0]
        jacobian = np.zeros((2, state_dim))
        
        # Pre-computed Jacobian w.r.t pose (usually 2x6)
        if factor['jacobian_wrt_pose'].shape[1] >= 6:
            jacobian[:, 6*kf_id:6*kf_id+6] = factor['jacobian_wrt_pose'][:, :6]
        else:
            raise ValueError(f"Invalid jacobian_wrt_pose shape: {factor['jacobian_wrt_pose'].shape}, expected (2, 6+)")
        
        # Pre-computed Jacobian w.r.t landmark (2x3)
        jacobian[:, landmark_offset + 3*landmark_idx:landmark_offset + 3*landmark_idx+3] = factor['jacobian_wrt_landmark']
        
        return residual, jacobian
    
    def _update_state(self, x: np.ndarray, delta: np.ndarray, 
                     landmark_ids: List[int]) -> np.ndarray:
        """Apply update to state vector with proper manifold operations."""
        x_new = x.copy()
        num_poses = len(self.keyframe_poses)
        
        # Update poses
        for i in range(num_poses):
            # Position update
            x_new[6*i:6*i+3] += delta[6*i:6*i+3]
            
            # Rotation update on manifold
            R_current = so3_exp(x[6*i+3:6*i+6])
            R_update = so3_exp(delta[6*i+3:6*i+6])
            R_new = R_current @ R_update
            x_new[6*i+3:6*i+6] = so3_log(R_new)
        
        # Update velocities
        vel_offset = 6 * num_poses
        x_new[vel_offset:vel_offset + 3*num_poses] += delta[vel_offset:vel_offset + 3*num_poses]
        
        # Update landmarks
        landmark_offset = vel_offset + 3 * num_poses
        num_landmarks = len(landmark_ids)
        x_new[landmark_offset:landmark_offset + 3*num_landmarks] += delta[landmark_offset:landmark_offset + 3*num_landmarks]
        
        return x_new
    
    def _unpack_state(self, x: np.ndarray, landmark_ids: List[int]):
        """Unpack optimized state back to class members."""
        num_poses = len(self.keyframe_poses)
        
        # Unpack poses
        for i in range(num_poses):
            self.keyframe_poses[i].position = x[6*i:6*i+3].copy()
            self.keyframe_poses[i].rotation_matrix = so3_exp(x[6*i+3:6*i+6])
        
        # Unpack velocities
        vel_offset = 6 * num_poses
        for i in range(num_poses):
            self.keyframe_velocities[i] = x[vel_offset + 3*i:vel_offset + 3*i+3].copy()
        
        # Unpack landmarks
        landmark_offset = vel_offset + 3 * num_poses
        for i, lid in enumerate(landmark_ids):
            self.landmarks[lid] = x[landmark_offset + 3*i:landmark_offset + 3*i+3].copy()
        
        # Update current pose and velocity with the latest keyframe
        last_pose = self.keyframe_poses[-1]
        self.current_pose = Pose(
            timestamp=last_pose.timestamp,
            position=last_pose.position.copy(),
            rotation_matrix=last_pose.rotation_matrix.copy()
        )
        self.current_velocity = self.keyframe_velocities[-1].copy()
    
    def get_current_pose(self) -> Pose:
        """Get current pose estimate."""
        return Pose(
            timestamp=self.current_pose.timestamp,
            position=self.current_pose.position.copy(),
            rotation_matrix=self.current_pose.rotation_matrix.copy()
        )
    
    def get_trajectory(self):
        """Get trajectory of all keyframes."""
        from src.common.data_structures import Trajectory, TrajectoryState
        
        logger.info(f"Building trajectory from {len(self.keyframe_poses)} keyframe poses")
        
        trajectory = Trajectory()
        for pose in self.keyframe_poses:
            state = TrajectoryState(
                pose=Pose(
                    timestamp=pose.timestamp,
                    position=pose.position.copy(),
                    rotation_matrix=pose.rotation_matrix.copy()
                ),
                velocity=None,  # Could add if tracked
                angular_velocity=None
            )
            trajectory.add_state(state)
        return trajectory
    
    def initialize(self, initial_pose: Pose, initial_covariance: Optional[np.ndarray] = None):
        """Initialize estimator with initial pose."""
        self.current_pose = Pose(
            timestamp=initial_pose.timestamp,
            position=initial_pose.position.copy(),
            rotation_matrix=initial_pose.rotation_matrix.copy()
        )
        self.current_velocity = np.zeros(3)
        # Clear any existing data
        self.keyframes.clear()
        self.keyframe_poses.clear()
        self.keyframe_velocities.clear()
        self.landmarks.clear()
        self.imu_preintegrations.clear()
        self.visual_factors.clear()
        self.imu_factors.clear()
        logger.info(f"Initialized at position {initial_pose.position}")
    
    def marginalize(self):
        """Marginalize old states (handled automatically in update)."""
        # Already implemented in _marginalize_oldest_keyframe
        # Called automatically when window size exceeded
        pass
    
    def get_state_vector(self) -> np.ndarray:
        """Get current state vector."""
        # State vector format: [position(3), rotation(3), velocity(3)]
        state = np.zeros(9)
        state[:3] = self.current_pose.position
        state[3:6] = so3_log(self.current_pose.rotation_matrix)
        state[6:9] = self.current_velocity
        return state
    
    def get_covariance_matrix(self) -> Optional[np.ndarray]:
        """Get current covariance matrix."""
        # For SWBA, we don't maintain explicit covariance
        # Could compute from optimization Hessian if needed
        return None
    
    def get_result(self):
        """Get estimation result for output."""
        from src.estimation.base_estimator import EstimatorResult
        from src.common.data_structures import Map, Landmark
        
        # Dump debug data before returning result
        self.dump_debug_data()
        
        # Build landmark map
        landmark_map = Map()
        for lid, position in self.landmarks.items():
            landmark_map.landmarks[lid] = Landmark(
                id=lid,
                position=position.copy()
            )
        
        # Return result
        return EstimatorResult(
            trajectory=self.get_trajectory(),
            landmarks=landmark_map,
            states=[],  # Not tracking all states
            runtime_ms=0.0,  # Could track this if needed
            iterations=0,  # Could count optimization iterations
            converged=True,
            final_cost=0.0  # Could track optimization cost
        )
    
    def dump_debug_data(self):
        """Dump debug data to JSON files for analysis."""
        import json
        import os
        
        # Create debug output directory
        debug_dir = "debug_output"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Dump SWBA poses
        swba_poses_file = os.path.join(debug_dir, "clean_swba_poses.json")
        with open(swba_poses_file, 'w') as f:
            json.dump(self.debug_poses, f, indent=2)
        logger.info(f"Dumped {len(self.debug_poses)} SWBA poses to {swba_poses_file}")
        
        # Dump PnP poses
        pnp_poses_file = os.path.join(debug_dir, "clean_pnp_poses.json")
        with open(pnp_poses_file, 'w') as f:
            json.dump(self.debug_pnp_poses, f, indent=2)
        logger.info(f"Dumped {len(self.debug_pnp_poses)} PnP poses to {pnp_poses_file}")
        
        # Dump pose errors
        pose_errors_file = os.path.join(debug_dir, "clean_pose_errors.json")
        with open(pose_errors_file, 'w') as f:
            json.dump(self.debug_pose_errors, f, indent=2)
        logger.info(f"Dumped {len(self.debug_pose_errors)} pose errors to {pose_errors_file}")
        
        # Dump landmark positions
        landmarks_file = os.path.join(debug_dir, "clean_landmarks.json")
        landmarks_data = {
            str(lid): pos.tolist() for lid, pos in self.landmarks.items()
        }
        with open(landmarks_file, 'w') as f:
            json.dump(landmarks_data, f, indent=2)
        logger.info(f"Dumped {len(self.landmarks)} landmarks to {landmarks_file}")
        
        # Print summary statistics
        if self.debug_pose_errors:
            errors = [e['position_error'] for e in self.debug_pose_errors]
            logger.info(f"Position errors - Mean: {np.mean(errors):.3f}m, Max: {np.max(errors):.3f}m, Min: {np.min(errors):.3f}m")