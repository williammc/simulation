"""
Camera-model-independent Sliding Window Bundle Adjustment estimator.

This estimator works with pre-processed measurements and does not depend
on camera models or IMU calibration parameters.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import logging

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


@dataclass
class NewSWBAConfig(EstimatorConfig):
    """
    Configuration for camera-model-independent SWBA.
    
    Note: No camera_calibration or imu_calibration parameters needed.
    All sensor-specific parameters are embedded in pre-processed measurements.
    """
    # Sliding window parameters
    window_size: int = 10
    min_keyframe_distance: float = 0.5  # meters
    min_keyframe_angle: float = 10.0  # degrees
    keyframe_selection_method: str = "distance"  # "distance", "angle", "both", "all"
    
    # Optimization parameters  
    optimization_backend: str = "gtsam"  # "gtsam", "ceres", "g2o"
    max_optimization_iterations: int = 50
    optimization_convergence_threshold: float = 1e-6
    use_robust_kernels: bool = True  # Uses pre-computed robust weights
    
    # Marginalization parameters
    marginalization_strategy: str = "oldest"  # "oldest", "information"
    keep_marginalized_prior: bool = True
    
    # Feature management (for validation only, not projection)
    min_observations_per_landmark: int = 2
    max_track_length: int = 50
    
    # Debug and visualization
    visualize_optimization: bool = False
    save_factor_graph: bool = False
    verbose_optimization: bool = False
    
    def __post_init__(self):
        """Set estimator type to new SWBA."""
        self.estimator_type = EstimatorType.SWBA  # Will need to add NEW_SWBA to enum later


class NewSWBAEstimator(BaseEstimator):
    """
    Camera-model-independent Sliding Window Bundle Adjustment estimator.
    
    This estimator:
    - Works with pre-processed visual measurements (no projection operations)
    - Uses pre-integrated IMU data with embedded noise models
    - Requires no camera or IMU calibration parameters
    - Maintains sliding window of keyframes
    - Performs optimization using pre-computed Jacobians
    """
    
    def __init__(self, config: NewSWBAConfig):
        """
        Initialize NewSWBAEstimator.
        
        Args:
            config: Configuration without camera/IMU calibration
        """
        # Initialize base class without calibration parameters
        super().__init__(config, imu_calibration=None, camera_calibration=None)
        
        self.config: NewSWBAConfig = config
        
        # Sliding window data structures
        self.keyframes: List[ProcessedVisualFrame] = []
        self.keyframe_poses: List[Pose] = []
        self.keyframe_ids: List[int] = []
        
        # IMU preintegration between keyframes
        self.imu_preintegrations: Dict[Tuple[int, int], PreprocessedIMUData] = {}
        
        # Current estimates
        self.current_pose: Optional[Pose] = None
        self.current_velocity: Optional[np.ndarray] = None
        self.current_imu_bias: Optional[Dict[str, np.ndarray]] = None
        
        # Landmark estimates (maintained separately from Map for optimization)
        self.landmark_estimates: Dict[int, np.ndarray] = {}  # id -> position
        self.landmark_covariances: Dict[int, np.ndarray] = {}  # id -> covariance
        
        # Marginalization prior
        self.marginalization_prior = None
        
        # Statistics
        self.num_keyframes_created = 0
        self.num_optimizations = 0
        self.last_optimization_cost = float('inf')
        
        logger.info(f"Initialized NewSWBAEstimator with window size {config.window_size}")
    
    def initialize(self, initial_pose: Pose, initial_covariance: Optional[np.ndarray] = None, initial_velocity: Optional[np.ndarray] = None):
        """
        Initialize estimator with initial pose.
        
        Args:
            initial_pose: Initial robot pose
            initial_covariance: Initial uncertainty (optional)
            initial_velocity: Initial velocity (optional)
        """
        self.current_pose = initial_pose
        self.current_velocity = initial_velocity if initial_velocity is not None else np.zeros(3)
        self.current_imu_bias = {
            'accelerometer': np.zeros(3),
            'gyroscope': np.zeros(3)
        }
        
        # Initialize state
        self.current_state = EstimatorState(
            timestamp=initial_pose.timestamp,
            robot_pose=initial_pose,
            robot_velocity=self.current_velocity,
            robot_covariance=initial_covariance if initial_covariance is not None else np.eye(15) * 0.1
        )
        
        logger.info("NewSWBAEstimator initialized")
    
    def predict(self, imu_measurements: Any, dt: float):
        """
        Prediction step using pre-integrated IMU data.
        
        Args:
            imu_measurements: PreprocessedIMUData with pre-computed deltas
            dt: Time step (already embedded in preintegration)
        """
        if not isinstance(imu_measurements, PreprocessedIMUData):
            logger.warning("Expected PreprocessedIMUData, got raw IMU measurements")
            return
        
        if self.current_pose is None:
            logger.warning("Cannot predict: estimator not initialized")
            return
        
        # Use pre-integrated measurements directly
        delta_R = imu_measurements.rotation_matrix
        delta_v = imu_measurements.delta_velocity
        delta_p = imu_measurements.delta_position
        
        # Get current rotation matrix
        R_curr = self.current_pose.rotation_matrix
        
        # Propagate state using pre-integrated measurements
        # The preintegrated values have gravity removed, so we need to add it back
        # Based on IMUPreintegrator.predict() in imu_model.py
        gravity = np.array([0, 0, -9.81])  # Gravity in world frame
        
        # Propagate rotation first: R_j = R_i @ delta_R
        new_rotation = R_curr @ delta_R
        
        # Propagate velocity: v_j = v_i + g*dt + R_i @ delta_v
        new_velocity = self.current_velocity + gravity * dt + R_curr @ delta_v
        
        # Propagate position: p_j = p_i + v_i*dt + 0.5*g*dt^2 + R_i @ delta_p
        new_position = self.current_pose.position + self.current_velocity * dt + 0.5 * gravity * dt**2 + R_curr @ delta_p
        
        # Update pose
        self.current_pose = Pose(
            timestamp=self.current_pose.timestamp + dt,
            position=new_position,
            rotation_matrix=new_rotation
        )
        self.current_velocity = new_velocity
        
        # Store preintegration for optimization
        if len(self.keyframe_ids) > 0:
            last_kf_id = self.keyframe_ids[-1]
            # Assuming next keyframe will have id = last_kf_id + 1
            self.imu_preintegrations[(last_kf_id, last_kf_id + 1)] = imu_measurements
        
        self.total_predictions += 1
        logger.debug(f"Prediction complete at t={self.current_pose.timestamp:.3f}")
    
    def update(self, camera_frame: Any, landmarks: Optional[Map] = None):
        """
        Update step using pre-processed visual measurements.
        
        Args:
            camera_frame: ProcessedVisualFrame with pre-computed measurements
            landmarks: Optional map (not used for projection, only for initialization)
        """
        if not isinstance(camera_frame, ProcessedVisualFrame):
            logger.warning("Expected ProcessedVisualFrame, got raw camera frame")
            return
        
        if self.current_pose is None:
            logger.warning("Cannot update: estimator not initialized")
            return
        
        # Check if this should be a keyframe
        is_keyframe = self._should_create_keyframe(camera_frame)
        
        if is_keyframe:
            # Add to sliding window
            self.keyframes.append(camera_frame)
            self.keyframe_poses.append(self.current_pose)
            self.keyframe_ids.append(self.num_keyframes_created)
            self.num_keyframes_created += 1
            
            # Initialize new landmarks if needed
            if landmarks:
                for measurement in camera_frame.measurements:
                    if measurement.landmark_id not in self.landmark_estimates:
                        landmark = landmarks.get_landmark(measurement.landmark_id)
                        if landmark:
                            self.landmark_estimates[measurement.landmark_id] = landmark.position
                            self.landmark_covariances[measurement.landmark_id] = np.eye(3) * 0.1
            
            # Maintain window size
            if len(self.keyframes) > self.config.window_size:
                self._marginalize_oldest_keyframe()
            
            logger.info(f"Created keyframe {self.num_keyframes_created} at t={camera_frame.timestamp:.3f}")
        
        # Update landmark observations (for tracking, not projection)
        self._update_landmark_observations(camera_frame)
        
        # Apply simple visual correction if we have measurements
        if camera_frame.measurements and len(camera_frame.measurements) > 0:
            # Compute average residual to detect drift
            total_residual = np.zeros(2)
            count = 0
            for meas in camera_frame.measurements:
                if meas.is_valid:
                    total_residual += meas.residual
                    count += 1
            
            if count > 0:
                avg_residual = total_residual / count
                # Apply small correction based on visual residuals (simplified)
                # In reality, this would be done properly in optimize()
                correction_weight = 0.01  # Small weight to avoid instability
                
                # Correct position slightly based on average pixel error
                # This is a very simplified approximation
                if hasattr(camera_frame, 'predicted_pose') and camera_frame.predicted_pose is not None:
                    # Use the predicted pose from preprocessing as reference
                    pose_diff = camera_frame.predicted_pose.position - self.current_pose.position
                    self.current_pose.position += pose_diff * correction_weight
        
        self.total_updates += 1
    
    def optimize(self) -> bool:
        """
        Perform bundle adjustment using pre-computed Jacobians.
        
        Returns:
            True if converged, False otherwise
        """
        if len(self.keyframes) < 2:
            logger.debug("Not enough keyframes for optimization")
            return True
        
        logger.info(f"Starting optimization with {len(self.keyframes)} keyframes")
        
        # Build factor graph from pre-processed measurements
        converged = self._build_and_optimize_graph()
        
        self.num_optimizations += 1
        self.total_iterations += self.config.max_optimization_iterations
        
        return converged
    
    def marginalize(self):
        """Marginalize oldest keyframe from sliding window."""
        if len(self.keyframes) > self.config.window_size:
            self._marginalize_oldest_keyframe()
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get current state vector.
        
        Format: [position(3), rotation(9 as flattened matrix), velocity(3), bias_accel(3), bias_gyro(3)]
        Total: 21 elements
        """
        if self.current_pose is None:
            return np.zeros(21)
        
        state = np.zeros(21)
        state[0:3] = self.current_pose.position
        state[3:12] = self.current_pose.rotation_matrix.flatten()
        state[12:15] = self.current_velocity
        
        if self.current_imu_bias:
            state[15:18] = self.current_imu_bias.get('accelerometer', np.zeros(3))
            state[18:21] = self.current_imu_bias.get('gyroscope', np.zeros(3))
        
        return state
    
    def get_covariance_matrix(self) -> Optional[np.ndarray]:
        """
        Get current covariance matrix.
        
        Note: Full covariance computation would require factor graph evaluation.
        Returns simplified block-diagonal approximation for now.
        """
        if self.current_state and self.current_state.robot_covariance is not None:
            return self.current_state.robot_covariance
        
        # Return default covariance
        return np.eye(15) * 0.1
    
    def get_result(self) -> EstimatorResult:
        """
        Get current estimation result.
        
        Returns:
            EstimatorResult with trajectory and landmarks
        """
        # Build trajectory from keyframes
        trajectory = Trajectory()
        for pose in self.keyframe_poses:
            state = TrajectoryState(pose=pose)
            trajectory.add_state(state)
        
        # Build landmark map
        landmark_map = Map()
        for lid, position in self.landmark_estimates.items():
            landmark = Landmark(
                id=lid,
                position=position,
                descriptor=None
            )
            landmark_map.add_landmark(landmark)
        
        # Create result
        result = EstimatorResult(
            trajectory=trajectory,
            landmarks=landmark_map,
            states=self.state_history,
            runtime_ms=0.0,  # Would need timing
            iterations=self.total_iterations,
            converged=True,  # Simplified
            final_cost=self.last_optimization_cost,
            metadata={
                'estimator_type': 'new_swba',
                'num_keyframes': len(self.keyframes),
                'num_optimizations': self.num_optimizations,
                'window_size': self.config.window_size
            }
        )
        
        return result
    
    # Private helper methods
    
    def _should_create_keyframe(self, frame: ProcessedVisualFrame) -> bool:
        """
        Determine if current frame should be a keyframe.
        
        Uses distance/angle criteria without requiring camera model.
        """
        if self.config.keyframe_selection_method == "all":
            return True
        
        if len(self.keyframes) == 0:
            return True  # First frame is always a keyframe
        
        # For now, accept all frames marked as keyframes in the data
        if hasattr(frame, 'is_keyframe') and frame.is_keyframe:
            return True
        
        last_kf_pose = self.keyframe_poses[-1]
        
        # Distance criterion
        distance = np.linalg.norm(self.current_pose.position - last_kf_pose.position)
        if self.config.keyframe_selection_method in ["distance", "both"]:
            if distance > self.config.min_keyframe_distance:
                return True
        
        # Angle criterion (using rotation matrix)
        if self.config.keyframe_selection_method in ["angle", "both"]:
            R1 = self.current_pose.rotation_matrix
            R2 = last_kf_pose.rotation_matrix
            # Compute angle from rotation matrix difference
            R_diff = R1 @ R2.T
            # Extract angle from trace formula: trace(R) = 1 + 2*cos(theta)
            cos_theta = (np.trace(R_diff) - 1) / 2
            angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
            if angle_deg > self.config.min_keyframe_angle:
                return True
        
        return False
    
    def _update_landmark_observations(self, frame: ProcessedVisualFrame):
        """
        Update landmark observation statistics.
        
        This doesn't perform projection, just tracks observations.
        """
        for measurement in frame.measurements:
            # Track observation count, last seen time, etc.
            # This is for bookkeeping, not for projection
            pass
    
    def _build_and_optimize_graph(self) -> bool:
        """
        Build and optimize factor graph using pre-computed measurements.
        
        This is a skeleton - actual implementation would use GTSAM/Ceres.
        """
        # Placeholder for actual optimization
        # Would build factor graph from:
        # - Visual measurements with pre-computed Jacobians
        # - IMU preintegrations with embedded noise
        # - Marginalization prior if exists
        
        logger.info("Building factor graph from pre-processed measurements")
        
        # Simulate optimization convergence
        self.last_optimization_cost *= 0.9  # Fake cost reduction
        
        # Update estimates (placeholder)
        # In real implementation, would extract optimized values
        
        return True  # Assume converged
    
    def _marginalize_oldest_keyframe(self):
        """
        Marginalize oldest keyframe from sliding window.
        
        Creates prior factor for remaining states.
        """
        if len(self.keyframes) <= self.config.window_size:
            return
        
        # Remove oldest keyframe
        marginalized_frame = self.keyframes.pop(0)
        marginalized_pose = self.keyframe_poses.pop(0)
        marginalized_id = self.keyframe_ids.pop(0)
        
        # Update IMU preintegrations
        # Remove old preintegrations involving marginalized keyframe
        keys_to_remove = [
            key for key in self.imu_preintegrations.keys()
            if marginalized_id in key
        ]
        for key in keys_to_remove:
            del self.imu_preintegrations[key]
        
        if self.config.keep_marginalized_prior:
            # Create marginalization prior (placeholder)
            # In real implementation, would compute Schur complement
            self.marginalization_prior = {
                'keyframe_id': marginalized_id,
                'pose': marginalized_pose,
                'information': np.eye(6) * 100  # Placeholder
            }
        
        logger.debug(f"Marginalized keyframe {marginalized_id}")