"""
Simplified Extended Kalman Filter (EKF) for keyframe-based trajectory estimation.

DEPRECATED: This implementation is maintained for backward compatibility only.
Please use GtsamEkfEstimator instead for better performance and stability.

This is a minimal implementation that:
- Processes only preintegrated IMU data between keyframes
- Updates state at keyframes (visual updates optional)
- Maintains covariance propagation
- No SLAM features or landmark mapping
"""

import numpy as np
import warnings
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
import logging

from src.estimation.base_estimator import (
    BaseEstimator, EstimatorState, EstimatorConfig,
    EstimatorResult, EstimatorType
)
from src.common.config import EKFConfig
from src.estimation.imu_integration import IMUState
from src.common.data_structures import (
    CameraFrame, Map, Landmark,
    Trajectory, TrajectoryState, Pose,
    CameraCalibration, IMUCalibration,
    PreintegratedIMUData
)
from src.utils.math_utils import (
    quaternion_to_rotation_matrix, rotation_matrix_to_quaternion,
    skew, so3_exp, project_to_so3
)

logger = logging.getLogger(__name__)


class EKFState:
    """
    EKF state vector representation using SO3.
    
    Nominal state: [position(3), velocity(3), rotation_matrix(3x3), accel_bias(3), gyro_bias(3)]
    
    Error state: [position(3), velocity(3), rotation(3), accel_bias(3), gyro_bias(3)]
    Total dimension: 15 (using SO3 tangent space for rotation)
    """
    
    def __init__(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        rotation_matrix: Optional[np.ndarray] = None,
        quaternion: Optional[np.ndarray] = None,  # For backward compatibility
        accel_bias: np.ndarray = None,
        gyro_bias: np.ndarray = None,
        covariance: Optional[np.ndarray] = None,
        timestamp: float = 0.0
    ):
        """
        Initialize EKF state.
        
        Args:
            position: 3D position in world frame
            velocity: 3D velocity in world frame
            rotation_matrix: SO3 rotation matrix (3x3)
            quaternion: Legacy quaternion support (will be converted)
            accel_bias: Accelerometer bias
            gyro_bias: Gyroscope bias
            covariance: State covariance matrix (15x15)
            timestamp: State timestamp
        """
        self.position = position.copy()
        self.velocity = velocity.copy()
        
        # Handle rotation representation
        if rotation_matrix is not None:
            self.rotation_matrix = project_to_so3(rotation_matrix)
        elif quaternion is not None:
            # Legacy support: convert quaternion to rotation matrix
            self.rotation_matrix = quaternion_to_rotation_matrix(quaternion)
        else:
            self.rotation_matrix = np.eye(3)
        
        self.accel_bias = accel_bias.copy() if accel_bias is not None else np.zeros(3)
        self.gyro_bias = gyro_bias.copy() if gyro_bias is not None else np.zeros(3)
        self.timestamp = timestamp
        
        # Error state covariance (15x15)
        if covariance is None:
            self.covariance = np.eye(15) * 1e-3
        else:
            self.covariance = covariance.copy()
    
    @property
    def quaternion(self):
        """Legacy quaternion access (for backward compatibility)."""
        return rotation_matrix_to_quaternion(self.rotation_matrix)
    
    
    def copy(self) -> 'EKFState':
        """Create a deep copy."""
        return EKFState(
            position=self.position,
            velocity=self.velocity,
            rotation_matrix=self.rotation_matrix,
            accel_bias=self.accel_bias,
            gyro_bias=self.gyro_bias,
            covariance=self.covariance,
            timestamp=self.timestamp
        )
    
    def get_pose(self) -> Pose:
        """Get pose from state."""
        return Pose(
            timestamp=self.timestamp,
            position=self.position.copy(),
            rotation_matrix=self.rotation_matrix.copy()
        )


# Config now imported from src.common.config


class EKFSlam(BaseEstimator):
    """
    Simplified Extended Kalman Filter for trajectory estimation.
    
    Implements:
    - Preintegrated IMU prediction between keyframes
    - State and covariance propagation
    - Minimal update interface (no SLAM)
    """
    
    def __init__(
        self,
        config: EKFConfig,
        camera_calibration: CameraCalibration,
        imu_calibration: Optional[IMUCalibration] = None
    ):
        """
        Initialize EKF-SLAM.
        
        Args:
            config: EKF configuration
            camera_calibration: Camera calibration parameters
            imu_calibration: Optional IMU calibration
        """
        # Issue deprecation warning
        warnings.warn(
            "EKFSlam is deprecated and will be removed in a future version. "
            "Please use GtsamEkfEstimator instead by specifying 'gtsam-ekf' as the estimator type. "
            "The GTSAM implementation provides better numerical stability and performance.",
            DeprecationWarning,
            stacklevel=2
        )
        
        super().__init__(config)
        self.config = config
        self.camera_calib = camera_calibration
        self.imu_calib = imu_calibration
        
        # Initialize state
        self.state: Optional[EKFState] = None
        
        # Store trajectory history
        self.trajectory_history: List[TrajectoryState] = []
        
        # Gravity vector for preintegrated IMU
        self.gravity = np.array([0, 0, -config.gravity_magnitude])
        
        # Simplified: No landmark tracking in this version
        self.landmarks: Dict[int, Landmark] = {}
        
        # Statistics
        self.num_updates = 0
        self.num_outliers = 0
    
    def initialize(
        self,
        initial_pose: Pose,
        initial_covariance: Optional[np.ndarray] = None
    ) -> None:
        """
        Initialize filter state.
        
        Args:
            initial_pose: Initial robot pose
            initial_covariance: Initial covariance matrix (optional)
        """
        # Create initial state
        self.state = EKFState(
            position=initial_pose.position,
            velocity=np.zeros(3),  # Start with zero velocity
            rotation_matrix=initial_pose.rotation_matrix,
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp=initial_pose.timestamp
        )
        
        # Save initial state to trajectory history
        self.trajectory_history.append(TrajectoryState(
            pose=self.state.get_pose(),
            velocity=self.state.velocity.copy()
        ))
        
        # Initialize covariance
        if initial_covariance is not None:
            self.state.covariance = initial_covariance
        else:
            P = np.zeros((15, 15))
            P[0:3, 0:3] = np.eye(3) * self.config.initial_position_std**2
            P[3:6, 3:6] = np.eye(3) * self.config.initial_velocity_std**2
            P[6:9, 6:9] = np.eye(3) * self.config.initial_orientation_std**2
            P[9:12, 9:12] = np.eye(3) * self.config.initial_accel_bias_std**2
            P[12:15, 12:15] = np.eye(3) * self.config.initial_gyro_bias_std**2
            self.state.covariance = P
        
        logger.info(f"EKF initialized at time {initial_pose.timestamp}")
    
    def predict(self, imu_data: PreintegratedIMUData) -> None:
        """
        IMU prediction step using preintegrated measurements.
        
        Propagates state and covariance using preintegrated IMU data.
        
        Args:
            imu_data: Preintegrated IMU data between keyframes
        """
        if self.state is None:
            raise RuntimeError("EKF not initialized")
        
        if not isinstance(imu_data, PreintegratedIMUData):
            raise TypeError("EKF now only accepts PreintegratedIMUData")
        
        self._predict_preintegrated(imu_data)
    
    def _predict_preintegrated(self, preintegrated: PreintegratedIMUData) -> None:
        """
        Predict using preintegrated IMU measurements.
        
        Uses preintegrated delta values to directly update the state without
        iterating through individual measurements.
        
        Args:
            preintegrated: Preintegrated IMU data between keyframes
        """
        # Update position using preintegrated delta
        # The preintegrated values are gravity-free, so we need to add gravity effects
        R_old = self.state.rotation_matrix
        gravity_contribution = 0.5 * self.gravity * preintegrated.dt**2
        self.state.position += (
            self.state.velocity * preintegrated.dt + 
            R_old @ preintegrated.delta_position + 
            gravity_contribution  # Add gravity effect
        )
        
        # Update velocity using preintegrated delta
        # Also add gravity effect to velocity
        self.state.velocity += (
            R_old @ preintegrated.delta_velocity + 
            self.gravity * preintegrated.dt  # Add gravity effect
        )
        
        # Update rotation using preintegrated delta
        # R_new = R_old * delta_R
        self.state.rotation_matrix = R_old @ preintegrated.delta_rotation
        
        # Biases remain constant in the prediction step
        # (they are only updated during optimization/correction)
        
        # Update timestamp
        self.state.timestamp += preintegrated.dt
        
        # Propagate covariance using preintegrated covariance
        # The preintegrated covariance already accounts for noise propagation
        # We need to transform it to the global frame
        
        # State transition matrix for preintegrated measurements
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * preintegrated.dt  # Position depends on velocity
        F[0:3, 6:9] = -R_old @ skew(preintegrated.delta_position)  # Position depends on rotation
        F[3:6, 6:9] = -R_old @ skew(preintegrated.delta_velocity)  # Velocity depends on rotation
        
        # If jacobian w.r.t biases is provided, use it
        if preintegrated.jacobian is not None:
            # The jacobian relates changes in biases to changes in the preintegrated values
            # F should incorporate this dependency
            F[0:3, 9:12] = R_old @ preintegrated.jacobian[0:3, 0:3]  # Position w.r.t accel bias
            F[0:3, 12:15] = R_old @ preintegrated.jacobian[0:3, 3:6]  # Position w.r.t gyro bias
            F[3:6, 9:12] = R_old @ preintegrated.jacobian[3:6, 0:3]  # Velocity w.r.t accel bias
            F[3:6, 12:15] = R_old @ preintegrated.jacobian[3:6, 3:6]  # Velocity w.r.t gyro bias
            F[6:9, 12:15] = preintegrated.jacobian[6:9, 3:6]  # Rotation w.r.t gyro bias
        
        # Propagate covariance
        # P_new = F * P_old * F' + Q_preintegrated
        P = self.state.covariance
        self.state.covariance = F @ P @ F.T + preintegrated.covariance
        
        # Save state to trajectory history after prediction
        self.trajectory_history.append(TrajectoryState(
            pose=self.state.get_pose(),
            velocity=self.state.velocity.copy()
        ))
    
    def update(self, camera_frame: Optional[CameraFrame] = None, landmarks: Optional[Map] = None) -> None:
        """
        Simple update step (minimal implementation).
        
        In this simplified version, we just track that an update occurred.
        Real visual updates would go here if needed.
        
        Args:
            camera_frame: Camera measurements (optional)
            landmarks: Map with landmark positions (optional)
        """
        if self.state is None:
            raise RuntimeError("EKF not initialized")
        
        # Simplified: just increment counter for keyframes
        if camera_frame and camera_frame.is_keyframe:
            self.num_updates += 1
            
            # Optional: Add small correction to demonstrate update
            # In practice, this would be based on visual measurements
            # For now, just maintain state
    
    def _apply_correction(self, dx: np.ndarray) -> None:
        """
        Apply error state correction to nominal state.
        
        Args:
            dx: Error state correction [dp, dv, dtheta, dba, dbg]
        """
        # Position and velocity corrections
        self.state.position += dx[0:3]
        self.state.velocity += dx[3:6]
        
        # Rotation correction using SO3 exponential map
        dtheta = dx[6:9]  # Rotation error in tangent space
        
        # Apply rotation correction: R_new = R @ exp(dtheta)
        delta_R = so3_exp(dtheta)
        self.state.rotation_matrix = self.state.rotation_matrix @ delta_R
        
        # Ensure rotation matrix stays on SO3 manifold
        self.state.rotation_matrix = project_to_so3(self.state.rotation_matrix)
        
        # Bias corrections
        self.state.accel_bias += dx[9:12]
        self.state.gyro_bias += dx[12:15]
    
    
    def optimize(self) -> None:
        """
        Optimization step (not used in EKF).
        
        EKF is a filtering approach, not batch optimization.
        """
        pass  # No batch optimization in EKF
    
    def get_state(self) -> EstimatorState:
        """
        Get current estimator state.
        
        Returns:
            Current state with pose and covariance
        """
        if self.state is None:
            raise RuntimeError("EKF not initialized")
        
        return EstimatorState(
            timestamp=self.state.timestamp,
            robot_pose=self.state.get_pose(),
            robot_velocity=self.state.velocity.copy(),
            robot_covariance=self.state.covariance[0:6, 0:6],  # Position and velocity
            landmarks=self.landmarks
        )
    
    def get_trajectory(self) -> Trajectory:
        """
        Get estimated trajectory.
        
        Returns:
            Trajectory with all historical states
        """
        trajectory = Trajectory()
        
        # Add all states from history
        for state in self.trajectory_history:
            trajectory.add_state(state)
        
        return trajectory
    
    def get_map(self) -> Map:
        """
        Get estimated map.
        
        Returns:
            Map with tracked landmarks
        """
        map_data = Map()
        for landmark in self.landmarks.values():
            map_data.add_landmark(landmark)
        return map_data
    
    def get_result(self) -> EstimatorResult:
        """
        Get complete estimation result.
        
        Returns:
            Result with trajectory, landmarks, and statistics
        """
        return EstimatorResult(
            trajectory=self.get_trajectory(),
            landmarks=self.get_map(),
            states=[self.get_state()],
            runtime_ms=0.0,  # Would need timing
            iterations=self.num_updates,
            converged=True,  # EKF doesn't have convergence criterion
            final_cost=0.0,  # No cost function in EKF
            metadata={
                "num_updates": self.num_updates,
                "num_outliers": self.num_outliers,
                "final_timestamp": self.state.timestamp if self.state else 0.0
            }
        )
    
    def reset(self) -> None:
        """Reset filter state."""
        self.state = None
        self.landmarks = {}
        self.trajectory_history = []
        self.num_updates = 0
        self.num_outliers = 0
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get state vector.
        
        Returns:
            State vector [position, velocity, quaternion, accel_bias, gyro_bias]
        """
        if self.state is None:
            return np.zeros(16)
        
        return np.concatenate([
            self.state.position,
            self.state.velocity,
            self.state.quaternion,
            self.state.accel_bias,
            self.state.gyro_bias
        ])
    
    def get_covariance_matrix(self) -> Optional[np.ndarray]:
        """
        Get covariance matrix.
        
        Returns:
            15x15 error state covariance matrix
        """
        if self.state is None:
            return None
        
        return self.state.covariance.copy()
    
    def marginalize(self) -> None:
        """
        Marginalize old states.
        
        EKF doesn't maintain a sliding window, so this is a no-op.
        """
        pass  # No marginalization needed in standard EKF