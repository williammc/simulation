"""
Extended Kalman Filter (EKF) for Visual-Inertial SLAM.

Implements an EKF-based SLAM estimator with IMU prediction and camera updates.
State vector includes robot pose, velocity, and IMU biases.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from src.estimation.base_estimator import (
    BaseEstimator, EstimatorState, EstimatorConfig,
    EstimatorResult, EstimatorType
)
from src.estimation.imu_integration import (
    IMUIntegrator, IMUState, IntegrationMethod,
    compute_imu_jacobian
)
from src.estimation.camera_model import (
    CameraMeasurementModel, ReprojectionError
)
from src.common.data_structures import (
    IMUMeasurement, CameraFrame, Map, Landmark,
    Trajectory, TrajectoryState, Pose,
    CameraCalibration, IMUCalibration
)
from src.utils.math_utils import (
    quaternion_to_rotation_matrix, rotation_matrix_to_quaternion,
    skew, so3_exp, so3_log, project_to_so3
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
    
    def to_imu_state(self) -> IMUState:
        """Convert to IMU state for integration."""
        return IMUState(
            position=self.position,
            velocity=self.velocity,
            rotation_matrix=self.rotation_matrix,
            accel_bias=self.accel_bias,
            gyro_bias=self.gyro_bias,
            timestamp=self.timestamp
        )
    
    def from_imu_state(self, imu_state: IMUState):
        """Update from IMU state."""
        self.position = imu_state.position.copy()
        self.velocity = imu_state.velocity.copy()
        self.rotation_matrix = imu_state.rotation_matrix.copy()
        self.accel_bias = imu_state.accel_bias.copy()
        self.gyro_bias = imu_state.gyro_bias.copy()
        self.timestamp = imu_state.timestamp
    
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


@dataclass
class EKFConfig(EstimatorConfig):
    """
    EKF-specific configuration.
    
    Extends base estimator config with EKF parameters.
    """
    # Initial uncertainties
    initial_position_std: float = 0.1  # meters
    initial_velocity_std: float = 0.1  # m/s
    initial_rotation_std: float = 0.01  # radians
    initial_accel_bias_std: float = 0.01  # m/s^2
    initial_gyro_bias_std: float = 0.001  # rad/s
    
    # Process noise
    accel_noise_density: float = 0.01  # m/s^2/sqrt(Hz)
    gyro_noise_density: float = 0.001  # rad/s/sqrt(Hz)
    accel_bias_random_walk: float = 0.001  # m/s^3/sqrt(Hz)
    gyro_bias_random_walk: float = 0.0001  # rad/s^2/sqrt(Hz)
    
    # Measurement noise
    pixel_noise_std: float = 1.0  # pixels
    
    # Outlier rejection
    chi2_threshold: float = 5.991  # 95% confidence for 2 DOF
    max_iterations: int = 5  # Max iterations for outlier rejection
    
    # Integration method
    integration_method: str = "euler"  # euler, rk4, midpoint
    
    # Gravity
    gravity_magnitude: float = 9.81  # m/s^2
    
    def __post_init__(self):
        """Set estimator type."""
        self.estimator_type = EstimatorType.EKF


class EKFSlam(BaseEstimator):
    """
    Extended Kalman Filter for Visual-Inertial SLAM.
    
    Implements:
    - IMU-based prediction with error state formulation
    - Camera measurement updates with outlier rejection
    - Online calibration of IMU biases
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
        super().__init__(config)
        self.config = config
        self.camera_calib = camera_calibration
        self.imu_calib = imu_calibration
        
        # Initialize state
        self.state: Optional[EKFState] = None
        
        # Camera model
        self.camera_model = CameraMeasurementModel(camera_calibration)
        
        # IMU integrator
        gravity = np.array([0, 0, -config.gravity_magnitude])
        method_map = {
            "euler": IntegrationMethod.EULER,
            "rk4": IntegrationMethod.RK4,
            "midpoint": IntegrationMethod.MIDPOINT
        }
        self.imu_integrator = IMUIntegrator(
            gravity=gravity,
            method=method_map.get(config.integration_method, IntegrationMethod.EULER)
        )
        
        # Landmarks
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
        
        # Initialize covariance
        if initial_covariance is not None:
            self.state.covariance = initial_covariance
        else:
            P = np.zeros((15, 15))
            P[0:3, 0:3] = np.eye(3) * self.config.initial_position_std**2
            P[3:6, 3:6] = np.eye(3) * self.config.initial_velocity_std**2
            P[6:9, 6:9] = np.eye(3) * self.config.initial_rotation_std**2
            P[9:12, 9:12] = np.eye(3) * self.config.initial_accel_bias_std**2
            P[12:15, 12:15] = np.eye(3) * self.config.initial_gyro_bias_std**2
            self.state.covariance = P
        
        logger.info(f"EKF initialized at time {initial_pose.timestamp}")
    
    def predict(self, imu_measurements: List[IMUMeasurement], dt: float) -> None:
        """
        IMU prediction step.
        
        Propagates state and covariance using IMU measurements.
        
        Args:
            imu_measurements: List of IMU measurements
            dt: Total time step
        """
        if self.state is None:
            raise RuntimeError("EKF not initialized")
        
        if not imu_measurements:
            return
        
        # Convert to IMU state
        imu_state = self.state.to_imu_state()
        
        # Integrate IMU measurements
        for i, measurement in enumerate(imu_measurements):
            # Compute time step
            if i == 0:
                meas_dt = measurement.timestamp - self.state.timestamp
            else:
                meas_dt = measurement.timestamp - imu_measurements[i-1].timestamp
            
            if meas_dt <= 0:
                continue
            
            # Compute Jacobians before integration
            F, G = compute_imu_jacobian(
                imu_state, measurement, meas_dt,
                self.imu_integrator.gravity
            )
            
            # Integrate state
            imu_state = self.imu_integrator.integrate(
                imu_state, measurement, meas_dt
            )
            
            # Update covariance
            # P = F * P * F' + G * Q * G'
            P = self.state.covariance
            
            # Process noise covariance
            Q = np.zeros((12, 12))
            Q[0:3, 0:3] = np.eye(3) * self.config.accel_noise_density**2 / meas_dt
            Q[3:6, 3:6] = np.eye(3) * self.config.gyro_noise_density**2 / meas_dt
            Q[6:9, 6:9] = np.eye(3) * self.config.accel_bias_random_walk**2 * meas_dt
            Q[9:12, 9:12] = np.eye(3) * self.config.gyro_bias_random_walk**2 * meas_dt
            
            # Propagate covariance
            P = F @ P @ F.T + G @ Q @ G.T
            self.state.covariance = P
        
        # Update state from IMU integration
        self.state.from_imu_state(imu_state)
    
    def update(self, camera_frame: CameraFrame, landmarks: Map) -> None:
        """
        Camera measurement update.
        
        Updates state using camera observations with outlier rejection.
        
        Args:
            camera_frame: Camera measurements
            landmarks: Map with landmark positions
        """
        if self.state is None:
            raise RuntimeError("EKF not initialized")
        
        # Get current pose
        current_pose = self.state.get_pose()
        
        # Process each observation
        valid_observations = []
        H_list = []  # Measurement Jacobians
        residuals = []
        
        for obs in camera_frame.observations:
            # Get landmark
            if obs.landmark_id not in self.landmarks:
                # Try to get from provided map
                if obs.landmark_id in landmarks.landmarks:
                    self.landmarks[obs.landmark_id] = landmarks.landmarks[obs.landmark_id]
                else:
                    continue
            
            landmark = self.landmarks[obs.landmark_id]
            
            # Compute reprojection error and Jacobians
            error = self.camera_model.compute_reprojection_error(
                obs, landmark, current_pose
            )
            
            # Check chi-squared test for outlier rejection
            if self._chi2_test(error.residual, obs.landmark_id):
                valid_observations.append(obs)
                
                # Build measurement Jacobian for error state
                # H maps error state [dp, dv, dtheta, dba, dbg] to pixel error
                H = np.zeros((2, 15))
                H[:, 0:3] = error.jacobian_pose[:, 3:6]  # Position
                H[:, 6:9] = error.jacobian_pose[:, 0:3]  # Rotation
                
                H_list.append(H)
                residuals.append(error.residual)
            else:
                self.num_outliers += 1
        
        if not valid_observations:
            return
        
        # Stack measurements
        H = np.vstack(H_list)
        z = np.concatenate(residuals)
        
        # Measurement noise covariance
        R = np.eye(len(z)) * self.config.pixel_noise_std**2
        
        # Kalman gain
        # K = P * H' * (H * P * H' + R)^-1
        P = self.state.covariance
        S = H @ P @ H.T + R  # Innovation covariance
        K = P @ H.T @ np.linalg.inv(S)
        
        # State update (error state)
        dx = K @ z
        
        # Apply correction to state
        self._apply_correction(dx)
        
        # Covariance update
        # P = (I - K * H) * P
        I = np.eye(15)
        self.state.covariance = (I - K @ H) @ P
        
        self.num_updates += 1
    
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
    
    def _chi2_test(self, residual: np.ndarray, landmark_id: int) -> bool:
        """
        Chi-squared test for outlier rejection.
        
        Args:
            residual: 2D pixel residual
            landmark_id: Landmark ID for tracking
        
        Returns:
            True if observation passes test (inlier)
        """
        # Compute Mahalanobis distance
        # For now, use simple threshold on pixel error
        error_norm = np.linalg.norm(residual)
        
        # Chi-squared test (2 DOF)
        threshold = np.sqrt(self.config.chi2_threshold) * self.config.pixel_noise_std
        
        return error_norm < threshold
    
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
            Trajectory with single current state
        """
        trajectory = Trajectory()
        
        if self.state is not None:
            traj_state = TrajectoryState(
                pose=self.state.get_pose(),
                velocity=self.state.velocity.copy()
            )
            trajectory.add_state(traj_state)
        
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