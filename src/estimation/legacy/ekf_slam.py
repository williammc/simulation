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
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from src.estimation.base_estimator import (
    BaseEstimator, EstimatorState, EstimatorConfig,
    EstimatorResult, EstimatorType
)
from src.common.config import EKFConfig
from src.simulation.imu_integration import IMUState
from src.common.data_structures import (
    CameraFrame, Map, Landmark,
    Trajectory, TrajectoryState, Pose,
    CameraCalibration, IMUCalibration,
    PreintegratedIMUData
)
from src.utils.math_utils import (
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
        imu_calibration: Optional[IMUCalibration] = None,
        use_preintegrated_imu: Optional[bool] = None
    ):
        """
        Initialize EKF-SLAM.
        
        Args:
            config: EKF configuration
            camera_calibration: Camera calibration parameters
            imu_calibration: Optional IMU calibration
            use_preintegrated_imu: Override config to use raw (False) or preintegrated (True) IMU
        """
        # Issue deprecation warning
        warnings.warn(
            "EKFSlam is deprecated and will be removed in a future version. "
            "Please use NewSWBAEstimator instead by specifying 'new-swba' as the estimator type. "
            "The new implementation provides better performance and is camera-model independent.",
            DeprecationWarning,
            stacklevel=2
        )
        
        super().__init__(config)
        self.config = config
        self.camera_calib = camera_calibration
        self.imu_calib = imu_calibration
        
        # Determine whether to use raw or preintegrated IMU
        if use_preintegrated_imu is not None:
            self.use_preintegrated_imu = use_preintegrated_imu
        else:
            self.use_preintegrated_imu = getattr(config, 'use_preintegrated_imu', True)
        
        # Initialize state
        self.state: Optional[EKFState] = None
        
        # Store trajectory history
        self.trajectory_history: List[TrajectoryState] = []
        
        # Gravity vector for IMU processing
        self.gravity = np.array([0, 0, -config.gravity_magnitude])
        
        # Simplified: No landmark tracking in this version
        self.landmarks: Dict[int, Landmark] = {}
        
        # Statistics
        self.num_updates = 0
        self.num_outliers = 0
    
    def initialize(
        self,
        initial_pose: Pose,
        initial_covariance: Optional[np.ndarray] = None,
        initial_velocity: Optional[np.ndarray] = None
    ) -> None:
        """
        Initialize filter state.
        
        Args:
            initial_pose: Initial robot pose
            initial_covariance: Initial covariance matrix (optional)
            initial_velocity: Initial velocity (optional, defaults to zero)
        """
        # Use provided velocity or default to zero
        if initial_velocity is not None:
            velocity = initial_velocity.copy()
        else:
            velocity = np.zeros(3)
            # Warn if initializing with zero velocity when it might be wrong
            if hasattr(initial_pose, 'timestamp') and initial_pose.timestamp == 0.0:
                logger.warning("Initializing EKF with zero velocity - consider providing initial velocity if object is moving")
        
        # Create initial state
        self.state = EKFState(
            position=initial_pose.position,
            velocity=velocity,
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
    
    def predict(self, imu_data, dt: Optional[float] = None) -> None:
        """
        IMU prediction step using either raw or preintegrated measurements.
        
        Propagates state and covariance using IMU data.
        
        Args:
            imu_data: Either PreintegratedIMUData or list of IMUMeasurement objects
            dt: Time step (used for raw IMU processing)
        """
        if self.state is None:
            raise RuntimeError("EKF not initialized")
        
        # Handle None or empty data
        if imu_data is None:
            return
        
        # Check if we're using preintegrated or raw IMU
        if self.use_preintegrated_imu:
            if not isinstance(imu_data, PreintegratedIMUData):
                # Try to handle as list of preintegrated (for compatibility)
                if isinstance(imu_data, list) and len(imu_data) > 0:
                    if isinstance(imu_data[0], PreintegratedIMUData):
                        # Process each preintegrated measurement
                        for preint in imu_data:
                            self._predict_preintegrated(preint)
                        return
                raise TypeError("Expected PreintegratedIMUData when use_preintegrated_imu=True")
            self._predict_preintegrated(imu_data)
        else:
            # Raw IMU processing
            from src.common.data_structures import IMUMeasurement
            if isinstance(imu_data, list) and all(isinstance(m, IMUMeasurement) for m in imu_data):
                self._predict_raw_imu(imu_data)
            else:
                raise TypeError("Expected list of IMUMeasurement when use_preintegrated_imu=False")
    
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
        # Check for duplicate timestamps
        if not self.trajectory_history or self.state.timestamp > self.trajectory_history[-1].pose.timestamp:
            self.trajectory_history.append(TrajectoryState(
                pose=self.state.get_pose(),
                velocity=self.state.velocity.copy()
            ))
    
    def _predict_raw_imu(self, imu_measurements: List) -> None:
        """
        Predict using raw IMU measurements.
        
        Process each IMU measurement individually, integrating the accelerometer
        and gyroscope readings with proper gravity compensation.
        
        Args:
            imu_measurements: List of raw IMU measurements
        """
        from src.common.data_structures import IMUMeasurement
        
        if not imu_measurements:
            return
        
        # Process each IMU measurement sequentially
        for i, imu in enumerate(imu_measurements):
            if not isinstance(imu, IMUMeasurement):
                continue
                
            # Compute dt (time between consecutive measurements)
            if i == 0:
                # For first measurement, use a small default dt
                dt = 0.005  # 5ms default
            else:
                dt = imu.timestamp - imu_measurements[i-1].timestamp
            
            if dt <= 0:
                continue
            
            # Extract measurements and correct for biases
            accel = imu.accelerometer - self.state.accel_bias  # Remove accelerometer bias
            gyro = imu.gyroscope - self.state.gyro_bias  # Remove gyroscope bias
            
            # Current state
            R = self.state.rotation_matrix
            v = self.state.velocity
            p = self.state.position
            
            # --- Rotation update ---
            # Integrate angular velocity to update rotation
            # Use exponential map: R_new = R * exp(gyro * dt)
            omega_dt = gyro * dt
            delta_R = so3_exp(omega_dt)
            R_new = R @ delta_R
            
            # --- Velocity update ---
            # Convert specific force to acceleration in world frame
            # IMU measures: f = a - g (in body frame)
            # For better accuracy, use mid-point rotation (average of R and R_new)
            # This is more accurate for high-frequency rotations
            R_mid = R @ so3_exp(omega_dt * 0.5)
            accel_world = R_mid @ accel + self.gravity
            v_new = v + accel_world * dt
            
            # --- Position update ---
            # Use trapezoidal integration for better accuracy
            p_new = p + v * dt + 0.5 * accel_world * dt**2
            
            # Update state
            self.state.rotation_matrix = project_to_so3(R_new)
            self.state.velocity = v_new
            self.state.position = p_new
            
            # --- Covariance propagation ---
            # State transition matrix
            F = np.eye(15)
            F[0:3, 3:6] = np.eye(3) * dt  # Position depends on velocity
            F[0:3, 6:9] = -R @ skew(accel) * dt  # Position depends on rotation
            F[3:6, 6:9] = -R @ skew(accel) * dt  # Velocity depends on rotation
            
            # Process noise matrix
            Q = np.zeros((15, 15))
            
            # Position process noise (small drift)
            Q[0:3, 0:3] = np.eye(3) * 1e-6 * dt**2
            
            # Velocity process noise (from accelerometer noise)
            if self.imu_calib:
                accel_noise_var = self.imu_calib.accelerometer_noise_density**2 * (1/dt)
            else:
                accel_noise_var = self.config.accel_noise_density**2 * (1/dt)
            Q[3:6, 3:6] = np.eye(3) * accel_noise_var * dt**2
            
            # Rotation process noise (from gyroscope noise)
            if self.imu_calib:
                gyro_noise_var = self.imu_calib.gyroscope_noise_density**2 * (1/dt)
            else:
                gyro_noise_var = self.config.gyro_noise_density**2 * (1/dt)
            Q[6:9, 6:9] = np.eye(3) * gyro_noise_var * dt**2
            
            # Bias random walk
            Q[9:12, 9:12] = np.eye(3) * self.config.accel_bias_random_walk * dt  # Accel bias
            Q[12:15, 12:15] = np.eye(3) * self.config.gyro_bias_random_walk * dt  # Gyro bias
            
            # Propagate covariance
            P = self.state.covariance
            self.state.covariance = F @ P @ F.T + Q
            
            # Update timestamp
            self.state.timestamp = imu.timestamp
        
        # Save state to trajectory history after processing all measurements
        # Only save if timestamp has advanced to avoid duplicate timestamps
        if imu_measurements and (not self.trajectory_history or 
                                  self.state.timestamp > self.trajectory_history[-1].pose.timestamp):
            self.trajectory_history.append(TrajectoryState(
                pose=self.state.get_pose(),
                velocity=self.state.velocity.copy()
            ))
    
    def update(self, camera_frame: Optional[CameraFrame] = None, landmarks: Optional[Map] = None) -> None:
        """
        Measurement update step using camera observations of landmarks.
        
        Performs EKF update using observed landmarks to correct the state estimate.
        
        Args:
            camera_frame: Camera measurements with observations
            landmarks: Map with known landmark positions
        """
        if self.state is None:
            raise RuntimeError("EKF not initialized")
        
        # Skip if no observations
        if not camera_frame or not camera_frame.observations or not landmarks:
            return
        
        # Only process keyframes
        if camera_frame and camera_frame.is_keyframe:
            self.num_updates += 1
            
            # Process each observation
            for obs in camera_frame.observations:
                # Skip if landmark not in map
                if obs.landmark_id not in landmarks.landmarks:
                    continue
                
                landmark = landmarks.landmarks[obs.landmark_id]
                
                # Predict measurement (project landmark to camera)
                predicted_pixel, jacobian_H = self._predict_measurement(
                    landmark.position,
                    self.state.position,
                    self.state.rotation_matrix
                )
                
                # Skip if projection failed
                if predicted_pixel is None:
                    continue
                
                # Compute innovation (measurement residual)
                z_observed = np.array([obs.pixel.u, obs.pixel.v])
                z_predicted = np.array([predicted_pixel[0], predicted_pixel[1]])
                innovation = z_observed - z_predicted
                
                # Innovation covariance: S = H * P * H' + R
                # H is the measurement Jacobian (2x15)
                H = jacobian_H  # Measurement Jacobian
                P = self.state.covariance
                R = np.eye(2) * self.config.pixel_noise_std**2  # Measurement noise
                S = H @ P @ H.T + R
                
                # Check for outliers using Mahalanobis distance
                chi2 = innovation.T @ np.linalg.inv(S) @ innovation
                if chi2 > self.config.chi2_threshold:
                    self.num_outliers += 1
                    continue  # Skip outlier
                
                # Kalman gain: K = P * H' * inv(S)
                K = P @ H.T @ np.linalg.inv(S)
                
                # State correction: dx = K * innovation
                dx = K @ innovation
                
                # Apply correction to state
                self._apply_correction(dx)
                
                # Covariance update: P = (I - K*H) * P
                self.state.covariance = (np.eye(15) - K @ H) @ P
                
                # Store/update landmark in our map
                if obs.landmark_id not in self.landmarks:
                    self.landmarks[obs.landmark_id] = landmark
    
    def _predict_measurement(
        self, 
        landmark_position: np.ndarray,
        robot_position: np.ndarray,
        robot_rotation: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Predict pixel measurement of a landmark and compute Jacobian.
        
        Args:
            landmark_position: 3D position of landmark in world frame
            robot_position: Current robot position in world frame
            robot_rotation: Current robot rotation matrix (world to body)
            
        Returns:
            (predicted_pixel, jacobian) or (None, None) if not visible
            predicted_pixel: [u, v] pixel coordinates
            jacobian: 2x15 measurement Jacobian matrix
        """
        # Transform landmark to body frame
        p_world = landmark_position - robot_position
        p_body = robot_rotation.T @ p_world  # R^T transforms from world to body
        
        # Transform from body to camera frame
        # Body: X forward, Y left, Z up (ENU)
        # Camera: Z forward, X right, Y down (optical)
        # Transform: C_x = -B_y, C_y = -B_z, C_z = B_x
        p_camera = np.array([
            -p_body[1],  # Camera X = -Body Y
            -p_body[2],  # Camera Y = -Body Z
            p_body[0]    # Camera Z = Body X (depth)
        ])
        
        # Check if landmark is in front of camera (Z > 0 in camera frame)
        if p_camera[2] <= 0.1:  # Behind or too close
            return None, None
        
        # Standard pinhole camera projection
        fx = self.camera_calib.intrinsics.fx if self.camera_calib else 500.0
        fy = self.camera_calib.intrinsics.fy if self.camera_calib else 500.0
        cx = self.camera_calib.intrinsics.cx if self.camera_calib else 320.0
        cy = self.camera_calib.intrinsics.cy if self.camera_calib else 240.0
        
        # Project to pixel coordinates
        # u = fx * X_c/Z_c + cx
        # v = fy * Y_c/Z_c + cy
        depth = p_camera[2]  # Z is depth in camera frame
        u = fx * p_camera[0] / depth + cx
        v = fy * p_camera[1] / depth + cy
        
        # Check image bounds
        width = self.camera_calib.intrinsics.width if self.camera_calib else 640
        height = self.camera_calib.intrinsics.height if self.camera_calib else 480
        if u < 0 or u >= width or v < 0 or v >= height:
            return None, None
        
        predicted_pixel = np.array([u, v])
        
        # Compute Jacobian of measurement w.r.t. state
        # State: [position(3), velocity(3), rotation(3), accel_bias(3), gyro_bias(3)]
        H = np.zeros((2, 15))
        
        # Jacobian computation in camera frame
        # u = fx * X_c/Z_c + cx, v = fy * Y_c/Z_c + cy
        z = p_camera[2]  # depth in camera frame
        z2 = z * z
        
        # du/dp_camera = [fx/Z, 0, -fx*X/Z²]
        # dv/dp_camera = [0, fy/Z, -fy*Y/Z²]
        du_dpc = np.array([fx/z, 0, -fx*p_camera[0]/z2])
        dv_dpc = np.array([0, fy/z, -fy*p_camera[1]/z2])
        
        # dp_camera/dp_body using our transform: C_x = -B_y, C_y = -B_z, C_z = B_x
        # This gives us the Jacobian matrix:
        dpc_dpb = np.array([
            [0, -1, 0],  # dC_x/dB = [0, -1, 0]
            [0, 0, -1],  # dC_y/dB = [0, 0, -1]
            [1, 0, 0]    # dC_z/dB = [1, 0, 0]
        ])
        
        # Chain rule: du/dp_body = du/dp_camera * dp_camera/dp_body
        dp_du = du_dpc @ dpc_dpb
        dp_dv = dv_dpc @ dpc_dpb
        
        # dp_body/dp_world = -R^T (derivative w.r.t. robot position)
        H[0, 0:3] = -dp_du @ robot_rotation.T
        H[1, 0:3] = -dp_dv @ robot_rotation.T
        
        # Jacobian w.r.t. rotation (using SO3 tangent space)
        # This is more complex - simplified here
        # dp_body/dtheta ≈ -R^T * skew(p_world)
        p_world_skew = skew(p_world)
        dp_dtheta = -robot_rotation.T @ p_world_skew
        
        H[0, 6:9] = dp_du @ dp_dtheta
        H[1, 6:9] = dp_dv @ dp_dtheta
        
        # Jacobians w.r.t. velocity and biases are zero for static measurements
        # H[:, 3:6] = 0  # Already zero
        # H[:, 9:15] = 0  # Already zero
        
        return predicted_pixel, H
    
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
        
        # Convert landmarks dict to Map object
        landmarks_map = Map()
        for landmark in self.landmarks.values():
            landmarks_map.add_landmark(landmark)
        
        return EstimatorState(
            timestamp=self.state.timestamp,
            robot_pose=self.state.get_pose(),
            robot_velocity=self.state.velocity.copy(),
            robot_covariance=self.state.covariance[0:6, 0:6],  # Position and velocity
            landmarks=landmarks_map if self.landmarks else None
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
            State vector [position, velocity, rotation_matrix_flattened, accel_bias, gyro_bias]
        """
        if self.state is None:
            return np.zeros(21)  # 3+3+9+3+3 = 21
        
        return np.concatenate([
            self.state.position,
            self.state.velocity,
            self.state.rotation_matrix.flatten(),  # 9 elements instead of 4
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