"""
Square Root Information Filter (SRIF) for Visual-Inertial SLAM.

Implements a numerically stable variant of the Extended Kalman Filter
using QR factorization to maintain the square root of the information matrix.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import logging
from scipy.linalg import qr, solve_triangular

from src.estimation.base_estimator import (
    BaseEstimator, EstimatorState, EstimatorConfig,
    EstimatorResult, EstimatorType
)
from src.common.config import SRIFConfig
from src.estimation.imu_integration import (
    IMUIntegrator, IMUState, IntegrationMethod
)
from src.estimation.camera_model import CameraMeasurementModel
from src.common.data_structures import (
    IMUMeasurement, CameraFrame, Map, Landmark,
    Trajectory, TrajectoryState, Pose,
    CameraCalibration, IMUCalibration
)
from src.utils.math_utils import (
    skew
)

logger = logging.getLogger(__name__)


@dataclass
class SRIFState:
    """
    SRIF state representation in information form.
    
    The state is represented as:
    - Information vector: y = R^T * R * x (where R^T R = P^{-1})
    - Square root information matrix: R (upper triangular)
    
    Attributes:
        position: Robot position [x, y, z]
        velocity: Robot velocity [vx, vy, vz]
        rotation_matrix: Robot orientation as 3x3 SO3 matrix
        accel_bias: Accelerometer bias [bax, bay, baz]
        gyro_bias: Gyroscope bias [bgx, bgy, bgz]
        timestamp: State timestamp
        information_vector: y = R^T R x (15x1)
        sqrt_information: R matrix (15x15 upper triangular)
    """
    position: np.ndarray
    velocity: np.ndarray
    rotation_matrix: np.ndarray  # 3x3 SO3 matrix
    accel_bias: np.ndarray
    gyro_bias: np.ndarray
    timestamp: float
    information_vector: np.ndarray = field(default_factory=lambda: np.zeros(15))
    sqrt_information: np.ndarray = field(default_factory=lambda: np.eye(15))
    
    def to_imu_state(self) -> IMUState:
        """Convert to IMU state."""
        return IMUState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            rotation_matrix=self.rotation_matrix.copy(),
            accel_bias=self.accel_bias.copy(),
            gyro_bias=self.gyro_bias.copy(),
            timestamp=self.timestamp
        )
    
    def from_imu_state(self, imu_state: IMUState) -> None:
        """Update from IMU state."""
        self.position = imu_state.position.copy()
        self.velocity = imu_state.velocity.copy()
        self.rotation_matrix = imu_state.rotation_matrix.copy()
        self.accel_bias = imu_state.accel_bias.copy()
        self.gyro_bias = imu_state.gyro_bias.copy()
        self.timestamp = imu_state.timestamp
    
    def get_covariance_matrix(self) -> np.ndarray:
        """
        Compute covariance matrix from square root information.
        P = (R^T R)^{-1}
        """
        try:
            # Solve R^T R P = I for P
            # First solve R^T y = I for y, then R P = y for P
            P = solve_triangular(
                self.sqrt_information,
                solve_triangular(
                    self.sqrt_information.T,
                    np.eye(15),
                    lower=True
                ),
                lower=False
            )
            return P
        except np.linalg.LinAlgError:
            logger.warning("Failed to compute covariance from sqrt_information")
            return np.eye(15) * 1e6  # Large uncertainty
    
    def get_state_vector(self) -> np.ndarray:
        """
        Compute state vector from information form.
        x = (R^T R)^{-1} y
        """
        try:
            # Solve R^T R x = y
            # First solve R^T z = y for z, then R x = z for x
            x = solve_triangular(
                self.sqrt_information,
                solve_triangular(
                    self.sqrt_information.T,
                    self.information_vector,
                    lower=True
                ),
                lower=False
            )
            return x
        except np.linalg.LinAlgError:
            logger.warning("Failed to compute state from information form")
            # Return current state estimate
            return self._pack_state_vector()
    
    def _pack_state_vector(self) -> np.ndarray:
        """Pack state into vector form."""
        from src.utils.math_utils import so3_log
        x = np.zeros(15)
        x[0:3] = self.position
        x[3:6] = self.velocity
        x[6:9] = so3_log(self.rotation_matrix)  # SO3 log map
        x[9:12] = self.accel_bias
        x[12:15] = self.gyro_bias
        return x
    
    def _unpack_state_vector(self, x: np.ndarray) -> None:
        """Unpack state from vector form."""
        from src.utils.math_utils import so3_exp
        self.position = x[0:3].copy()
        self.velocity = x[3:6].copy()
        self.rotation_matrix = so3_exp(x[6:9])  # SO3 exp map
        self.accel_bias = x[9:12].copy()
        self.gyro_bias = x[12:15].copy()


# Config now imported from src.common.config


class SRIFSlam(BaseEstimator):
    """
    Square Root Information Filter for Visual-Inertial SLAM.
    
    Uses QR factorization to maintain numerical stability while
    performing Kalman filter updates in information form.
    """
    
    def __init__(
        self,
        config: SRIFConfig,
        camera_calibration: CameraCalibration,
        imu_calibration: Optional[IMUCalibration] = None
    ):
        """
        Initialize SRIF.
        
        Args:
            config: SRIF configuration
            camera_calibration: Camera calibration
            imu_calibration: Optional IMU calibration
        """
        super().__init__(config)
        self.config = config
        self.camera_calib = camera_calibration
        self.imu_calib = imu_calibration
        
        # Initialize state
        self.state: Optional[SRIFState] = None
        
        # Camera model
        self.camera_model = CameraMeasurementModel(camera_calibration)
        
        # IMU integrator
        gravity = np.array([0, 0, -9.81])
        method_map = {
            "euler": IntegrationMethod.EULER,
            "rk4": IntegrationMethod.RK4,
            "midpoint": IntegrationMethod.MIDPOINT
        }
        self.imu_integrator = IMUIntegrator(
            gravity=gravity,
            method=method_map.get(config.integration_method, IntegrationMethod.RK4)
        )
        
        # Landmark estimates
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
        Initialize SRIF with initial state.
        
        Args:
            initial_pose: Initial robot pose
            initial_covariance: Initial uncertainty (will be converted to information form)
        """
        # Create initial state
        self.state = SRIFState(
            position=initial_pose.position.copy(),
            velocity=np.zeros(3),
            rotation_matrix=initial_pose.rotation_matrix.copy(),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp=initial_pose.timestamp
        )
        
        # Initialize information form
        if initial_covariance is not None:
            # Convert covariance to information
            try:
                # P^{-1} = R^T R
                # Compute R via Cholesky of P^{-1}
                P_inv = np.linalg.inv(initial_covariance)
                self.state.sqrt_information = np.linalg.cholesky(P_inv).T
            except np.linalg.LinAlgError:
                logger.warning("Failed to initialize from covariance, using default")
                self._initialize_default_information()
        else:
            self._initialize_default_information()
        
        # Initialize information vector
        x = self.state._pack_state_vector()
        self.state.information_vector = self.state.sqrt_information.T @ self.state.sqrt_information @ x
        
        logger.info(f"SRIF initialized at time {initial_pose.timestamp}")
    
    def _initialize_default_information(self) -> None:
        """Initialize with default information matrix."""
        # Create diagonal sqrt information matrix
        sqrt_info_diag = np.zeros(15)
        sqrt_info_diag[0:3] = 1.0 / self.config.initial_position_std
        sqrt_info_diag[3:6] = 1.0 / self.config.initial_velocity_std
        sqrt_info_diag[6:9] = 1.0 / self.config.initial_orientation_std  # SO3 tangent space
        sqrt_info_diag[9:12] = 1.0 / self.config.initial_bias_std  # accel bias
        sqrt_info_diag[12:15] = 1.0 / self.config.initial_bias_std  # gyro bias
        
        self.state.sqrt_information = np.diag(sqrt_info_diag)
    
    def predict(self, imu_measurements: List[IMUMeasurement], dt: float) -> None:
        """
        Prediction step using IMU measurements.
        
        For SRIF, we need to:
        1. Convert to covariance form
        2. Perform standard EKF prediction
        3. Convert back to information form
        
        Args:
            imu_measurements: List of IMU measurements
            dt: Total time step
        """
        if self.state is None:
            logger.warning("SRIF not initialized, skipping prediction")
            return
        
        if not imu_measurements:
            return
        
        # Convert to covariance form for prediction
        P = self.state.get_covariance_matrix()
        x = self.state.get_state_vector()
        
        # Unpack current state
        self.state._unpack_state_vector(x)
        
        # Propagate state using IMU
        imu_state = self.state.to_imu_state()
        for meas in imu_measurements:
            imu_state = self.imu_integrator.integrate(
                imu_state, meas, dt / len(imu_measurements)
            )
        self.state.from_imu_state(imu_state)
        
        # Compute state transition Jacobian
        F = self._compute_state_transition_jacobian(imu_measurements, dt)
        
        # Process noise
        Q = self._compute_process_noise(dt)
        
        # Propagate covariance: P = F P F^T + Q
        P_pred = F @ P @ F.T + Q
        
        # Convert back to information form using QR decomposition
        self._covariance_to_information(P_pred)
        
        # Update information vector
        x_pred = self.state._pack_state_vector()
        self.state.information_vector = self.state.sqrt_information.T @ self.state.sqrt_information @ x_pred
        
        # Update timestamp
        if imu_measurements:
            self.state.timestamp = imu_measurements[-1].timestamp
    
    def update(self, camera_frame: CameraFrame, landmarks: Map) -> None:
        """
        Measurement update using camera observations.
        
        Uses QR factorization for numerically stable updates.
        
        Args:
            camera_frame: Camera observations
            landmarks: Known landmarks
        """
        if self.state is None:
            logger.warning("SRIF not initialized, skipping update")
            return
        
        # Process each observation
        for obs in camera_frame.observations:
            if obs.landmark_id not in landmarks.landmarks:
                continue
            
            landmark = landmarks.landmarks[obs.landmark_id]
            
            # Perform QR update for this measurement
            self._qr_measurement_update(obs, landmark)
        
        self.num_updates += 1
    
    def _qr_measurement_update(
        self,
        observation: 'CameraObservation',
        landmark: Landmark
    ) -> None:
        """
        Perform measurement update using QR factorization.
        
        This maintains the triangular structure of the square root
        information matrix for numerical stability.
        
        Args:
            observation: Camera observation
            landmark: Corresponding landmark
        """
        # Get current state estimate
        x = self.state.get_state_vector()
        self.state._unpack_state_vector(x)
        
        # Create pose for projection
        pose = Pose(
            timestamp=self.state.timestamp,
            position=self.state.position,
            rotation_matrix=self.state.rotation_matrix
        )
        
        # Compute measurement residual and Jacobian
        error = self.camera_model.compute_reprojection_error(
            observation, landmark, pose
        )
        
        if error is None:
            return
        
        # Extract measurement model
        z_residual = error.residual  # 2x1
        H = error.jacobian_pose  # 2x6 (only pose part)
        
        # Expand H to full state dimension (2x15)
        H_full = np.zeros((2, 15))
        H_full[:, 0:3] = H[:, 3:6]  # Position
        # Map rotation Jacobian (simplified)
        H_full[:, 6:9] = np.zeros((2, 3))  # Simplified SO3 Jacobian
        
        # Measurement noise
        R_meas = np.eye(2) * self.config.pixel_noise_std**2
        R_meas_sqrt_inv = np.eye(2) / self.config.pixel_noise_std
        
        # Form augmented system for QR update
        # [R_a] = [    R    ] [R_new]
        # [H_a]   [H*Rsqrt^-1] [  0  ]
        
        n = self.state.sqrt_information.shape[0]
        augmented = np.zeros((n + 2, n))
        augmented[0:n, :] = self.state.sqrt_information
        augmented[n:n+2, :] = R_meas_sqrt_inv @ H_full
        
        # QR decomposition
        Q, R_new = qr(augmented, mode='economic')
        
        # Update square root information matrix
        self.state.sqrt_information = R_new[0:n, :]
        
        # Update information vector
        # y_new = y_old + H^T R_meas^{-1} z_residual
        self.state.information_vector += H_full.T @ np.linalg.solve(R_meas, z_residual)
    
    def _compute_state_transition_jacobian(
        self,
        imu_measurements: List[IMUMeasurement],
        dt: float
    ) -> np.ndarray:
        """
        Compute state transition Jacobian for IMU prediction.
        
        Simplified version - in practice would use IMU preintegration.
        """
        F = np.eye(15)
        
        # Position depends on velocity
        F[0:3, 3:6] = np.eye(3) * dt
        
        # Velocity depends on orientation (gravity rotation)
        # Simplified - would need actual rotation derivatives
        
        return F
    
    def _compute_process_noise(self, dt: float) -> np.ndarray:
        """Compute process noise covariance."""
        Q = np.zeros((15, 15))
        
        # Position noise (from velocity integration)
        Q[0:3, 0:3] = np.eye(3) * (self.config.accel_noise_std * dt**2)**2
        
        # Velocity noise
        Q[3:6, 3:6] = np.eye(3) * (self.config.accel_noise_std * dt)**2
        
        # Orientation noise (SO3 tangent space)
        Q[6:9, 6:9] = np.eye(3) * (self.config.gyro_noise_std * dt)**2
        
        # Bias noise
        Q[9:12, 9:12] = np.eye(3) * (self.config.accel_bias_noise_std * dt)**2  # accel bias
        Q[12:15, 12:15] = np.eye(3) * (self.config.gyro_bias_noise_std * dt)**2  # gyro bias
        
        return Q
    
    def _covariance_to_information(self, P: np.ndarray) -> None:
        """
        Convert covariance matrix to square root information form.
        
        Uses QR decomposition of P^{-1/2} for numerical stability.
        """
        try:
            # Compute P^{-1} via Cholesky
            # P = L L^T, so P^{-1} = L^{-T} L^{-1}
            L = np.linalg.cholesky(P)
            L_inv = solve_triangular(L, np.eye(P.shape[0]), lower=True)
            
            # R = L^{-T} so R^T R = P^{-1}
            self.state.sqrt_information = L_inv.T
            
        except np.linalg.LinAlgError:
            logger.warning("Cholesky failed, using eigendecomposition")
            # Fallback to eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(P)
            eigvals = np.maximum(eigvals, 1e-10)  # Ensure positive
            P_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            # QR decomposition to get upper triangular
            self.state.sqrt_information, _ = qr(P_sqrt_inv)
    
    # Required abstract methods
    
    def get_state(self) -> EstimatorState:
        """Get current estimator state."""
        if self.state is None:
            raise RuntimeError("SRIF not initialized")
        
        return EstimatorState(
            timestamp=self.state.timestamp,
            robot_pose=Pose(
                timestamp=self.state.timestamp,
                position=self.state.position.copy(),
                rotation_matrix=self.state.rotation_matrix.copy()
            ),
            robot_velocity=self.state.velocity.copy(),
            landmarks=Map() if not self.landmarks else None,  # Return empty map if no landmarks
            covariance_matrix=self.state.get_covariance_matrix()
        )
    
    def get_trajectory(self) -> Trajectory:
        """Get estimated trajectory."""
        trajectory = Trajectory()
        
        if self.state:
            state = TrajectoryState(
                pose=Pose(
                    timestamp=self.state.timestamp,
                    position=self.state.position.copy(),
                    rotation_matrix=self.state.rotation_matrix.copy()
                ),
                velocity=self.state.velocity.copy()
            )
            trajectory.add_state(state)
        
        return trajectory
    
    def get_map(self) -> Map:
        """Get estimated map."""
        map_data = Map()
        for landmark in self.landmarks.values():
            map_data.add_landmark(landmark)
        return map_data
    
    def get_result(self) -> EstimatorResult:
        """Get complete estimation result."""
        return EstimatorResult(
            trajectory=self.get_trajectory(),
            landmarks=self.get_map(),
            states=[self.get_state()],
            runtime_ms=0.0,
            iterations=self.num_updates,
            converged=True,
            final_cost=0.0,
            metadata={
                "num_updates": self.num_updates,
                "num_outliers": self.num_outliers,
                "estimator_type": "SRIF"
            }
        )
    
    def get_state_vector(self) -> np.ndarray:
        """Get state vector."""
        if self.state is None:
            return np.array([])
        return self.state.get_state_vector()
    
    def get_covariance_matrix(self) -> Optional[np.ndarray]:
        """Get covariance matrix."""
        if self.state is None:
            return None
        return self.state.get_covariance_matrix()
    
    def get_information_matrix(self) -> Optional[np.ndarray]:
        """
        Get information matrix.
        
        Returns:
            Information matrix (P^{-1})
        """
        if self.state is None:
            return None
        R = self.state.sqrt_information
        return R.T @ R
    
    def get_sqrt_information_matrix(self) -> Optional[np.ndarray]:
        """
        Get square root information matrix.
        
        Returns:
            Square root information matrix R where R^T R = P^{-1}
        """
        if self.state is None:
            return None
        return self.state.sqrt_information.copy()
    
    def optimize(self) -> None:
        """
        Optimize state (not needed for SRIF as it's a filter).
        
        SRIF is a filtering method, not a batch optimization method,
        so this is a no-op.
        """
        pass
    
    def marginalize(self) -> None:
        """Marginalize old states (not needed for basic SRIF)."""
        pass
    
    def reset(self) -> None:
        """Reset estimator state."""
        self.state = None
        self.landmarks.clear()
        self.num_updates = 0
        self.num_outliers = 0