"""
GTSAM IMU Preintegration Wrapper

Provides a wrapper around GTSAM's PreintegratedImuMeasurements for use
with our data structures and the CombinedImuFactor.
"""

import numpy as np
import gtsam
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from src.common.data_structures import IMUMeasurement, IMUCalibration


@dataclass
class GTSAMPreintegrationParams:
    """Parameters for GTSAM IMU preintegration."""
    gravity_magnitude: float = 9.81
    accel_noise_density: float = 0.01
    gyro_noise_density: float = 0.001
    integration_error_cov: float = 1e-8
    accel_bias_rw: float = 0.001  # Random walk for accelerometer bias
    gyro_bias_rw: float = 0.0001   # Random walk for gyroscope bias
    use_2nd_order_coriolis: bool = False
    
    @classmethod
    def from_imu_calibration(cls, calib: IMUCalibration) -> 'GTSAMPreintegrationParams':
        """Create parameters from IMU calibration."""
        return cls(
            gravity_magnitude=calib.gravity_magnitude if hasattr(calib, 'gravity_magnitude') else 9.81,
            accel_noise_density=calib.accelerometer_noise_density,
            gyro_noise_density=calib.gyroscope_noise_density,
            accel_bias_rw=calib.accelerometer_random_walk,
            gyro_bias_rw=calib.gyroscope_random_walk
        )


class GTSAMPreintegration:
    """
    Wrapper for GTSAM's IMU preintegration.
    
    This class manages GTSAM's PreintegratedImuMeasurements and provides
    methods to create CombinedImuFactors for the factor graph.
    """
    
    def __init__(self, params: Optional[GTSAMPreintegrationParams] = None):
        """
        Initialize GTSAM preintegration wrapper.
        
        Args:
            params: Preintegration parameters (uses defaults if None)
        """
        self.params = params or GTSAMPreintegrationParams()
        
        # Create GTSAM preintegration parameters
        # MakeSharedU assumes gravity along negative Z axis
        # Use PreintegrationCombinedParams for CombinedImuFactor
        self.gtsam_params = gtsam.PreintegrationCombinedParams.MakeSharedU(
            self.params.gravity_magnitude
        )
        
        # Set noise parameters (continuous-time)
        self.gtsam_params.setAccelerometerCovariance(
            self.params.accel_noise_density**2 * np.eye(3)
        )
        self.gtsam_params.setGyroscopeCovariance(
            self.params.gyro_noise_density**2 * np.eye(3)
        )
        self.gtsam_params.setIntegrationCovariance(
            self.params.integration_error_cov * np.eye(3)
        )
        
        # Set Coriolis parameter
        self.gtsam_params.setUse2ndOrderCoriolis(self.params.use_2nd_order_coriolis)
        
        # Set bias random walk parameters for CombinedImuFactor
        self.gtsam_params.setBiasAccCovariance(
            self.params.accel_bias_rw**2 * np.eye(3)
        )
        self.gtsam_params.setBiasOmegaCovariance(
            self.params.gyro_bias_rw**2 * np.eye(3)
        )
        
        # Initialize with zero bias
        self.current_bias = gtsam.imuBias.ConstantBias()
        
        # Create preintegrated measurements object
        # Use PreintegratedCombinedMeasurements for CombinedImuFactor (supports bias estimation)
        self.pim = gtsam.PreintegratedCombinedMeasurements(
            self.gtsam_params, 
            self.current_bias
        )
        
        # Track measurements for debugging
        self.measurements: List[IMUMeasurement] = []
        self.total_time = 0.0
    
    def reset(self, bias: Optional[gtsam.imuBias.ConstantBias] = None):
        """
        Reset preintegration with optional new bias.
        
        Args:
            bias: New bias values (uses zero if None)
        """
        if bias is not None:
            self.current_bias = bias
        else:
            self.current_bias = gtsam.imuBias.ConstantBias()
        
        # Reset the preintegrated measurements
        self.pim.resetIntegrationAndSetBias(self.current_bias)
        
        # Clear tracked measurements
        self.measurements.clear()
        self.total_time = 0.0
    
    def add_measurement(self, measurement: IMUMeasurement, dt: Optional[float] = None):
        """
        Add an IMU measurement to preintegration.
        
        Args:
            measurement: IMU measurement with accelerometer and gyroscope data
            dt: Time delta (computed from timestamps if None)
        """
        # Store measurement
        self.measurements.append(measurement)
        
        # Compute dt if not provided
        if dt is None:
            if len(self.measurements) > 1:
                dt = measurement.timestamp - self.measurements[-2].timestamp
            else:
                # First measurement - use small default dt
                dt = 0.005  # 200 Hz default
        
        # Skip if dt is invalid
        if dt <= 0:
            return
        
        # Integrate measurement
        self.pim.integrateMeasurement(
            measurement.accelerometer,
            measurement.gyroscope,
            dt
        )
        
        self.total_time += dt
    
    def add_measurements_batch(self, measurements: List[IMUMeasurement]):
        """
        Add multiple IMU measurements.
        
        Args:
            measurements: List of IMU measurements
        """
        for i, meas in enumerate(measurements):
            if i == 0:
                dt = 0.005  # Default for first measurement
            else:
                dt = meas.timestamp - measurements[i-1].timestamp
            
            if dt > 0:
                self.add_measurement(meas, dt)
    
    def get_preintegrated_measurements(self) -> gtsam.PreintegratedImuMeasurements:
        """
        Get the GTSAM PreintegratedImuMeasurements object.
        
        Returns:
            Current preintegrated measurements
        """
        return self.pim
    
    def create_combined_imu_factor(
        self,
        pose_i_key: int,
        vel_i_key: int,
        pose_j_key: int,
        vel_j_key: int,
        bias_i_key: int,
        bias_j_key: int
    ) -> gtsam.CombinedImuFactor:
        """
        Create a CombinedImuFactor for the factor graph.
        
        Args:
            pose_i_key: Key for pose at time i
            vel_i_key: Key for velocity at time i
            pose_j_key: Key for pose at time j
            vel_j_key: Key for velocity at time j
            bias_i_key: Key for bias at time i
            bias_j_key: Key for bias at time j
        
        Returns:
            CombinedImuFactor connecting the states
        """
        return gtsam.CombinedImuFactor(
            pose_i_key, vel_i_key,
            pose_j_key, vel_j_key,
            bias_i_key, bias_j_key,
            self.pim
        )
    
    def predict_state(
        self,
        prev_pose: gtsam.Pose3,
        prev_velocity: np.ndarray,
        prev_bias: Optional[gtsam.imuBias.ConstantBias] = None
    ) -> Tuple[gtsam.Pose3, np.ndarray]:
        """
        Predict next state using preintegrated measurements.
        
        This properly handles gravity compensation internally.
        
        Args:
            prev_pose: Previous pose
            prev_velocity: Previous velocity (3D vector)
            prev_bias: Previous bias (uses current if None)
        
        Returns:
            (predicted_pose, predicted_velocity)
        """
        if prev_bias is None:
            prev_bias = self.current_bias
        
        # Create NavState for prediction
        prev_nav_state = gtsam.NavState(prev_pose, prev_velocity)
        
        # Predict using GTSAM's built-in prediction
        predicted_nav_state = self.pim.predict(prev_nav_state, prev_bias)
        
        return predicted_nav_state.pose(), predicted_nav_state.velocity()
    
    def get_delta_values(self) -> Dict[str, np.ndarray]:
        """
        Get preintegrated delta values for debugging.
        
        Returns:
            Dictionary with delta_position, delta_velocity, delta_rotation
        """
        return {
            'delta_position': self.pim.deltaPij(),
            'delta_velocity': self.pim.deltaVij(),
            'delta_rotation': self.pim.deltaRij().matrix(),
            'total_time': self.total_time,
            'num_measurements': len(self.measurements)
        }
    
    def get_covariance(self) -> np.ndarray:
        """
        Get the preintegration covariance matrix.
        
        Returns:
            9x9 covariance matrix for [rotation, velocity, position]
        """
        return self.pim.preintMeasCov()
    
    @staticmethod
    def create_bias_noise_model(
        accel_bias_rw: float = 1e-3,
        gyro_bias_rw: float = 1e-5
    ) -> gtsam.noiseModel.Diagonal:
        """
        Create noise model for bias random walk.
        
        Args:
            accel_bias_rw: Accelerometer bias random walk
            gyro_bias_rw: Gyroscope bias random walk
        
        Returns:
            Diagonal noise model for bias
        """
        # Create 6D noise vector [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        bias_noise = np.array([
            accel_bias_rw, accel_bias_rw, accel_bias_rw,
            gyro_bias_rw, gyro_bias_rw, gyro_bias_rw
        ])
        return gtsam.noiseModel.Diagonal.Sigmas(bias_noise)


class PreintegrationManager:
    """
    Manages multiple preintegration instances for keyframe-based SLAM.
    """
    
    def __init__(self, params: Optional[GTSAMPreintegrationParams] = None):
        """
        Initialize preintegration manager.
        
        Args:
            params: Preintegration parameters
        """
        self.params = params or GTSAMPreintegrationParams()
        self.current_preintegration = GTSAMPreintegration(self.params)
        self.stored_preintegrations: Dict[Tuple[int, int], GTSAMPreintegration] = {}
    
    def start_new_preintegration(
        self,
        from_keyframe_id: int,
        to_keyframe_id: int,
        bias: Optional[gtsam.imuBias.ConstantBias] = None
    ):
        """
        Start preintegrating for a new keyframe interval.
        
        Args:
            from_keyframe_id: Starting keyframe ID
            to_keyframe_id: Ending keyframe ID
            bias: Initial bias
        """
        # Store current if it has measurements
        if len(self.current_preintegration.measurements) > 0:
            # Store with placeholder key if needed
            self.stored_preintegrations[(-1, -1)] = self.current_preintegration
        
        # Create new preintegration
        self.current_preintegration = GTSAMPreintegration(self.params)
        if bias is not None:
            self.current_preintegration.reset(bias)
    
    def get_preintegration(
        self,
        from_keyframe_id: int,
        to_keyframe_id: int
    ) -> Optional[GTSAMPreintegration]:
        """
        Get stored preintegration between keyframes.
        
        Args:
            from_keyframe_id: Starting keyframe ID
            to_keyframe_id: Ending keyframe ID
        
        Returns:
            Stored preintegration or None if not found
        """
        return self.stored_preintegrations.get((from_keyframe_id, to_keyframe_id))
    
    def finalize_current(
        self,
        from_keyframe_id: int,
        to_keyframe_id: int
    ):
        """
        Finalize and store current preintegration.
        
        Args:
            from_keyframe_id: Starting keyframe ID
            to_keyframe_id: Ending keyframe ID
        """
        self.stored_preintegrations[(from_keyframe_id, to_keyframe_id)] = \
            self.current_preintegration
        self.current_preintegration = GTSAMPreintegration(self.params)