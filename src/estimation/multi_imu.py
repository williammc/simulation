"""
Multi-IMU fusion for improved motion estimation.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.common.data_structures import (
    IMUMeasurement, IMUCalibration
)
from src.estimation.imu_integration import IMUIntegrator, IntegrationMethod, IMUState
from src.utils.math_utils import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    skew
)


class IMUFusionMethod(Enum):
    """Methods for fusing multiple IMU measurements."""
    WEIGHTED_AVERAGE = "weighted_average"
    KALMAN_FILTER = "kalman_filter"
    COMPLEMENTARY_FILTER = "complementary_filter"
    VOTING = "voting"
    MAXIMUM_LIKELIHOOD = "maximum_likelihood"


@dataclass
class IMUConfig:
    """Configuration for a single IMU in multi-IMU setup."""
    imu_id: str
    calibration: IMUCalibration
    weight: float = 1.0  # Weight for fusion
    trust_factor: float = 1.0  # Trust factor based on noise characteristics
    max_deviation: float = 3.0  # Max deviation for outlier detection (in std devs)
    enabled: bool = True


@dataclass
class FusedIMUMeasurement:
    """Fused measurement from multiple IMUs."""
    timestamp: float
    acceleration: np.ndarray
    angular_velocity: np.ndarray
    covariance_accel: np.ndarray
    covariance_gyro: np.ndarray
    contributing_imus: List[str]
    fusion_confidence: float


class MultiIMUFusion:
    """
    Fuses measurements from multiple IMUs for improved robustness.
    
    Handles:
    - Different IMU locations and orientations
    - Outlier detection and rejection
    - Weighted fusion based on noise characteristics
    - Fault detection
    """
    
    def __init__(
        self,
        imu_configs: Dict[str, IMUConfig],
        fusion_method: IMUFusionMethod = IMUFusionMethod.WEIGHTED_AVERAGE,
        reference_frame: str = "body",
        outlier_threshold: float = 3.0
    ):
        """
        Initialize multi-IMU fusion.
        
        Args:
            imu_configs: Configuration for each IMU
            fusion_method: Method for fusing measurements
            reference_frame: Reference frame for fused measurements
        """
        self.imu_configs = imu_configs
        self.fusion_method = fusion_method
        self.reference_frame = reference_frame
        self.outlier_threshold = outlier_threshold
        
        # Compute fusion weights based on noise characteristics
        self._compute_fusion_weights()
        
        # Initialize integrators for each IMU
        self.integrators = {}
        for imu_id, config in imu_configs.items():
            self.integrators[imu_id] = IMUIntegrator(
                method=IntegrationMethod.RK4,
                gravity=config.calibration.gravity_magnitude
            )
        
        # Fault detection state
        self.fault_history = {imu_id: [] for imu_id in imu_configs}
        self.fault_threshold = 5  # Number of consecutive faults before disabling
    
    def _compute_fusion_weights(self):
        """Compute fusion weights based on IMU noise characteristics."""
        # Extract noise levels
        noise_levels = {}
        for imu_id, config in self.imu_configs.items():
            # Use trace of noise covariance as overall noise metric
            accel_noise = np.trace(config.calibration.noise_characteristics.accelerometer_noise_density * np.eye(3))
            gyro_noise = np.trace(config.calibration.noise_characteristics.gyroscope_noise_density * np.eye(3))
            total_noise = accel_noise + gyro_noise
            noise_levels[imu_id] = total_noise
        
        # Compute weights (inverse of noise)
        min_noise = min(noise_levels.values())
        for imu_id in self.imu_configs:
            if noise_levels[imu_id] > 0:
                # Weight inversely proportional to noise
                weight = min_noise / noise_levels[imu_id]
                self.imu_configs[imu_id].weight = weight * self.imu_configs[imu_id].trust_factor
    
    def fuse_measurements(
        self,
        measurements: Dict[str, IMUMeasurement],
        body_state: Optional[IMUState] = None
    ) -> Optional[FusedIMUMeasurement]:
        """
        Fuse measurements from multiple IMUs.
        
        Args:
            measurements: Dictionary of measurements by IMU ID
            body_state: Optional current body state for transformation
        
        Returns:
            Fused IMU measurement or None if insufficient valid measurements
        """
        # Filter out disabled IMUs
        valid_measurements = {
            imu_id: meas for imu_id, meas in measurements.items()
            if imu_id in self.imu_configs and self.imu_configs[imu_id].enabled
        }
        
        if not valid_measurements:
            return None
        
        # Transform measurements to common reference frame
        transformed_measurements = self._transform_to_reference_frame(
            valid_measurements, body_state
        )
        
        # Detect and remove outliers
        inlier_measurements = self._detect_outliers(transformed_measurements)
        
        # If too many outliers detected, use more lenient approach for voting method
        if (self.fusion_method == IMUFusionMethod.VOTING and 
            len(inlier_measurements) < len(transformed_measurements) * 0.6):
            # Fall back to less aggressive outlier detection for voting
            inlier_measurements = self._detect_outliers_lenient(transformed_measurements)
        
        if not inlier_measurements:
            return None
        
        # Fuse based on selected method
        if self.fusion_method == IMUFusionMethod.WEIGHTED_AVERAGE:
            return self._fuse_weighted_average(inlier_measurements)
        elif self.fusion_method == IMUFusionMethod.KALMAN_FILTER:
            return self._fuse_kalman_filter(inlier_measurements)
        elif self.fusion_method == IMUFusionMethod.COMPLEMENTARY_FILTER:
            return self._fuse_complementary_filter(inlier_measurements)
        elif self.fusion_method == IMUFusionMethod.VOTING:
            return self._fuse_voting(inlier_measurements)
        elif self.fusion_method == IMUFusionMethod.MAXIMUM_LIKELIHOOD:
            return self._fuse_maximum_likelihood(inlier_measurements)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def _transform_to_reference_frame(
        self,
        measurements: Dict[str, IMUMeasurement],
        body_state: Optional[IMUState] = None
    ) -> Dict[str, IMUMeasurement]:
        """
        Transform all measurements to common reference frame.
        
        Args:
            measurements: Raw measurements from each IMU
            body_state: Optional body state for compensation
        
        Returns:
            Transformed measurements
        """
        transformed = {}
        
        for imu_id, meas in measurements.items():
            config = self.imu_configs[imu_id]
            calib = config.calibration
            
            # Get IMU to body transformation
            T_B_I = calib.extrinsics.B_T_S
            R_B_I = T_B_I[:3, :3]
            p_B_I = T_B_I[:3, 3]
            
            # Transform acceleration (account for lever arm if rotating)
            accel_body = R_B_I @ meas.accelerometer
            
            # Transform angular velocity
            omega_body = R_B_I @ meas.gyroscope
            
            # Compensate for lever arm effects if body is rotating
            if body_state is not None and body_state.angular_velocity is not None:
                # Centripetal acceleration: a = omega x (omega x r)
                omega_cross_r = np.cross(body_state.angular_velocity, p_B_I)
                centripetal = np.cross(body_state.angular_velocity, omega_cross_r)
                
                # Tangential acceleration: a = alpha x r (if angular acceleration known)
                # For now, we skip this as we don't track angular acceleration
                
                accel_body -= centripetal
            
            # Create transformed measurement
            transformed[imu_id] = IMUMeasurement(
                timestamp=meas.timestamp,
                accelerometer=accel_body,
                gyroscope=omega_body
            )
        
        return transformed
    
    def _detect_outliers(
        self,
        measurements: Dict[str, IMUMeasurement]
    ) -> Dict[str, IMUMeasurement]:
        """
        Detect and remove outlier measurements.
        
        Uses Mahalanobis distance for outlier detection.
        
        Args:
            measurements: Transformed measurements
        
        Returns:
            Dictionary of inlier measurements
        """
        if len(measurements) < 3:
            # Not enough measurements for outlier detection
            return measurements
        
        # Collect all measurements
        accels = np.array([m.accelerometer for m in measurements.values()])
        gyros = np.array([m.gyroscope for m in measurements.values()])
        
        # Compute median and MAD (Median Absolute Deviation)
        accel_median = np.median(accels, axis=0)
        gyro_median = np.median(gyros, axis=0)
        
        accel_mad = np.median(np.abs(accels - accel_median), axis=0)
        gyro_mad = np.median(np.abs(gyros - gyro_median), axis=0)
        
        # Robust standard deviation estimate
        accel_std = 1.4826 * accel_mad  # MAD to std conversion
        gyro_std = 1.4826 * gyro_mad
        
        # Add minimum threshold to avoid rejecting good measurements when std is small
        min_accel_threshold = 0.1  # m/s^2
        min_gyro_threshold = 0.01  # rad/s
        
        # Check each measurement
        inliers = {}
        for imu_id, meas in measurements.items():
            config = self.imu_configs[imu_id]
            
            # Compute deviation from median
            accel_dev = np.abs(meas.accelerometer - accel_median)
            gyro_dev = np.abs(meas.gyroscope - gyro_median)
            
            # Compute thresholds for this IMU - use global outlier threshold if available
            threshold_multiplier = getattr(self, 'outlier_threshold', config.max_deviation)
            accel_threshold = np.maximum(threshold_multiplier * accel_std, min_accel_threshold)
            gyro_threshold = np.maximum(threshold_multiplier * gyro_std, min_gyro_threshold)
            
            # Check if within threshold
            accel_ok = np.all(accel_dev < accel_threshold)
            gyro_ok = np.all(gyro_dev < gyro_threshold)
            
            if accel_ok and gyro_ok:
                inliers[imu_id] = meas
                # Clear fault history
                self.fault_history[imu_id] = []
            else:
                # Record fault
                self.fault_history[imu_id].append(meas.timestamp)
                
                # Check if too many consecutive faults
                if len(self.fault_history[imu_id]) > self.fault_threshold:
                    print(f"Warning: IMU {imu_id} has excessive faults, disabling")
                    self.imu_configs[imu_id].enabled = False
        
        return inliers
    
    def _detect_outliers_lenient(
        self,
        measurements: Dict[str, IMUMeasurement]
    ) -> Dict[str, IMUMeasurement]:
        """
        Detect outliers using a more lenient approach for voting methods.
        
        Uses a larger threshold and only removes extreme outliers.
        
        Args:
            measurements: Transformed measurements
        
        Returns:
            Dictionary of inlier measurements
        """
        if len(measurements) < 3:
            return measurements
        
        # Collect all measurements
        accels = np.array([m.accelerometer for m in measurements.values()])
        gyros = np.array([m.gyroscope for m in measurements.values()])
        
        # Use median for robustness
        accel_median = np.median(accels, axis=0)
        gyro_median = np.median(gyros, axis=0)
        
        # Use larger threshold for lenient detection (5 standard deviations)
        accel_mad = np.median(np.abs(accels - accel_median), axis=0)
        gyro_mad = np.median(np.abs(gyros - gyro_median), axis=0)
        
        accel_std = 1.4826 * accel_mad
        gyro_std = 1.4826 * gyro_mad
        
        # Very lenient thresholds - only remove extreme outliers
        accel_threshold = np.maximum(5.0 * accel_std, 2.0)  # 2 m/s^2 minimum
        gyro_threshold = np.maximum(5.0 * gyro_std, 0.2)   # 0.2 rad/s minimum
        
        inliers = {}
        for imu_id, meas in measurements.items():
            accel_dev = np.abs(meas.accelerometer - accel_median)
            gyro_dev = np.abs(meas.gyroscope - gyro_median)
            
            # Only reject extremely deviant measurements
            accel_ok = np.all(accel_dev < accel_threshold)
            gyro_ok = np.all(gyro_dev < gyro_threshold)
            
            if accel_ok and gyro_ok:
                inliers[imu_id] = meas
        
        return inliers
    
    def _fuse_weighted_average(
        self,
        measurements: Dict[str, IMUMeasurement]
    ) -> FusedIMUMeasurement:
        """
        Fuse measurements using weighted average.
        
        Args:
            measurements: Inlier measurements
        
        Returns:
            Fused measurement
        """
        # Collect weights
        weights = np.array([self.imu_configs[imu_id].weight 
                           for imu_id in measurements.keys()])
        weights = weights / np.sum(weights)  # Normalize
        
        # Weighted average of measurements
        accel_fused = np.zeros(3)
        gyro_fused = np.zeros(3)
        
        for (imu_id, meas), w in zip(measurements.items(), weights):
            accel_fused += w * meas.accelerometer
            gyro_fused += w * meas.gyroscope
        
        # Estimate covariance (weighted combination)
        accel_cov = np.zeros((3, 3))
        gyro_cov = np.zeros((3, 3))
        
        for (imu_id, _), w in zip(measurements.items(), weights):
            config = self.imu_configs[imu_id]
            noise = config.calibration.noise_characteristics
            
            accel_cov += (w ** 2) * noise.accelerometer_noise_density * np.eye(3)
            gyro_cov += (w ** 2) * noise.gyroscope_noise_density * np.eye(3)
        
        # Compute fusion confidence (based on agreement between sensors)
        accels = np.array([m.accelerometer for m in measurements.values()])
        gyros = np.array([m.gyroscope for m in measurements.values()])
        
        if len(measurements) == 1:
            # Single sensor - moderate confidence
            confidence = 0.6
        elif len(measurements) == 2:
            # Two sensors - limited redundancy
            accel_std = np.std(accels, axis=0)
            accel_disagreement = np.mean(accel_std) / 0.05  # More sensitive to disagreement
            base_confidence = 0.7  # Lower base for limited redundancy
            confidence = np.clip(base_confidence / (1.0 + accel_disagreement), 0.2, 0.7)
        else:
            # Multiple sensors - full confidence calculation
            accel_std = np.std(accels, axis=0)
            gyro_std = np.std(gyros, axis=0)
            
            # Scale disagreement measures to reasonable range
            accel_disagreement = np.mean(accel_std) / 0.05  # More sensitive threshold
            gyro_disagreement = np.mean(gyro_std) / 0.005   # More sensitive threshold
            
            # Confidence decreases with disagreement
            if accel_disagreement < 0.1 and gyro_disagreement < 0.1:
                # Very good agreement
                confidence = 0.95
            elif accel_disagreement > 2.0 or gyro_disagreement > 2.0:
                # Poor agreement
                confidence = 0.1
            else:
                # Moderate agreement
                raw_confidence = 1.0 / (1.0 + accel_disagreement + gyro_disagreement)
                confidence = np.clip(raw_confidence, 0.2, 0.9)
        
        return FusedIMUMeasurement(
            timestamp=list(measurements.values())[0].timestamp,
            acceleration=accel_fused,
            angular_velocity=gyro_fused,
            covariance_accel=accel_cov,
            covariance_gyro=gyro_cov,
            contributing_imus=list(measurements.keys()),
            fusion_confidence=confidence
        )
    
    def _fuse_kalman_filter(
        self,
        measurements: Dict[str, IMUMeasurement]
    ) -> FusedIMUMeasurement:
        """
        Fuse measurements using Kalman filter.
        
        This is a simplified implementation. A full implementation
        would maintain state across time steps.
        
        Args:
            measurements: Inlier measurements
        
        Returns:
            Fused measurement
        """
        # For now, fall back to weighted average
        # A full implementation would use a Kalman filter to track
        # the IMU biases and optimally combine measurements
        return self._fuse_weighted_average(measurements)
    
    def _fuse_complementary_filter(
        self,
        measurements: Dict[str, IMUMeasurement]
    ) -> FusedIMUMeasurement:
        """
        Fuse measurements using complementary filter.
        
        Combines high-frequency and low-frequency components optimally.
        
        Args:
            measurements: Inlier measurements
        
        Returns:
            Fused measurement
        """
        # For IMU fusion, complementary filter is similar to weighted average
        # but with frequency-dependent weights
        # For simplicity, we use weighted average here
        return self._fuse_weighted_average(measurements)
    
    def _fuse_maximum_likelihood(
        self,
        measurements: Dict[str, IMUMeasurement]
    ) -> FusedIMUMeasurement:
        """
        Fuse measurements using maximum likelihood estimation.
        
        Assumes Gaussian noise and finds the ML estimate.
        
        Args:
            measurements: Inlier measurements
        
        Returns:
            Fused measurement
        """
        # For Gaussian noise, ML estimate is weighted average with inverse covariance weights
        # Use noise characteristics from calibration
        weights = []
        for imu_id, meas in measurements.items():
            config = self.imu_configs[imu_id]
            # Weight inversely proportional to noise variance
            # Use a simplified weight based on trust factor and noise characteristics
            weight = config.weight * config.trust_factor
            weights.append(weight)
        
        weights = np.array(weights) / np.sum(weights)
        
        # Compute weighted average
        accels = np.array([m.accelerometer for m in measurements.values()])
        gyros = np.array([m.gyroscope for m in measurements.values()])
        
        accel_fused = np.sum(accels * weights[:, np.newaxis], axis=0)
        gyro_fused = np.sum(gyros * weights[:, np.newaxis], axis=0)
        
        # Estimate covariance
        accel_cov = np.zeros((3, 3))
        gyro_cov = np.zeros((3, 3))
        
        for i, (imu_id, meas) in enumerate(measurements.items()):
            accel_diff = meas.accelerometer - accel_fused
            gyro_diff = meas.gyroscope - gyro_fused
            
            accel_cov += weights[i] * np.outer(accel_diff, accel_diff)
            gyro_cov += weights[i] * np.outer(gyro_diff, gyro_diff)
        
        # Confidence based on agreement
        if len(measurements) == 1:
            confidence = 0.8
        else:
            # Scale covariance traces to reasonable confidence range
            accel_uncertainty = np.trace(accel_cov) / 0.01
            gyro_uncertainty = np.trace(gyro_cov) / 0.0001
            raw_confidence = 1.0 / (1.0 + accel_uncertainty + gyro_uncertainty)
            confidence = np.clip(raw_confidence, 0.1, 0.95)
        
        return FusedIMUMeasurement(
            timestamp=list(measurements.values())[0].timestamp,
            acceleration=accel_fused,
            angular_velocity=gyro_fused,
            covariance_accel=accel_cov,
            covariance_gyro=gyro_cov,
            contributing_imus=list(measurements.keys()),
            fusion_confidence=confidence
        )
    
    def _fuse_voting(
        self,
        measurements: Dict[str, IMUMeasurement]
    ) -> FusedIMUMeasurement:
        """
        Fuse measurements using voting/median approach.
        
        More robust to outliers than weighted average.
        
        Args:
            measurements: Inlier measurements
        
        Returns:
            Fused measurement
        """
        # Use median for robustness
        accels = np.array([m.accelerometer for m in measurements.values()])
        gyros = np.array([m.gyroscope for m in measurements.values()])
        
        accel_fused = np.median(accels, axis=0)
        gyro_fused = np.median(gyros, axis=0)
        
        # Estimate covariance from spread
        accel_cov = np.diag(np.var(accels, axis=0))
        gyro_cov = np.diag(np.var(gyros, axis=0))
        
        # Confidence based on agreement
        if len(measurements) == 1:
            confidence = 0.8
        else:
            accel_mad = np.median(np.abs(accels - accel_fused), axis=0)
            gyro_mad = np.median(np.abs(gyros - gyro_fused), axis=0)
            
            # Scale MAD values to confidence range
            accel_spread = np.mean(accel_mad) / 0.05  # Normalize by reasonable spread
            gyro_spread = np.mean(gyro_mad) / 0.005   # Normalize by reasonable gyro spread
            
            raw_confidence = 1.0 / (1.0 + accel_spread + gyro_spread)
            confidence = np.clip(raw_confidence, 0.1, 0.95)
        
        return FusedIMUMeasurement(
            timestamp=list(measurements.values())[0].timestamp,
            acceleration=accel_fused,
            angular_velocity=gyro_fused,
            covariance_accel=accel_cov,
            covariance_gyro=gyro_cov,
            contributing_imus=list(measurements.keys()),
            fusion_confidence=confidence
        )
    
    def check_imu_health(self) -> Dict[str, bool]:
        """
        Check health status of all IMUs.
        
        Returns:
            Dictionary of IMU health status
        """
        health = {}
        for imu_id, config in self.imu_configs.items():
            health[imu_id] = config.enabled and len(self.fault_history[imu_id]) == 0
        return health
    
    def reset_imu(self, imu_id: str):
        """
        Reset an IMU after fault recovery.
        
        Args:
            imu_id: ID of IMU to reset
        """
        if imu_id in self.imu_configs:
            self.imu_configs[imu_id].enabled = True
            self.fault_history[imu_id] = []
            self.integrators[imu_id].reset()


def create_multi_imu_setup(
    num_imus: int = 3,
    configuration: str = "orthogonal"
) -> Dict[str, IMUConfig]:
    """
    Create a multi-IMU configuration.
    
    Args:
        num_imus: Number of IMUs
        configuration: Configuration type ("orthogonal", "redundant", "distributed")
    
    Returns:
        Dictionary of IMU configurations
    """
    from src.common.data_structures import IMUNoiseCharacteristics, IMUExtrinsics
    
    configs = {}
    
    # Base noise characteristics
    base_noise = IMUNoiseCharacteristics(
        accelerometer_noise_density=0.001,
        gyroscope_noise_density=0.0001,
        accelerometer_random_walk=0.0001,
        gyroscope_random_walk=0.00001
    )
    
    if configuration == "orthogonal":
        # IMUs oriented orthogonally for better observability
        # For more than 3 IMUs, add variations
        base_orientations = [
            np.eye(3),  # Aligned with body
            np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),  # Rotated 90° around Y
            np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])   # Rotated 90° around X
        ]
        
        orientations = []
        positions = []
        
        for i in range(num_imus):
            if i < 3:
                orientations.append(base_orientations[i])
                positions.append(np.array([i * 0.05, 0, 0]))
            else:
                # Add more IMUs with slight variations
                base_idx = i % 3
                angle_variation = (i // 3) * 0.1
                R_variation = np.array([
                    [np.cos(angle_variation), -np.sin(angle_variation), 0],
                    [np.sin(angle_variation), np.cos(angle_variation), 0],
                    [0, 0, 1]
                ])
                orientations.append(base_orientations[base_idx] @ R_variation)
                positions.append(np.array([i * 0.05, (i // 3) * 0.05, 0]))
        
    elif configuration == "redundant":
        # Multiple IMUs with same orientation for redundancy
        orientations = [np.eye(3)] * num_imus
        positions = [np.array([i * 0.05, 0, 0]) for i in range(num_imus)]
        
    elif configuration == "distributed":
        # IMUs distributed around the platform
        angles = np.linspace(0, 2 * np.pi, num_imus, endpoint=False)
        orientations = []
        positions = []
        
        for angle in angles:
            # Each IMU faces outward
            R = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            orientations.append(R)
            
            # Position on circle
            radius = 0.15
            pos = np.array([radius * np.cos(angle), radius * np.sin(angle), 0])
            positions.append(pos)
    else:
        raise ValueError(f"Unknown configuration: {configuration}")
    
    # Create IMU configs
    for i in range(num_imus):
        imu_id = f"imu_{i}"
        
        # Create transform
        T_B_I = np.eye(4)
        T_B_I[:3, :3] = orientations[i]
        T_B_I[:3, 3] = positions[i]
        
        # Add some variation in noise characteristics
        noise_scale = 1.0 + 0.1 * np.random.randn()
        noise = IMUNoiseCharacteristics(
            accelerometer_noise_density=base_noise.accelerometer_noise_density * noise_scale,
            gyroscope_noise_density=base_noise.gyroscope_noise_density * noise_scale,
            accelerometer_random_walk=base_noise.accelerometer_random_walk * noise_scale,
            gyroscope_random_walk=base_noise.gyroscope_random_walk * noise_scale
        )
        
        calib = IMUCalibration(
            imu_id=imu_id,
            extrinsics=IMUExtrinsics(B_T_S=T_B_I),
            noise_characteristics=noise,
            gravity_magnitude=9.81
        )
        
        configs[imu_id] = IMUConfig(
            imu_id=imu_id,
            calibration=calib,
            weight=1.0,
            trust_factor=1.0 / noise_scale  # Better sensors get higher trust
        )
    
    return configs