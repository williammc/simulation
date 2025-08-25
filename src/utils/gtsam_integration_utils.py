"""
GTSAM Integration Utilities

Provides conversion utilities between our data structures and GTSAM,
noise model creation, and other helper functions for GTSAM integration.
"""

import numpy as np
import gtsam
from typing import List, Dict, Optional, Tuple

from src.common.data_structures import (
    IMUMeasurement, IMUCalibration, Pose, 
    PreintegratedIMUData, CameraCalibration
)
from src.estimation.gtsam_imu_preintegration import GTSAMPreintegrationParams


def pose_to_gtsam(pose: Pose) -> gtsam.Pose3:
    """
    Convert our Pose to GTSAM Pose3.
    
    Args:
        pose: Our Pose object
    
    Returns:
        GTSAM Pose3
    """
    return gtsam.Pose3(
        gtsam.Rot3(pose.rotation_matrix),
        gtsam.Point3(pose.position)
    )


def gtsam_to_pose(gtsam_pose: gtsam.Pose3, timestamp: float = 0.0) -> Pose:
    """
    Convert GTSAM Pose3 to our Pose.
    
    Args:
        gtsam_pose: GTSAM Pose3
        timestamp: Timestamp for the pose
    
    Returns:
        Our Pose object
    """
    return Pose(
        timestamp=timestamp,
        position=gtsam_pose.translation(),
        rotation_matrix=gtsam_pose.rotation().matrix()
    )


def create_imu_params(
    calib: IMUCalibration,
    gravity_magnitude: Optional[float] = None
) -> gtsam.PreintegrationParams:
    """
    Create GTSAM PreintegrationParams from IMU calibration.
    
    Args:
        calib: IMU calibration
        gravity_magnitude: Gravity magnitude (uses calib value if None)
    
    Returns:
        GTSAM PreintegrationParams
    """
    if gravity_magnitude is None:
        gravity_magnitude = calib.gravity_magnitude if hasattr(calib, 'gravity_magnitude') else 9.81
    
    # Create params with gravity along negative Z
    params = gtsam.PreintegrationParams.MakeSharedU(gravity_magnitude)
    
    # Set noise parameters (continuous-time)
    params.setAccelerometerCovariance(
        calib.accelerometer_noise_density**2 * np.eye(3)
    )
    params.setGyroscopeCovariance(
        calib.gyroscope_noise_density**2 * np.eye(3)
    )
    
    # Set integration covariance (small value for numerical stability)
    params.setIntegrationCovariance(1e-8 * np.eye(3))
    
    return params


def create_pose_prior_noise(
    position_sigma: float = 0.1,
    rotation_sigma: float = 0.1
) -> gtsam.noiseModel.Diagonal:
    """
    Create noise model for pose prior.
    
    Args:
        position_sigma: Position standard deviation (meters)
        rotation_sigma: Rotation standard deviation (radians)
    
    Returns:
        Diagonal noise model for pose (6D: rotation, position)
    """
    sigmas = np.array([
        rotation_sigma, rotation_sigma, rotation_sigma,  # Rotation
        position_sigma, position_sigma, position_sigma   # Position
    ])
    return gtsam.noiseModel.Diagonal.Sigmas(sigmas)


def create_velocity_prior_noise(velocity_sigma: float = 0.1) -> gtsam.noiseModel.Diagonal:
    """
    Create noise model for velocity prior.
    
    Args:
        velocity_sigma: Velocity standard deviation (m/s)
    
    Returns:
        Diagonal noise model for velocity (3D)
    """
    sigmas = np.array([velocity_sigma, velocity_sigma, velocity_sigma])
    return gtsam.noiseModel.Diagonal.Sigmas(sigmas)


def create_bias_prior_noise(
    accel_bias_sigma: float = 0.1,
    gyro_bias_sigma: float = 0.01
) -> gtsam.noiseModel.Diagonal:
    """
    Create noise model for bias prior.
    
    Args:
        accel_bias_sigma: Accelerometer bias standard deviation
        gyro_bias_sigma: Gyroscope bias standard deviation
    
    Returns:
        Diagonal noise model for bias (6D)
    """
    sigmas = np.array([
        accel_bias_sigma, accel_bias_sigma, accel_bias_sigma,  # Accel bias
        gyro_bias_sigma, gyro_bias_sigma, gyro_bias_sigma      # Gyro bias
    ])
    return gtsam.noiseModel.Diagonal.Sigmas(sigmas)


def create_between_bias_noise(
    accel_bias_rw: float = 1e-3,
    gyro_bias_rw: float = 1e-5,
    dt: float = 1.0
) -> gtsam.noiseModel.Diagonal:
    """
    Create noise model for bias random walk between keyframes.
    
    Args:
        accel_bias_rw: Accelerometer bias random walk
        gyro_bias_rw: Gyroscope bias random walk
        dt: Time interval
    
    Returns:
        Diagonal noise model for bias change
    """
    # Scale by sqrt(dt) for random walk
    sigmas = np.sqrt(dt) * np.array([
        accel_bias_rw, accel_bias_rw, accel_bias_rw,
        gyro_bias_rw, gyro_bias_rw, gyro_bias_rw
    ])
    return gtsam.noiseModel.Diagonal.Sigmas(sigmas)


def extract_imu_between_keyframes(
    all_measurements: List[IMUMeasurement],
    from_time: float,
    to_time: float
) -> List[IMUMeasurement]:
    """
    Extract IMU measurements between two keyframe timestamps.
    
    Args:
        all_measurements: All IMU measurements
        from_time: Start timestamp (inclusive)
        to_time: End timestamp (exclusive)
    
    Returns:
        List of measurements in the time interval
    """
    result = []
    for meas in all_measurements:
        if from_time <= meas.timestamp < to_time:
            result.append(meas)
    return result


def compute_initial_velocity(
    pose_i: Pose,
    pose_j: Pose,
    dt: float
) -> np.ndarray:
    """
    Compute initial velocity estimate from two poses.
    
    Args:
        pose_i: First pose
        pose_j: Second pose
        dt: Time interval
    
    Returns:
        Estimated velocity at pose_i
    """
    if dt <= 0:
        return np.zeros(3)
    
    # Simple finite difference
    velocity = (pose_j.position - pose_i.position) / dt
    return velocity


def create_camera_projection_factor(
    pose_key: int,
    landmark_key: int,
    measurement: np.ndarray,
    calib: CameraCalibration,
    pixel_noise: float = 1.0
) -> gtsam.GenericProjectionFactorCal3_S2:
    """
    Create camera projection factor.
    
    Args:
        pose_key: Pose variable key
        landmark_key: Landmark variable key
        measurement: 2D pixel measurement
        calib: Camera calibration
        pixel_noise: Pixel measurement noise (standard deviation)
    
    Returns:
        Projection factor for the measurement
    """
    # Create GTSAM camera calibration
    K = gtsam.Cal3_S2(
        calib.intrinsics.fx,
        calib.intrinsics.fy,
        0.0,  # No skew
        calib.intrinsics.cx,
        calib.intrinsics.cy
    )
    
    # Create noise model
    noise = gtsam.noiseModel.Isotropic.Sigma(2, pixel_noise)
    
    # Create projection factor
    factor = gtsam.GenericProjectionFactorCal3_S2(
        measurement,
        noise,
        pose_key,
        landmark_key,
        K
    )
    
    return factor


class FactorGraphBuilder:
    """
    Helper class to build GTSAM factor graphs with proper key management.
    """
    
    def __init__(self):
        """Initialize factor graph builder."""
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_values = gtsam.Values()
        self.pose_count = 0
        self.landmark_count = 0
        self.bias_count = 0
        
        # Key generation functions
        self.X = lambda i: gtsam.symbol('x', i)  # Poses
        self.V = lambda i: gtsam.symbol('v', i)  # Velocities
        self.B = lambda i: gtsam.symbol('b', i)  # Biases
        self.L = lambda i: gtsam.symbol('l', i)  # Landmarks
    
    def add_pose_prior(
        self,
        pose_idx: int,
        pose: gtsam.Pose3,
        noise: gtsam.noiseModel.Base
    ):
        """Add pose prior factor."""
        key = self.X(pose_idx)
        self.graph.add(gtsam.PriorFactorPose3(key, pose, noise))
        if not self.initial_values.exists(key):
            self.initial_values.insert(key, pose)
    
    def add_velocity_prior(
        self,
        vel_idx: int,
        velocity: np.ndarray,
        noise: gtsam.noiseModel.Base
    ):
        """Add velocity prior factor."""
        key = self.V(vel_idx)
        self.graph.add(gtsam.PriorFactorVector(key, velocity, noise))
        if not self.initial_values.exists(key):
            self.initial_values.insert(key, velocity)
    
    def add_bias_prior(
        self,
        bias_idx: int,
        bias: gtsam.imuBias.ConstantBias,
        noise: gtsam.noiseModel.Base
    ):
        """Add bias prior factor."""
        key = self.B(bias_idx)
        self.graph.add(gtsam.PriorFactorConstantBias(key, bias, noise))
        if not self.initial_values.exists(key):
            self.initial_values.insert(key, bias)
    
    def add_combined_imu_factor(
        self,
        pose_i_idx: int,
        vel_i_idx: int,
        pose_j_idx: int,
        vel_j_idx: int,
        bias_i_idx: int,
        bias_j_idx: int,
        pim: gtsam.PreintegratedImuMeasurements
    ):
        """Add CombinedImuFactor to the graph."""
        factor = gtsam.CombinedImuFactor(
            self.X(pose_i_idx), self.V(vel_i_idx),
            self.X(pose_j_idx), self.V(vel_j_idx),
            self.B(bias_i_idx), self.B(bias_j_idx),
            pim
        )
        self.graph.add(factor)
    
    def add_between_bias_factor(
        self,
        bias_i_idx: int,
        bias_j_idx: int,
        bias_change: gtsam.imuBias.ConstantBias,
        noise: gtsam.noiseModel.Base
    ):
        """Add between factor for bias random walk."""
        factor = gtsam.BetweenFactorConstantBias(
            self.B(bias_i_idx),
            self.B(bias_j_idx),
            bias_change,
            noise
        )
        self.graph.add(factor)
    
    def optimize(
        self,
        optimizer_params: Optional[gtsam.LevenbergMarquardtParams] = None
    ) -> gtsam.Values:
        """
        Optimize the factor graph.
        
        Args:
            optimizer_params: Optimizer parameters
        
        Returns:
            Optimized values
        """
        if optimizer_params is None:
            optimizer_params = gtsam.LevenbergMarquardtParams()
            optimizer_params.setVerbosity("ERROR")
        
        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self.graph,
            self.initial_values,
            optimizer_params
        )
        
        return optimizer.optimize()
    
    def optimize_isam2(
        self,
        isam2_params: Optional[gtsam.ISAM2Params] = None
    ) -> gtsam.ISAM2:
        """
        Create and update ISAM2 with current graph.
        
        Args:
            isam2_params: ISAM2 parameters
        
        Returns:
            Updated ISAM2 instance
        """
        if isam2_params is None:
            isam2_params = gtsam.ISAM2Params()
            isam2_params.relinearizeThreshold = 0.01
            isam2_params.relinearizeSkip = 1
        
        isam2 = gtsam.ISAM2(isam2_params)
        isam2.update(self.graph, self.initial_values)
        
        return isam2