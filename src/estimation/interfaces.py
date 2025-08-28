"""
Camera-model-independent interfaces for estimators.

This module defines data structures and interfaces that allow estimators
to work with pre-processed measurements without depending on camera models.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Protocol
import numpy as np

from src.common.data_structures import Pose


@dataclass
class VisualMeasurement:
    """
    Pre-processed visual measurement, camera-model independent.
    
    This structure contains all information needed for visual updates
    without requiring the estimator to perform projection operations.
    """
    # Landmark identification
    landmark_id: int
    
    # Pixel coordinates
    observed_pixel: np.ndarray  # shape: (2,) - actual camera observation
    predicted_pixel: np.ndarray  # shape: (2,) - predicted from current state
    
    # Pre-computed residual (observed - predicted)
    residual: np.ndarray  # shape: (2,)
    
    # Measurement uncertainty
    pixel_covariance: np.ndarray  # shape: (2, 2)
    
    # Optional pre-computed Jacobians for optimization
    jacobian_wrt_pose: Optional[np.ndarray] = None  # shape: (2, 6) for SE(3) or (2, 15) for full state
    jacobian_wrt_landmark: Optional[np.ndarray] = None  # shape: (2, 3)
    
    # Pre-computed robust weight (1.0 for no robustification)
    robust_weight: float = 1.0
    
    # Information matrix (inverse of covariance)
    information_matrix: Optional[np.ndarray] = None  # shape: (2, 2)
    
    def __post_init__(self):
        """Validate and compute derived fields."""
        # Ensure numpy arrays
        self.observed_pixel = np.asarray(self.observed_pixel).reshape(2)
        self.predicted_pixel = np.asarray(self.predicted_pixel).reshape(2)
        self.residual = np.asarray(self.residual).reshape(2)
        self.pixel_covariance = np.asarray(self.pixel_covariance).reshape(2, 2)
        
        # Compute information matrix if not provided
        if self.information_matrix is None:
            try:
                self.information_matrix = np.linalg.inv(self.pixel_covariance)
            except np.linalg.LinAlgError:
                # If covariance is singular, use pseudo-inverse
                self.information_matrix = np.linalg.pinv(self.pixel_covariance)
    
    @property
    def weighted_residual(self) -> np.ndarray:
        """Get residual weighted by robust weight."""
        return self.residual * self.robust_weight
    
    @property
    def is_valid(self) -> bool:
        """Check if measurement is valid for use in optimization."""
        return (
            self.robust_weight > 0.0 and
            np.isfinite(self.residual).all() and
            np.isfinite(self.pixel_covariance).all()
        )


@dataclass
class ProcessedVisualFrame:
    """
    Container for pre-processed visual measurements from one camera frame.
    
    This replaces CameraFrame for camera-model-independent estimators.
    """
    timestamp: float
    frame_id: int
    is_keyframe: bool
    keyframe_id: Optional[int] = None
    measurements: List[VisualMeasurement] = field(default_factory=list)
    
    # Optional: pre-computed pose prediction for this frame
    predicted_pose: Optional[Pose] = None
    
    # Camera ID for multi-camera systems
    camera_id: str = "cam0"
    
    @property
    def num_measurements(self) -> int:
        """Number of visual measurements in this frame."""
        return len(self.measurements)
    
    @property
    def valid_measurements(self) -> List[VisualMeasurement]:
        """Get only valid measurements."""
        return [m for m in self.measurements if m.is_valid]
    
    @property
    def total_residual_norm(self) -> float:
        """Compute total residual norm across all measurements."""
        if not self.measurements:
            return 0.0
        residuals = np.concatenate([m.weighted_residual for m in self.valid_measurements])
        return np.linalg.norm(residuals)
    
    def get_measurement_for_landmark(self, landmark_id: int) -> Optional[VisualMeasurement]:
        """Get measurement for a specific landmark if it exists."""
        for measurement in self.measurements:
            if measurement.landmark_id == landmark_id:
                return measurement
        return None


@dataclass
class PreprocessedIMUData:
    """
    IMU measurements with pre-computed integration and embedded noise model.
    
    This structure contains pre-integrated IMU data with all noise parameters
    already applied, so estimators don't need IMU calibration.
    """
    # Keyframe indices
    from_keyframe_id: int
    to_keyframe_id: int
    
    # Pre-integrated changes
    delta_position: np.ndarray  # shape: (3,)
    delta_velocity: np.ndarray  # shape: (3,)
    delta_rotation: np.ndarray  # shape: (3, 3) rotation matrix or (4,) quaternion
    
    # Pre-computed covariance including noise model
    covariance: np.ndarray  # shape: (9, 9) for [rotation, velocity, position] or (15, 15) with biases
    
    # Time interval
    delta_t: float
    
    # Pre-computed Jacobians w.r.t. biases (optional)
    jacobian_wrt_accel_bias: Optional[np.ndarray] = None  # shape: (9, 3)
    jacobian_wrt_gyro_bias: Optional[np.ndarray] = None  # shape: (9, 3)
    
    # Number of integrated measurements
    num_measurements: int = 0
    
    # Optional bias estimates
    accel_bias: Optional[np.ndarray] = None  # shape: (3,)
    gyro_bias: Optional[np.ndarray] = None  # shape: (3,)
    
    def __post_init__(self):
        """Validate and ensure proper array shapes."""
        self.delta_position = np.asarray(self.delta_position).reshape(3)
        self.delta_velocity = np.asarray(self.delta_velocity).reshape(3)
        
        # Handle rotation (could be matrix or quaternion)
        delta_rot = np.asarray(self.delta_rotation)
        if delta_rot.shape == (3, 3):
            self.delta_rotation = delta_rot
        elif delta_rot.size == 4:
            self.delta_rotation = delta_rot.reshape(4)
        else:
            raise ValueError(f"Invalid rotation shape: {delta_rot.shape}")
        
        # Ensure covariance is proper size
        cov = np.asarray(self.covariance)
        if cov.shape not in [(9, 9), (15, 15)]:
            # Try to reshape if it's flattened
            if cov.size == 81:
                self.covariance = cov.reshape(9, 9)
            elif cov.size == 225:
                self.covariance = cov.reshape(15, 15)
            else:
                raise ValueError(f"Invalid covariance shape: {cov.shape}")
        else:
            self.covariance = cov
    
    @property
    def has_bias_jacobians(self) -> bool:
        """Check if bias Jacobians are available."""
        return (
            self.jacobian_wrt_accel_bias is not None and
            self.jacobian_wrt_gyro_bias is not None
        )
    
    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get rotation as a 3x3 matrix."""
        if self.delta_rotation.shape == (3, 3):
            return self.delta_rotation
        else:
            # Convert quaternion to rotation matrix
            from src.utils.math_utils import quaternion_to_rotation_matrix
            return quaternion_to_rotation_matrix(self.delta_rotation)


class ProjectionInterface(Protocol):
    """
    Abstract interface for projection operations.
    
    This allows different camera models to be used with the same estimator
    by providing a common interface for projection operations.
    """
    
    def project(
        self, 
        landmark_position: np.ndarray,  # shape: (3,)
        camera_pose: Pose,
        compute_jacobians: bool = False
    ) -> VisualMeasurement:
        """
        Project 3D landmark to image plane with uncertainty.
        
        Args:
            landmark_position: 3D position of landmark in world frame
            camera_pose: Current camera pose estimate
            compute_jacobians: Whether to compute Jacobians
            
        Returns:
            VisualMeasurement with predicted pixel and optional Jacobians
        """
        ...
    
    def unproject(
        self,
        pixel: np.ndarray,  # shape: (2,)
        depth: float
    ) -> np.ndarray:  # shape: (3,)
        """
        Unproject pixel to 3D point given depth.
        
        Args:
            pixel: 2D pixel coordinates
            depth: Distance along optical axis
            
        Returns:
            3D point in camera frame
        """
        ...
    
    def compute_jacobians(
        self,
        landmark_position: np.ndarray,
        camera_pose: Pose
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute projection Jacobians.
        
        Args:
            landmark_position: 3D position of landmark
            camera_pose: Current camera pose
            
        Returns:
            Tuple of (jacobian_wrt_pose, jacobian_wrt_landmark)
        """
        ...