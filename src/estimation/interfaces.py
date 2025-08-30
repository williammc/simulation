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
    
    # Ideal/Normalized coordinates (camera-model independent)
    # These are in the ideal pinhole camera plane (z=1)
    observed_ideal: Optional[np.ndarray] = None  # shape: (2,) - undistorted normalized coordinates
    predicted_ideal: Optional[np.ndarray] = None  # shape: (2,) - predicted normalized coordinates
    ideal_residual: Optional[np.ndarray] = None  # shape: (2,) - residual in ideal plane
    
    # Optional pre-computed Jacobians for optimization
    jacobian_wrt_pose: Optional[np.ndarray] = None  # shape: (2, 6) for SE(3) or (2, 15) for full state
    jacobian_wrt_landmark: Optional[np.ndarray] = None  # shape: (2, 3)
    
    # Jacobians in ideal/normalized coordinates (more stable for optimization)
    ideal_jacobian_wrt_pose: Optional[np.ndarray] = None  # shape: (2, 6) or (2, 15)
    ideal_jacobian_wrt_landmark: Optional[np.ndarray] = None  # shape: (2, 3)
    
    # Bearing vector and depth (for triangulation)
    bearing_vector: Optional[np.ndarray] = None  # shape: (3,) - unit vector in camera frame
    estimated_depth: Optional[float] = None  # estimated distance along bearing vector
    
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
        
        # Ensure ideal coordinates if provided
        if self.observed_ideal is not None:
            self.observed_ideal = np.asarray(self.observed_ideal).reshape(2)
        if self.predicted_ideal is not None:
            self.predicted_ideal = np.asarray(self.predicted_ideal).reshape(2)
        if self.ideal_residual is not None:
            self.ideal_residual = np.asarray(self.ideal_residual).reshape(2)
        elif self.observed_ideal is not None and self.predicted_ideal is not None:
            self.ideal_residual = self.observed_ideal - self.predicted_ideal
        
        # Ensure bearing vector is unit length if provided
        if self.bearing_vector is not None:
            self.bearing_vector = np.asarray(self.bearing_vector).reshape(3)
            norm = np.linalg.norm(self.bearing_vector)
            if norm > 0:
                self.bearing_vector = self.bearing_vector / norm
        
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
    
    @property
    def has_ideal_coordinates(self) -> bool:
        """Check if ideal/normalized coordinates are available."""
        return (
            self.observed_ideal is not None and
            self.predicted_ideal is not None
        )
    
    @property
    def ideal_weighted_residual(self) -> np.ndarray:
        """Get ideal residual weighted by robust weight."""
        if self.ideal_residual is None:
            return np.zeros(2)
        return self.ideal_residual * self.robust_weight


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
    # Frame indices
    from_frame_id: int
    to_frame_id: int
    
    # Pre-integrated changes
    delta_position: np.ndarray  # shape: (3,)
    delta_velocity: np.ndarray  # shape: (3,)
    delta_rotation: np.ndarray  # shape: (3, 3) rotation matrix
    
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
        
        # Handle rotation matrix only (no quaternion support)
        delta_rot = np.asarray(self.delta_rotation)
        if delta_rot.shape == (3, 3):
            self.delta_rotation = delta_rot
        else:
            raise ValueError(f"Invalid rotation shape: {delta_rot.shape}. Only (3,3) rotation matrices supported.")
        
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
        return self.delta_rotation  # Always a 3x3 matrix now


@dataclass
class IdealProjection:
    """
    Ideal/normalized projection result.
    
    Represents projection onto the ideal pinhole camera plane (z=1),
    independent of camera intrinsics and distortion.
    """
    ideal_point: np.ndarray  # shape: (2,) - point on ideal plane
    pixel_point: np.ndarray  # shape: (2,) - corresponding pixel after camera model
    jacobian_ideal_to_pixel: Optional[np.ndarray] = None  # shape: (2, 2) - transformation Jacobian
    pixel_covariance: Optional[np.ndarray] = None  # shape: (2, 2) - uncertainty in pixel space
    ideal_covariance: Optional[np.ndarray] = None  # shape: (2, 2) - uncertainty in ideal space
    
    def __post_init__(self):
        """Ensure proper array shapes."""
        self.ideal_point = np.asarray(self.ideal_point).reshape(2)
        self.pixel_point = np.asarray(self.pixel_point).reshape(2)
        if self.jacobian_ideal_to_pixel is not None:
            self.jacobian_ideal_to_pixel = np.asarray(self.jacobian_ideal_to_pixel).reshape(2, 2)
        if self.pixel_covariance is not None:
            self.pixel_covariance = np.asarray(self.pixel_covariance).reshape(2, 2)
        if self.ideal_covariance is not None:
            self.ideal_covariance = np.asarray(self.ideal_covariance).reshape(2, 2)


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
    
    def project_ideal(
        self,
        point_3d: np.ndarray,  # shape: (3,) in camera frame
        compute_jacobians: bool = False
    ) -> IdealProjection:
        """
        Project 3D point to both ideal plane and pixel coordinates.
        
        The ideal plane is the normalized camera plane at z=1, which is
        independent of camera intrinsics and distortion. This enables
        camera-model-independent algorithms.
        
        Args:
            point_3d: 3D point in camera frame
            compute_jacobians: Whether to compute transformation Jacobians
            
        Returns:
            IdealProjection containing both ideal and pixel coordinates
        """
        ...
    
    def unproject_ideal(
        self,
        ideal_point: np.ndarray,  # shape: (2,) on ideal plane
        depth: float
    ) -> np.ndarray:  # shape: (3,)
        """
        Unproject from ideal plane to 3D point.
        
        Args:
            ideal_point: 2D point on ideal plane (z=1)
            depth: Distance along ray from camera center
            
        Returns:
            3D point in camera frame
        """
        ...
    
    def pixel_to_ideal(
        self,
        pixel: np.ndarray  # shape: (2,)
    ) -> np.ndarray:  # shape: (2,)
        """
        Convert pixel coordinates to ideal/normalized coordinates.
        
        This removes camera intrinsics and distortion, mapping to
        the ideal pinhole plane at z=1.
        
        Args:
            pixel: 2D pixel coordinates
            
        Returns:
            2D point on ideal plane
        """
        ...
    
    def ideal_to_pixel(
        self,
        ideal_point: np.ndarray  # shape: (2,)
    ) -> np.ndarray:  # shape: (2,)
        """
        Convert ideal/normalized coordinates to pixel coordinates.
        
        This applies camera intrinsics and distortion.
        
        Args:
            ideal_point: 2D point on ideal plane
            
        Returns:
            2D pixel coordinates
        """
        ...
    
    def pixel_to_bearing(
        self,
        pixel: np.ndarray  # shape: (2,)
    ) -> np.ndarray:  # shape: (3,)
        """
        Convert pixel to unit bearing vector in camera frame.
        
        This is useful for triangulation and bearing-only measurements.
        
        Args:
            pixel: 2D pixel coordinates
            
        Returns:
            Unit bearing vector in camera frame
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