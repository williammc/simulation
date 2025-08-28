"""
Preprocessing pipeline for converting raw sensor data to camera-model-independent measurements.
"""

from typing import Optional, Dict, Any
import numpy as np
import logging

from src.estimation.interfaces import (
    VisualMeasurement, 
    ProcessedVisualFrame,
    ProjectionInterface
)
from src.common.data_structures import (
    CameraFrame,
    Pose,
    Map,
    TrajectoryState
)

# Create State alias for compatibility
State = TrajectoryState

logger = logging.getLogger(__name__)


class VisualMeasurementPreprocessor:
    """
    Converts raw camera frames to processed measurements.
    
    This class handles the conversion of raw camera observations to
    pre-processed measurements that can be used by camera-model-independent
    estimators.
    """
    
    def __init__(
        self,
        projection_service: ProjectionInterface,
        pixel_noise_std: float = 1.0,
        robust_kernel: Optional[str] = None,
        huber_delta: float = 1.0
    ):
        """
        Initialize the preprocessor.
        
        Args:
            projection_service: Service for projection operations
            pixel_noise_std: Standard deviation of pixel noise
            robust_kernel: Type of robust kernel ('huber', 'cauchy', None)
            huber_delta: Delta parameter for Huber kernel
        """
        self.projection_service = projection_service
        self.pixel_noise_std = pixel_noise_std
        self.robust_kernel = robust_kernel
        self.huber_delta = huber_delta
        
        # Default pixel covariance
        self.default_pixel_covariance = np.eye(2) * (pixel_noise_std ** 2)
    
    def process_frame(
        self,
        raw_frame: CameraFrame,
        current_state: State,
        landmarks: Map,
        compute_jacobians: bool = True,
        chi2_threshold: Optional[float] = None
    ) -> ProcessedVisualFrame:
        """
        Process raw camera frame into measurements.
        
        Args:
            raw_frame: Raw camera observations
            current_state: Current state estimate
            landmarks: Map of known landmarks
            compute_jacobians: Whether to compute Jacobians
            chi2_threshold: Chi-squared threshold for outlier rejection
            
        Returns:
            ProcessedVisualFrame with pre-processed measurements
        """
        # Create processed frame container
        processed = ProcessedVisualFrame(
            timestamp=raw_frame.timestamp,
            frame_id=getattr(raw_frame, 'frame_id', -1),
            is_keyframe=raw_frame.is_keyframe,
            keyframe_id=raw_frame.keyframe_id,
            camera_id=raw_frame.camera_id,
            measurements=[],
            predicted_pose=current_state.pose
        )
        
        # Process each observation
        num_outliers = 0
        for obs in raw_frame.observations:
            # Get landmark from map
            landmark = landmarks.get_landmark(obs.landmark_id)
            if landmark is None:
                logger.debug(f"Landmark {obs.landmark_id} not found in map")
                continue
            
            # Project landmark to get predicted measurement
            try:
                pred_meas = self.projection_service.project(
                    landmark.position,
                    current_state.pose,
                    compute_jacobians=compute_jacobians
                )
            except Exception as e:
                logger.warning(f"Failed to project landmark {obs.landmark_id}: {e}")
                continue
            
            # Convert observation pixel to numpy array
            observed_pixel = np.array([obs.pixel.u, obs.pixel.v])
            
            # Compute residual
            residual = observed_pixel - pred_meas.predicted_pixel
            
            # Check for outliers using chi-squared test if threshold provided
            if chi2_threshold is not None:
                chi2 = residual.T @ np.linalg.inv(self.default_pixel_covariance) @ residual
                if chi2 > chi2_threshold:
                    num_outliers += 1
                    logger.debug(f"Landmark {obs.landmark_id} rejected as outlier (chi2={chi2:.2f})")
                    continue
            
            # Compute ideal coordinates and bearing vectors
            observed_ideal = None
            predicted_ideal = None
            ideal_residual = None
            bearing_vector = None
            ideal_jacobian_wrt_pose = None
            ideal_jacobian_wrt_landmark = None
            
            try:
                # Convert observed pixel to ideal coordinates
                observed_ideal = self.projection_service.pixel_to_ideal(observed_pixel)
                
                # Convert predicted pixel to ideal coordinates
                predicted_ideal = self.projection_service.pixel_to_ideal(pred_meas.predicted_pixel)
                
                # Compute ideal residual
                ideal_residual = observed_ideal - predicted_ideal
                
                # Get bearing vector for observed pixel
                bearing_vector = self.projection_service.pixel_to_bearing(observed_pixel)
                
                # If we have Jacobians, transform them to ideal space
                if compute_jacobians and pred_meas.jacobian_wrt_pose is not None:
                    # Get the transformation Jacobian from pixel to ideal
                    # This is approximately the inverse of the intrinsics matrix
                    if hasattr(self.projection_service, 'camera_calib'):
                        calib = self.projection_service.camera_calib
                        if hasattr(calib, 'intrinsics'):
                            fx = calib.intrinsics.fx
                            fy = calib.intrinsics.fy
                        elif hasattr(calib, 'K'):
                            fx = calib.K[0,0]
                            fy = calib.K[1,1]
                        else:
                            fx = fy = 500.0
                    else:
                        fx = fy = 500.0
                    
                    # Jacobian of ideal w.r.t. pixel is diag(1/fx, 1/fy)
                    J_ideal_pixel = np.diag([1.0/fx, 1.0/fy])
                    
                    # Transform Jacobians to ideal space
                    ideal_jacobian_wrt_pose = J_ideal_pixel @ pred_meas.jacobian_wrt_pose
                    if pred_meas.jacobian_wrt_landmark is not None:
                        ideal_jacobian_wrt_landmark = J_ideal_pixel @ pred_meas.jacobian_wrt_landmark
                    
            except Exception as e:
                logger.debug(f"Could not compute ideal coordinates: {e}")
            
            # Create visual measurement with ideal coordinates
            vis_meas = VisualMeasurement(
                landmark_id=obs.landmark_id,
                observed_pixel=observed_pixel,
                predicted_pixel=pred_meas.predicted_pixel,
                residual=residual,
                pixel_covariance=pred_meas.pixel_covariance if pred_meas.pixel_covariance is not None else self.default_pixel_covariance,
                jacobian_wrt_pose=pred_meas.jacobian_wrt_pose,
                jacobian_wrt_landmark=pred_meas.jacobian_wrt_landmark,
                # New ideal/normalized fields
                observed_ideal=observed_ideal,
                predicted_ideal=predicted_ideal,
                ideal_residual=ideal_residual,
                ideal_jacobian_wrt_pose=ideal_jacobian_wrt_pose,
                ideal_jacobian_wrt_landmark=ideal_jacobian_wrt_landmark,
                bearing_vector=bearing_vector
            )
            
            # Apply robust kernel if configured
            if self.robust_kernel:
                vis_meas.robust_weight = self.compute_robust_weight(
                    vis_meas.residual,
                    vis_meas.pixel_covariance
                )
            
            processed.measurements.append(vis_meas)
        
        if num_outliers > 0:
            logger.info(f"Frame {raw_frame.timestamp}: {num_outliers} outliers rejected")
        
        return processed
    
    def compute_robust_weight(
        self,
        residual: np.ndarray,
        covariance: np.ndarray
    ) -> float:
        """
        Compute robust weight for a residual.
        
        Args:
            residual: Measurement residual
            covariance: Measurement covariance
            
        Returns:
            Robust weight in [0, 1]
        """
        # Compute normalized residual (Mahalanobis distance)
        try:
            normalized_residual = np.sqrt(residual.T @ np.linalg.inv(covariance) @ residual)
        except np.linalg.LinAlgError:
            normalized_residual = np.linalg.norm(residual) / self.pixel_noise_std
        
        if self.robust_kernel == 'huber':
            return self._huber_weight(normalized_residual)
        elif self.robust_kernel == 'cauchy':
            return self._cauchy_weight(normalized_residual)
        else:
            return 1.0
    
    def _huber_weight(self, r: float) -> float:
        """Huber robust weight function."""
        if r <= self.huber_delta:
            return 1.0
        else:
            return self.huber_delta / r
    
    def _cauchy_weight(self, r: float, c: float = 2.3849) -> float:
        """Cauchy robust weight function."""
        return 1.0 / (1.0 + (r / c) ** 2)
    
    def process_batch(
        self,
        raw_frames: list[CameraFrame],
        states: list[State],
        landmarks: Map,
        compute_jacobians: bool = True
    ) -> list[ProcessedVisualFrame]:
        """
        Process multiple frames in batch.
        
        Args:
            raw_frames: List of raw camera frames
            states: Corresponding state estimates
            landmarks: Map of landmarks
            compute_jacobians: Whether to compute Jacobians
            
        Returns:
            List of processed frames
        """
        if len(raw_frames) != len(states):
            raise ValueError("Number of frames and states must match")
        
        processed_frames = []
        for frame, state in zip(raw_frames, states):
            processed = self.process_frame(
                frame, state, landmarks,
                compute_jacobians=compute_jacobians
            )
            processed_frames.append(processed)
        
        return processed_frames


class IMUPreprocessor:
    """
    Preprocessor for IMU measurements.
    
    This is mostly a pass-through since IMU data is already pre-integrated,
    but provides a consistent interface and can add additional processing.
    """
    
    def __init__(self, gravity_magnitude: float = 9.81):
        """
        Initialize IMU preprocessor.
        
        Args:
            gravity_magnitude: Magnitude of gravity vector
        """
        self.gravity_magnitude = gravity_magnitude
    
    def validate_preintegration(
        self,
        preintegrated_data: Any,
        expected_dt: Optional[float] = None
    ) -> bool:
        """
        Validate pre-integrated IMU data.
        
        Args:
            preintegrated_data: Pre-integrated IMU measurements
            expected_dt: Expected time interval (if known)
            
        Returns:
            True if data is valid
        """
        # Check required fields exist
        required_fields = [
            'delta_position', 'delta_velocity', 'delta_rotation',
            'covariance', 'delta_t'
        ]
        
        for field in required_fields:
            if not hasattr(preintegrated_data, field):
                logger.warning(f"Missing required field: {field}")
                return False
        
        # Check time interval
        if expected_dt is not None:
            if abs(preintegrated_data.delta_t - expected_dt) > 1e-3:
                logger.warning(
                    f"Unexpected dt: {preintegrated_data.delta_t} vs {expected_dt}"
                )
                return False
        
        # Check for NaN or Inf
        arrays_to_check = [
            ('delta_position', preintegrated_data.delta_position),
            ('delta_velocity', preintegrated_data.delta_velocity),
            ('covariance', preintegrated_data.covariance)
        ]
        
        for name, arr in arrays_to_check:
            try:
                if not np.isfinite(arr).all():
                    logger.warning(f"Non-finite values in {name}")
                    return False
            except (TypeError, AttributeError) as e:
                logger.warning(f"Invalid array for {name}: {e}")
                return False
        
        return True