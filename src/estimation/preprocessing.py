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


def skew_matrix(v: np.ndarray) -> np.ndarray:
    """
    Create skew-symmetric matrix from 3D vector.
    
    For vector v = [v1, v2, v3], returns:
    [  0  -v3   v2 ]
    [ v3    0  -v1 ]
    [-v2   v1    0 ]
    
    Used for SO(3) tangent space computations.
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


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
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        if self._debug_count < 3:
            print(f"[Preprocessor] Frame has {len(raw_frame.observations)} observations")
            if landmarks:
                print(f"[Preprocessor] Landmarks map has {len(landmarks.landmarks)} landmarks")
            else:
                print(f"[Preprocessor] No landmarks map provided")
            self._debug_count += 1
            
        found_count = 0
        missing_count = 0
        for obs in raw_frame.observations:
            # Get landmark from map
            landmark = landmarks.get_landmark(obs.landmark_id) if landmarks else None
            if landmark is None:
                missing_count += 1
                if missing_count == 1 and self._debug_count <= 3:
                    print(f"[Preprocessor] Looking for landmark {obs.landmark_id}, not found")
                    if landmarks and hasattr(landmarks, 'landmarks'):
                        sample_ids = list(landmarks.landmarks.keys())[:5]
                        print(f"[Preprocessor] Sample landmark IDs in map: {sample_ids}")
                continue
            found_count += 1
        
        if self._debug_count <= 3:
            print(f"[Preprocessor] Found {found_count}/{len(raw_frame.observations)} landmarks")
        
        # Re-iterate to process observations
        processed_count = 0
        for obs in raw_frame.observations:
            landmark = landmarks.get_landmark(obs.landmark_id) if landmarks else None
            if landmark is None:
                continue
                
            # For ideal-only projection, transform landmark to camera frame and project to ideal plane
            try:
                # Transform landmark to camera frame
                R_world_to_cam = current_state.pose.rotation_matrix.T
                t_world_to_cam = -R_world_to_cam @ current_state.pose.position
                landmark_cam = R_world_to_cam @ landmark.position + t_world_to_cam
                
                # Check if behind camera
                # Use a more permissive threshold to avoid rejecting landmarks due to small pose errors
                if landmark_cam[2] <= -0.5:  # Only reject if clearly behind camera
                    if self._debug_count <= 50 and processed_count == 0:  # Debug more frames
                        print(f"[Preprocessor] Landmark {obs.landmark_id} behind camera: z={landmark_cam[2]:.3f}")
                    continue
                elif landmark_cam[2] <= 0.1:
                    # Landmark is very close or slightly behind - use with caution
                    if self._debug_count <= 50 and processed_count == 0:
                        print(f"[Preprocessor] Warning: Landmark {obs.landmark_id} very close: z={landmark_cam[2]:.3f}")
                    # Still process it but with reduced weight later
                
                # Project to ideal plane (z=1)
                predicted_ideal = np.array([
                    landmark_cam[0] / landmark_cam[2],
                    landmark_cam[1] / landmark_cam[2]
                ])
                
                if self._debug_count <= 3 and processed_count == 0:
                    print(f"[Preprocessor] Landmark {obs.landmark_id} in cam frame: {landmark_cam}, ideal: {predicted_ideal}")
                    
            except Exception as e:
                if self._debug_count <= 3 and processed_count == 0:
                    print(f"[Preprocessor] Failed to project landmark {obs.landmark_id}: {e}")
                logger.warning(f"Failed to project landmark {obs.landmark_id}: {e}")
                continue
            processed_count += 1
            
            # Convert observation pixel to ideal coordinates
            observed_pixel = np.array([obs.pixel.u, obs.pixel.v])
            
            # Convert pixel to ideal using simple pinhole model
            calib = self.projection_service.camera_calib
            K = np.array([
                [calib.intrinsics.fx, 0, calib.intrinsics.cx],
                [0, calib.intrinsics.fy, calib.intrinsics.cy],
                [0, 0, 1]
            ])
            
            observed_ideal = np.array([
                (observed_pixel[0] - K[0, 2]) / K[0, 0],
                (observed_pixel[1] - K[1, 2]) / K[1, 1]
            ])
            
            # Compute ideal residual
            ideal_residual = observed_ideal - predicted_ideal
            
            # Debug residual
            if self._debug_count <= 3 and processed_count == 1:
                print(f"[Preprocessor] Observed ideal: {observed_ideal}, Predicted ideal: {predicted_ideal}")
                print(f"[Preprocessor] Ideal residual: {ideal_residual}, norm: {np.linalg.norm(ideal_residual):.4f}")
            
            # Check for outliers using ideal residual
            if False and chi2_threshold is not None:  # TEMPORARILY DISABLED
                # Use reasonable covariance for ideal coordinates
                # Ideal coords have typical range of [-1, 1], so use appropriate variance
                ideal_covariance = np.eye(2) * 0.01  # Reasonable for normalized coordinates
                chi2 = ideal_residual.T @ np.linalg.inv(ideal_covariance) @ ideal_residual
                if chi2 > chi2_threshold:
                    num_outliers += 1
                    if self._debug_count <= 3 and num_outliers == 1:
                        print(f"[Preprocessor] Landmark {obs.landmark_id} rejected as outlier (chi2={chi2:.2f} > {chi2_threshold})")
                    logger.debug(f"Landmark {obs.landmark_id} rejected as outlier (chi2={chi2:.2f})")
                    continue
            
            # Compute bearing vector from observed ideal coordinates
            bearing_vector = np.array([observed_ideal[0], observed_ideal[1], 1.0])
            bearing_vector = bearing_vector / np.linalg.norm(bearing_vector)
            
            # Compute Jacobians for ideal coordinates if requested
            if compute_jacobians:
                # We have: landmark_cam = R^T @ (landmark_world - t)
                # And: ideal = [x/z, y/z] where [x,y,z] = landmark_cam
                
                # Get world landmark position (we need this for Jacobian computation)
                landmark_world = landmark.position  # We already have this from earlier
                
                # 1. Jacobian of ideal coordinates w.r.t camera point
                x, y, z = landmark_cam
                z2 = z * z
                J_ideal_pcam = np.array([
                    [1/z, 0, -x/z2],
                    [0, 1/z, -y/z2]
                ])  # 2x3
                
                # 2. Jacobian of camera point w.r.t world position (translation)
                R = current_state.pose.rotation_matrix
                J_pcam_t = -R.T  # 3x3
                
                # 3. Jacobian of camera point w.r.t rotation (using SO3 tangent)
                # Using the standard pinhole projection Jacobian formulation
                # For rotation perturbation δω, the camera point changes as:
                # δp_cam = -[p_cam]_× @ δω (rotation acts on the point)
                # This gives us the Jacobian
                J_pcam_omega = -skew_matrix(landmark_cam)  # 3x3
                
                # 4. Combine for full pose Jacobian [position, rotation]
                J_pcam_pose = np.hstack([J_pcam_t, J_pcam_omega])  # 3x6
                ideal_jacobian_wrt_pose = J_ideal_pcam @ J_pcam_pose  # 2x6
                
                # Debug Jacobian magnitudes (only for first few)
                if self._debug_count <= 1 and processed_count == 1:
                    print(f"[Preprocessor] Jacobian magnitudes:")
                    print(f"  J_ideal_pcam norm: {np.linalg.norm(J_ideal_pcam):.4f}")
                    print(f"  J_pcam_t norm: {np.linalg.norm(J_pcam_t):.4f}")
                    print(f"  J_pcam_omega norm: {np.linalg.norm(J_pcam_omega):.4f}")
                    print(f"  ideal_jacobian_wrt_pose norm: {np.linalg.norm(ideal_jacobian_wrt_pose):.4f}")
                
                # 5. Jacobian w.r.t landmark position
                # ideal depends on landmark through p_cam = R^T @ (p_w - t)
                # So J_pcam_landmark = R^T
                ideal_jacobian_wrt_landmark = J_ideal_pcam @ R.T  # 2x3
            else:
                ideal_jacobian_wrt_pose = None
                ideal_jacobian_wrt_landmark = None
            
            # Skip the old projection service code
            if False:
                # Old code that uses projection service
                observed_ideal_old = self.projection_service.pixel_to_ideal(observed_pixel)
                
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
            
            # Compute predicted pixel from ideal coordinates (for compatibility)
            predicted_pixel = np.array([
                predicted_ideal[0] * K[0, 0] + K[0, 2],
                predicted_ideal[1] * K[1, 1] + K[1, 2]
            ])
            
            # Compute pixel residual
            pixel_residual = observed_pixel - predicted_pixel
            
            # Create visual measurement with ideal coordinates
            vis_meas = VisualMeasurement(
                landmark_id=obs.landmark_id,
                observed_pixel=observed_pixel,
                predicted_pixel=predicted_pixel,
                residual=pixel_residual,
                pixel_covariance=self.default_pixel_covariance,
                jacobian_wrt_pose=None,  # Not using pixel jacobians
                jacobian_wrt_landmark=None,
                # Ideal/normalized fields (what we actually use)
                observed_ideal=observed_ideal,
                predicted_ideal=predicted_ideal,
                ideal_residual=ideal_residual,
                ideal_jacobian_wrt_pose=ideal_jacobian_wrt_pose,
                ideal_jacobian_wrt_landmark=ideal_jacobian_wrt_landmark,
                bearing_vector=bearing_vector
            )
            
            # Apply robust kernel if configured (using ideal residual)
            if self.robust_kernel:
                vis_meas.robust_weight = self.compute_robust_weight(
                    ideal_residual,
                    np.eye(2) * 0.001  # Ideal covariance
                )
            
            processed.measurements.append(vis_meas)
        
        if num_outliers > 0:
            logger.info(f"Frame {raw_frame.timestamp}: {num_outliers} outliers rejected")
        
        if self._debug_count <= 3:
            print(f"[Preprocessor] Created {len(processed.measurements)} measurements")
        
        # Warn if no measurements created despite having observations
        if len(processed.measurements) == 0 and len(raw_frame.observations) > 0:
            if not hasattr(self, '_no_meas_warning_count'):
                self._no_meas_warning_count = 0
            if self._no_meas_warning_count < 5:
                logger.warning(f"Frame {raw_frame.timestamp:.3f}: No measurements created from {len(raw_frame.observations)} observations!")
                logger.warning(f"  Current pose: pos={current_state.pose.position}, z={current_state.pose.position[2]:.3f}")
                self._no_meas_warning_count += 1
        
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