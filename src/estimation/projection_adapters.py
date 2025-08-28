"""
Projection service adapters for different camera models.

These adapters implement the ProjectionInterface to provide camera-model-independent
projection services for the preprocessing pipeline.
"""

import numpy as np
from typing import Optional, Tuple
import logging

from src.estimation.interfaces import ProjectionInterface, VisualMeasurement
from src.common.data_structures import Pose, CameraCalibration
from src.estimation.camera_model import CameraMeasurementModel

logger = logging.getLogger(__name__)


class PinholeProjectionAdapter(ProjectionInterface):
    """
    Projection adapter for pinhole camera models.
    
    Wraps existing CameraMeasurementModel to provide ProjectionInterface.
    """
    
    def __init__(self, camera_calibration: CameraCalibration):
        """
        Initialize adapter with camera calibration.
        
        Args:
            camera_calibration: Camera calibration parameters
        """
        self.camera_calib = camera_calibration
        
        # Create underlying camera model
        self.camera_model = CameraMeasurementModel(camera_calibration)
        
        # Default pixel noise
        self.pixel_noise_std = 1.0
        self.pixel_covariance = np.eye(2) * (self.pixel_noise_std ** 2)
    
    def project(
        self, 
        landmark_position: np.ndarray,
        camera_pose: Pose,
        compute_jacobians: bool = False
    ) -> VisualMeasurement:
        """
        Project 3D landmark to image plane.
        
        Args:
            landmark_position: 3D position of landmark in world frame
            camera_pose: Current camera pose estimate
            compute_jacobians: Whether to compute Jacobians
            
        Returns:
            VisualMeasurement with predicted pixel (no observation)
        """
        # Use camera model's project method
        image_point, jacobian_wrt_pose, jacobian_wrt_landmark = self.camera_model.project(
            landmark_position,
            camera_pose,
            compute_jacobian=compute_jacobians
        )
        
        # Handle case where point is behind camera
        if image_point is None:
            # Use intrinsics width/height or default values
            if hasattr(self.camera_calib, 'intrinsics'):
                width = self.camera_calib.intrinsics.width
                height = self.camera_calib.intrinsics.height
            else:
                width = 640
                height = 480
            pixel = np.array([width/2, height/2])
        else:
            pixel = np.array([image_point.u, image_point.v])
        
        # Create measurement (predicted only, no observation)
        measurement = VisualMeasurement(
            landmark_id=-1,  # Will be set by caller
            observed_pixel=np.zeros(2),  # Will be set by caller
            predicted_pixel=pixel,
            residual=np.zeros(2),  # Will be computed by caller
            pixel_covariance=self.pixel_covariance,
            jacobian_wrt_pose=jacobian_wrt_pose,
            jacobian_wrt_landmark=jacobian_wrt_landmark
        )
        
        return measurement
    
    def unproject(self, pixel: np.ndarray, depth: float) -> np.ndarray:
        """
        Unproject pixel to 3D point given depth.
        
        Args:
            pixel: 2D pixel coordinates
            depth: Distance along optical axis
            
        Returns:
            3D point in camera frame
        """
        # Simple pinhole unprojection since CameraMeasurementModel doesn't have this method
        u, v = pixel
        if hasattr(self.camera_calib, 'intrinsics'):
            fx = self.camera_calib.intrinsics.fx
            fy = self.camera_calib.intrinsics.fy
            cx = self.camera_calib.intrinsics.cx
            cy = self.camera_calib.intrinsics.cy
        elif hasattr(self.camera_calib, 'K'):
            fx = self.camera_calib.K[0,0]
            fy = self.camera_calib.K[1,1]
            cx = self.camera_calib.K[0,2]
            cy = self.camera_calib.K[1,2]
        else:
            # Default values
            fx = fy = 500.0
            cx = 320.0
            cy = 240.0
        
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        return np.array([x, y, depth])
    
    def compute_jacobians(
        self,
        landmark_position: np.ndarray,
        camera_pose: Pose
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute projection Jacobians.
        
        Args:
            landmark_position: 3D position of landmark
            camera_pose: Current camera pose
            
        Returns:
            Tuple of (jacobian_wrt_pose, jacobian_wrt_landmark)
        """
        # Use camera model's project method to get Jacobians
        _, J_pose, J_landmark = self.camera_model.project(
            landmark_position,
            camera_pose,
            compute_jacobian=True
        )
        
        if J_pose is None:
            J_pose = np.zeros((2, 6))
        if J_landmark is None:
            J_landmark = np.zeros((2, 3))
            
        return J_pose, J_landmark
    


class MockProjectionService(ProjectionInterface):
    """
    Mock projection service for testing without camera model.
    
    Provides synthetic projections for testing the preprocessing pipeline.
    """
    
    def __init__(
        self,
        image_width: int = 640,
        image_height: int = 480,
        focal_length: float = 500.0,
        pixel_noise_std: float = 1.0
    ):
        """
        Initialize mock projection service.
        
        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels  
            focal_length: Focal length in pixels
            pixel_noise_std: Standard deviation of pixel noise
        """
        self.image_width = image_width
        self.image_height = image_height
        self.focal_length = focal_length
        self.cx = image_width / 2
        self.cy = image_height / 2
        self.pixel_noise_std = pixel_noise_std
        self.pixel_covariance = np.eye(2) * (pixel_noise_std ** 2)
    
    def project(
        self,
        landmark_position: np.ndarray,
        camera_pose: Pose,
        compute_jacobians: bool = False
    ) -> VisualMeasurement:
        """
        Mock projection that returns synthetic measurements.
        
        Args:
            landmark_position: 3D position of landmark
            camera_pose: Current camera pose
            compute_jacobians: Whether to compute Jacobians
            
        Returns:
            VisualMeasurement with synthetic data
        """
        # Simple pinhole projection
        R = camera_pose.rotation_matrix
        t = camera_pose.position
        p_cam = R.T @ (landmark_position - t)
        
        if p_cam[2] <= 0:
            # Behind camera
            predicted_pixel = np.array([self.cx, self.cy])
        else:
            # Project
            u = self.focal_length * p_cam[0] / p_cam[2] + self.cx
            v = self.focal_length * p_cam[1] / p_cam[2] + self.cy
            predicted_pixel = np.array([u, v])
        
        # Add small noise for realism
        predicted_pixel += np.random.randn(2) * 0.1
        
        # Generate synthetic Jacobians if requested
        jacobian_wrt_pose = None
        jacobian_wrt_landmark = None
        
        if compute_jacobians:
            # Random but reasonable Jacobians
            jacobian_wrt_pose = np.random.randn(2, 6) * 0.1
            jacobian_wrt_landmark = np.random.randn(2, 3) * 0.1
        
        measurement = VisualMeasurement(
            landmark_id=-1,
            observed_pixel=np.zeros(2),
            predicted_pixel=predicted_pixel,
            residual=np.zeros(2),
            pixel_covariance=self.pixel_covariance,
            jacobian_wrt_pose=jacobian_wrt_pose,
            jacobian_wrt_landmark=jacobian_wrt_landmark
        )
        
        return measurement
    
    def unproject(self, pixel: np.ndarray, depth: float) -> np.ndarray:
        """
        Mock unprojection.
        
        Args:
            pixel: 2D pixel coordinates
            depth: Distance along optical axis
            
        Returns:
            3D point in camera frame
        """
        u, v = pixel
        x = (u - self.cx) * depth / self.focal_length
        y = (v - self.cy) * depth / self.focal_length
        return np.array([x, y, depth])
    
    def compute_jacobians(
        self,
        landmark_position: np.ndarray,
        camera_pose: Pose
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mock Jacobians.
        
        Args:
            landmark_position: 3D position of landmark
            camera_pose: Current camera pose
            
        Returns:
            Tuple of synthetic Jacobians
        """
        # Return reasonable synthetic Jacobians
        jacobian_wrt_pose = np.random.randn(2, 6) * 0.1
        jacobian_wrt_landmark = np.random.randn(2, 3) * 0.1
        return jacobian_wrt_pose, jacobian_wrt_landmark