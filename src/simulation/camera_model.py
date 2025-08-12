"""
Camera projection model and visibility checking for SLAM simulation.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

from src.common.data_structures import (
    CameraCalibration, CameraIntrinsics, CameraExtrinsics,
    Pose, Landmark, Map, ImagePoint,
    CameraObservation, CameraFrame, CameraData
)
from src.utils.math_utils import quaternion_to_rotation_matrix


@dataclass
class CameraViewConfig:
    """Configuration for camera visibility checking."""
    min_depth: float = 0.1  # Minimum depth for visibility (meters)
    max_depth: float = 50.0  # Maximum depth for visibility (meters)
    check_fov: bool = True  # Whether to check field of view
    margin_pixels: float = 0.0  # Margin from image edges (pixels)


@dataclass
class CameraNoiseConfig:
    """Configuration for camera measurement noise."""
    pixel_noise_std: float = 1.0  # Standard deviation of pixel noise (pixels)
    add_noise: bool = False  # Whether to add noise to measurements
    outlier_probability: float = 0.0  # Probability of generating an outlier
    outlier_std: float = 10.0  # Standard deviation for outlier measurements
    seed: Optional[int] = None  # Random seed for reproducibility


class PinholeCamera:
    """Pinhole camera model with projection and visibility checking."""
    
    def __init__(
        self,
        calibration: CameraCalibration,
        view_config: Optional[CameraViewConfig] = None,
        noise_config: Optional[CameraNoiseConfig] = None
    ):
        """
        Initialize pinhole camera model.
        
        Args:
            calibration: Camera calibration (intrinsics and extrinsics)
            view_config: Visibility checking configuration
            noise_config: Noise configuration
        """
        self.calibration = calibration
        self.intrinsics = calibration.intrinsics
        self.extrinsics = calibration.extrinsics
        self.view_config = view_config or CameraViewConfig()
        self.noise_config = noise_config or CameraNoiseConfig()
        
        # Set random seed if provided
        if self.noise_config.seed is not None:
            np.random.seed(self.noise_config.seed)
        
        # Precompute camera matrix K
        self.K = np.array([
            [self.intrinsics.fx, 0, self.intrinsics.cx],
            [0, self.intrinsics.fy, self.intrinsics.cy],
            [0, 0, 1]
        ])
        
        # Image bounds for visibility checking
        self.u_min = self.view_config.margin_pixels
        self.u_max = self.intrinsics.width - self.view_config.margin_pixels
        self.v_min = self.view_config.margin_pixels
        self.v_max = self.intrinsics.height - self.view_config.margin_pixels
    
    def project_point(
        self,
        point_world: np.ndarray,
        W_T_B: np.ndarray
    ) -> Tuple[Optional[ImagePoint], float]:
        """
        Project 3D world point to image plane.
        
        Args:
            point_world: 3D point in world frame [x, y, z]
            W_T_B: 4x4 transformation from body to world frame
        
        Returns:
            (image_point, depth) or (None, depth) if not visible
            depth is the z-coordinate in camera frame
        """
        # Transform from world to body frame
        # B_T_W = inv(W_T_B)
        B_T_W = np.linalg.inv(W_T_B)
        point_world_hom = np.append(point_world, 1.0)
        point_body_hom = B_T_W @ point_world_hom
        point_body = point_body_hom[:3]
        
        # Transform from body to camera frame using extrinsics
        # C_T_B = inv(B_T_C)
        C_T_B = np.linalg.inv(self.extrinsics.B_T_C)
        point_body_hom = np.append(point_body, 1.0)
        point_camera_hom = C_T_B @ point_body_hom
        point_camera = point_camera_hom[:3]
        
        # Depth in camera frame
        depth = point_camera[2]
        
        # Check depth bounds
        if depth < self.view_config.min_depth or depth > self.view_config.max_depth:
            return None, depth
        
        # Project to image plane (simple pinhole, no distortion)
        if depth > 0:
            u = self.intrinsics.fx * point_camera[0] / depth + self.intrinsics.cx
            v = self.intrinsics.fy * point_camera[1] / depth + self.intrinsics.cy
            
            # Check image bounds
            if self.view_config.check_fov:
                if u < self.u_min or u > self.u_max or v < self.v_min or v > self.v_max:
                    return None, depth
            
            return ImagePoint(u=u, v=v), depth
        
        return None, depth
    
    def is_visible(
        self,
        point_world: np.ndarray,
        W_T_B: np.ndarray
    ) -> bool:
        """
        Check if a 3D point is visible from current camera pose.
        
        Args:
            point_world: 3D point in world frame
            W_T_B: Current body pose in world frame
        
        Returns:
            True if point is visible
        """
        pixel, depth = self.project_point(point_world, W_T_B)
        return pixel is not None
    
    def get_visible_landmarks(
        self,
        landmarks: Map,
        W_T_B: np.ndarray
    ) -> List[Tuple[Landmark, ImagePoint, float]]:
        """
        Get all visible landmarks from current pose.
        
        Args:
            landmarks: Map containing landmarks
            W_T_B: Current body pose in world frame
        
        Returns:
            List of (landmark, pixel_coords, depth) for visible landmarks
        """
        visible = []
        
        for landmark in landmarks.landmarks.values():
            pixel, depth = self.project_point(landmark.position, W_T_B)
            if pixel is not None:
                visible.append((landmark, pixel, depth))
        
        return visible
    
    def compute_field_of_view(self) -> Tuple[float, float]:
        """
        Compute horizontal and vertical field of view angles.
        
        Returns:
            (horizontal_fov, vertical_fov) in radians
        """
        h_fov = 2 * np.arctan(self.intrinsics.width / (2 * self.intrinsics.fx))
        v_fov = 2 * np.arctan(self.intrinsics.height / (2 * self.intrinsics.fy))
        return h_fov, v_fov
    
    def frustum_culling(
        self,
        points: np.ndarray,
        W_T_B: np.ndarray,
        margin_factor: float = 1.2
    ) -> np.ndarray:
        """
        Fast frustum culling to filter points outside camera view.
        
        Args:
            points: Nx3 array of world points
            W_T_B: Current body pose in world frame
            margin_factor: Factor to expand frustum for conservative culling
        
        Returns:
            Boolean mask of potentially visible points
        """
        if len(points) == 0:
            return np.array([], dtype=bool)
        
        # Transform all points to camera frame
        B_T_W = np.linalg.inv(W_T_B)
        C_T_B = np.linalg.inv(self.extrinsics.B_T_C)
        C_T_W = C_T_B @ B_T_W
        
        # Add homogeneous coordinate
        points_hom = np.hstack([points, np.ones((len(points), 1))])
        points_camera = (C_T_W @ points_hom.T).T[:, :3]
        
        # Depth check
        depth_valid = (points_camera[:, 2] >= self.view_config.min_depth) & \
                     (points_camera[:, 2] <= self.view_config.max_depth)
        
        # FOV check (conservative with margin)
        h_fov, v_fov = self.compute_field_of_view()
        h_fov *= margin_factor
        v_fov *= margin_factor
        
        # Check if points are within expanded frustum
        x_angle = np.arctan2(np.abs(points_camera[:, 0]), points_camera[:, 2])
        y_angle = np.arctan2(np.abs(points_camera[:, 1]), points_camera[:, 2])
        
        fov_valid = (x_angle < h_fov/2) & (y_angle < v_fov/2)
        
        return depth_valid & fov_valid


    def add_noise_to_pixel(self, pixel: ImagePoint) -> ImagePoint:
        """
        Add noise to pixel measurement.
        
        Args:
            pixel: Original pixel measurement
        
        Returns:
            Noisy pixel measurement
        """
        if not self.noise_config.add_noise:
            return pixel
        
        # Check if this should be an outlier
        if np.random.random() < self.noise_config.outlier_probability:
            # Generate outlier measurement
            noise_u = np.random.normal(0, self.noise_config.outlier_std)
            noise_v = np.random.normal(0, self.noise_config.outlier_std)
        else:
            # Normal noise
            noise_u = np.random.normal(0, self.noise_config.pixel_noise_std)
            noise_v = np.random.normal(0, self.noise_config.pixel_noise_std)
        
        # Add noise and ensure pixel stays within image bounds
        noisy_u = pixel.u + noise_u
        noisy_v = pixel.v + noise_v
        
        # Optionally clamp to image bounds
        noisy_u = np.clip(noisy_u, 0, self.intrinsics.width - 1)
        noisy_v = np.clip(noisy_v, 0, self.intrinsics.height - 1)
        
        return ImagePoint(u=noisy_u, v=noisy_v)


class StereoCamera:
    """Stereo camera model with left and right cameras."""
    
    def __init__(
        self,
        left_calibration: CameraCalibration,
        right_calibration: CameraCalibration,
        view_config: Optional[CameraViewConfig] = None
    ):
        """
        Initialize stereo camera model.
        
        Args:
            left_calibration: Left camera calibration
            right_calibration: Right camera calibration
            view_config: Visibility checking configuration
        """
        self.left_camera = PinholeCamera(left_calibration, view_config)
        self.right_camera = PinholeCamera(right_calibration, view_config)
        
        # Compute baseline (distance between cameras)
        left_pos = left_calibration.extrinsics.B_T_C[:3, 3]
        right_pos = right_calibration.extrinsics.B_T_C[:3, 3]
        self.baseline = np.linalg.norm(left_pos - right_pos)
    
    def get_stereo_observations(
        self,
        landmarks: Map,
        W_T_B: np.ndarray
    ) -> Tuple[List[Tuple[Landmark, ImagePoint, float]], 
               List[Tuple[Landmark, ImagePoint, float]]]:
        """
        Get stereo observations of landmarks.
        
        Args:
            landmarks: Map containing landmarks
            W_T_B: Current body pose in world frame
        
        Returns:
            (left_observations, right_observations)
        """
        left_obs = self.left_camera.get_visible_landmarks(landmarks, W_T_B)
        right_obs = self.right_camera.get_visible_landmarks(landmarks, W_T_B)
        
        # Filter to only include landmarks visible in both cameras
        left_ids = {obs[0].id for obs in left_obs}
        right_ids = {obs[0].id for obs in right_obs}
        common_ids = left_ids & right_ids
        
        left_filtered = [obs for obs in left_obs if obs[0].id in common_ids]
        right_filtered = [obs for obs in right_obs if obs[0].id in common_ids]
        
        return left_filtered, right_filtered
    
    def triangulate(
        self,
        left_pixel: ImagePoint,
        right_pixel: ImagePoint,
        W_T_B: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Triangulate 3D point from stereo correspondence.
        
        Args:
            left_pixel: Pixel in left image
            right_pixel: Pixel in right image
            W_T_B: Current body pose in world frame
        
        Returns:
            3D point in world frame, or None if triangulation fails
        """
        # Compute disparity
        disparity = left_pixel.u - right_pixel.u
        
        if abs(disparity) < 1e-3:  # Too small disparity
            return None
        
        # Simple triangulation using disparity
        # depth = baseline * fx / disparity
        depth = self.baseline * self.left_camera.intrinsics.fx / abs(disparity)
        
        if depth < 0 or depth > self.left_camera.view_config.max_depth:
            return None
        
        # Back-project from left camera
        x_cam = (left_pixel.u - self.left_camera.intrinsics.cx) * depth / self.left_camera.intrinsics.fx
        y_cam = (left_pixel.v - self.left_camera.intrinsics.cy) * depth / self.left_camera.intrinsics.fy
        z_cam = depth
        
        point_camera = np.array([x_cam, y_cam, z_cam])
        
        # Transform to world frame
        point_camera_hom = np.append(point_camera, 1.0)
        point_body_hom = self.left_camera.extrinsics.B_T_C @ point_camera_hom
        point_world_hom = W_T_B @ point_body_hom
        
        return point_world_hom[:3]


def generate_camera_observations(
    camera: PinholeCamera,
    landmarks: Map,
    pose: Pose,
    timestamp: float,
    camera_id: str = "cam0",
    add_noise: Optional[bool] = None
) -> CameraFrame:
    """
    Generate camera observations for a given pose.
    
    Args:
        camera: Camera model
        landmarks: Map of landmarks
        pose: Current pose
        timestamp: Observation timestamp
        camera_id: Camera identifier
        add_noise: Override camera's noise setting (None uses camera config)
    
    Returns:
        CameraFrame with observations
    """
    # Get transformation matrix from pose
    W_T_B = pose.to_matrix()
    
    # Get visible landmarks
    visible = camera.get_visible_landmarks(landmarks, W_T_B)
    
    # Determine if we should add noise
    should_add_noise = add_noise if add_noise is not None else camera.noise_config.add_noise
    
    # Create observations
    observations = []
    for landmark, pixel, depth in visible:
        # Add noise if configured
        if should_add_noise:
            pixel = camera.add_noise_to_pixel(pixel)
        
        obs = CameraObservation(
            landmark_id=landmark.id,
            pixel=pixel,
            descriptor=landmark.descriptor  # Pass through descriptor if available
        )
        observations.append(obs)
    
    # Sort by landmark ID for consistency
    observations.sort(key=lambda x: x.landmark_id)
    
    return CameraFrame(
        timestamp=timestamp,
        camera_id=camera_id,
        observations=observations
    )