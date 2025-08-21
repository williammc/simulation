"""
Camera measurement model for SLAM estimation.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass

from src.common.data_structures import (
    CameraObservation, ImagePoint, CameraCalibration,
    Pose, Landmark
)
from src.utils.math_utils import (
    quaternion_to_rotation_matrix, skew
)


@dataclass
class ReprojectionError:
    """
    Reprojection error for a single observation.
    
    Attributes:
        residual: 2D pixel error [u_error, v_error]
        jacobian_pose: Jacobian w.r.t. camera pose (2x6)
        jacobian_landmark: Jacobian w.r.t. landmark position (2x3)
        squared_error: Squared norm of residual
    """
    residual: np.ndarray
    jacobian_pose: np.ndarray
    jacobian_landmark: np.ndarray
    squared_error: float


class CameraMeasurementModel:
    """
    Camera measurement model for visual SLAM.
    
    Handles projection, unprojection, and Jacobian computation
    for optimization-based estimators.
    """
    
    def __init__(self, calibration: CameraCalibration):
        """
        Initialize camera model.
        
        Args:
            calibration: Camera calibration parameters
        """
        self.calibration = calibration
        self.intrinsics = calibration.intrinsics
        
        # Precompute intrinsic matrix
        self.K = np.array([
            [self.intrinsics.fx, 0, self.intrinsics.cx],
            [0, self.intrinsics.fy, self.intrinsics.cy],
            [0, 0, 1]
        ])
        
        # Camera extrinsics (body to camera)
        self.T_BC = calibration.extrinsics.B_T_C
        self.R_BC = self.T_BC[:3, :3]
        self.t_BC = self.T_BC[:3, 3]
    
    def project(
        self,
        landmark_position: np.ndarray,
        camera_pose: Pose,
        compute_jacobian: bool = False
    ) -> Tuple[Optional[ImagePoint], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Project 3D landmark to image plane.
        
        Args:
            landmark_position: 3D position in world frame
            camera_pose: Camera pose in world frame
            compute_jacobian: Whether to compute Jacobians
        
        Returns:
            Tuple of:
            - Projected pixel (None if behind camera)
            - Jacobian w.r.t. camera pose (2x6) if requested
            - Jacobian w.r.t. landmark (2x3) if requested
        """
        # Transform landmark to camera frame
        R_WB = camera_pose.rotation_matrix
        t_WB = camera_pose.position
        
        # World to body transformation
        p_B = R_WB.T @ (landmark_position - t_WB)
        
        # Body to camera transformation
        p_C = self.R_BC.T @ (p_B - self.t_BC)
        
        # Check if point is behind camera
        if p_C[2] <= 0:
            return None, None, None
        
        # Project to normalized image plane
        x_n = p_C[0] / p_C[2]
        y_n = p_C[1] / p_C[2]
        
        # Apply distortion if present
        if self.intrinsics.distortion is not None and np.any(self.intrinsics.distortion != 0):
            x_n, y_n = self._apply_distortion(x_n, y_n)
        
        # Project to pixel coordinates
        u = self.intrinsics.fx * x_n + self.intrinsics.cx
        v = self.intrinsics.fy * y_n + self.intrinsics.cy
        
        # Check if within image bounds
        if u < 0 or u >= self.intrinsics.width or v < 0 or v >= self.intrinsics.height:
            return None, None, None
        
        pixel = ImagePoint(u=u, v=v)
        
        # Compute Jacobians if requested
        if compute_jacobian:
            J_pose, J_landmark = self._compute_projection_jacobians(
                p_C, R_WB, camera_pose.position, landmark_position
            )
            return pixel, J_pose, J_landmark
        
        return pixel, None, None
    
    def unproject(
        self,
        pixel: ImagePoint,
        depth: float,
        camera_pose: Pose
    ) -> np.ndarray:
        """
        Unproject pixel to 3D point given depth.
        
        Args:
            pixel: Image point
            depth: Depth value (distance along camera z-axis)
            camera_pose: Camera pose in world frame
        
        Returns:
            3D point in world frame
        """
        # Normalize pixel coordinates
        x_n = (pixel.u - self.intrinsics.cx) / self.intrinsics.fx
        y_n = (pixel.v - self.intrinsics.cy) / self.intrinsics.fy
        
        # Remove distortion if present
        if self.intrinsics.distortion is not None and np.any(self.intrinsics.distortion != 0):
            x_n, y_n = self._remove_distortion(x_n, y_n)
        
        # 3D point in camera frame
        p_C = np.array([x_n * depth, y_n * depth, depth])
        
        # Transform to body frame
        p_B = self.R_BC @ p_C + self.t_BC
        
        # Transform to world frame
        R_WB = camera_pose.rotation_matrix
        t_WB = camera_pose.position
        p_W = R_WB @ p_B + t_WB
        
        return p_W
    
    def compute_reprojection_error(
        self,
        observation: CameraObservation,
        landmark: Landmark,
        camera_pose: Pose,
        robust_kernel: Optional[str] = None,
        kernel_threshold: float = 5.0
    ) -> ReprojectionError:
        """
        Compute reprojection error and Jacobians.
        
        Args:
            observation: Camera observation
            landmark: 3D landmark
            camera_pose: Camera pose
            robust_kernel: Robust cost function ('huber', 'cauchy', None)
            kernel_threshold: Threshold for robust kernel
        
        Returns:
            ReprojectionError with residual and Jacobians
        """
        # Project landmark
        projected, J_pose, J_landmark = self.project(
            landmark.position, camera_pose, compute_jacobian=True
        )
        
        if projected is None:
            # Point not visible, return large error
            return ReprojectionError(
                residual=np.array([1000.0, 1000.0]),
                jacobian_pose=np.zeros((2, 6)),
                jacobian_landmark=np.zeros((2, 3)),
                squared_error=2e6
            )
        
        # Compute residual
        residual = np.array([
            observation.pixel.u - projected.u,
            observation.pixel.v - projected.v
        ])
        
        # Apply robust kernel if specified
        weight = 1.0
        if robust_kernel:
            error_norm = np.linalg.norm(residual)
            weight = self._robust_weight(error_norm, robust_kernel, kernel_threshold)
            residual *= weight
            J_pose *= weight
            J_landmark *= weight
        
        squared_error = np.dot(residual, residual)
        
        return ReprojectionError(
            residual=residual,
            jacobian_pose=J_pose,
            jacobian_landmark=J_landmark,
            squared_error=squared_error
        )
    
    def _compute_projection_jacobians(
        self,
        p_C: np.ndarray,
        R_WB: np.ndarray,
        t_WB: np.ndarray,
        p_W: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Jacobians of projection.
        
        Args:
            p_C: Point in camera frame
            R_WB: World to body rotation
            t_WB: World to body translation
            p_W: Point in world frame
        
        Returns:
            J_pose: Jacobian w.r.t. camera pose (2x6)
            J_landmark: Jacobian w.r.t. landmark (2x3)
        """
        # Projection Jacobian
        x, y, z = p_C
        z2 = z * z
        
        J_proj = np.array([
            [self.intrinsics.fx / z, 0, -self.intrinsics.fx * x / z2],
            [0, self.intrinsics.fy / z, -self.intrinsics.fy * y / z2]
        ])
        
        # Transform Jacobians
        R_CB = self.R_BC.T
        R_BW = R_WB.T
        
        # Jacobian w.r.t. landmark position (in world frame)
        J_landmark = J_proj @ R_CB @ R_BW
        
        # Jacobian w.r.t. camera pose
        # Translation part
        J_translation = -J_landmark
        
        # Rotation part (using SO3 perturbation)
        p_B = R_BW @ (p_W - t_WB)
        J_rotation = J_proj @ R_CB @ skew(p_B)
        
        # Combine [rotation, translation]
        J_pose = np.hstack([J_rotation, J_translation])
        
        return J_pose, J_landmark
    
    def _apply_distortion(self, x_n: float, y_n: float) -> Tuple[float, float]:
        """
        Apply radial-tangential distortion model.
        
        Args:
            x_n: Normalized x coordinate
            y_n: Normalized y coordinate
        
        Returns:
            Distorted normalized coordinates
        """
        if len(self.intrinsics.distortion) < 4:
            return x_n, y_n
        
        k1, k2, p1, p2 = self.intrinsics.distortion[:4]
        
        r2 = x_n**2 + y_n**2
        radial = 1 + k1 * r2 + k2 * r2**2
        
        x_d = x_n * radial + 2 * p1 * x_n * y_n + p2 * (r2 + 2 * x_n**2)
        y_d = y_n * radial + p1 * (r2 + 2 * y_n**2) + 2 * p2 * x_n * y_n
        
        return x_d, y_d
    
    def _remove_distortion(self, x_d: float, y_d: float) -> Tuple[float, float]:
        """
        Remove distortion using iterative method.
        
        Args:
            x_d: Distorted normalized x coordinate
            y_d: Distorted normalized y coordinate
        
        Returns:
            Undistorted normalized coordinates
        """
        if len(self.intrinsics.distortion) < 4:
            return x_d, y_d
        
        # Newton iteration to invert distortion
        x_n, y_n = x_d, y_d
        
        for _ in range(5):
            x_dist, y_dist = self._apply_distortion(x_n, y_n)
            dx = x_d - x_dist
            dy = y_d - y_dist
            
            if abs(dx) < 1e-10 and abs(dy) < 1e-10:
                break
            
            x_n += dx
            y_n += dy
        
        return x_n, y_n
    
    def _robust_weight(
        self,
        error: float,
        kernel: str,
        threshold: float
    ) -> float:
        """
        Compute robust kernel weight.
        
        Args:
            error: Error magnitude
            kernel: Kernel type ('huber', 'cauchy')
            threshold: Kernel threshold
        
        Returns:
            Weight for robust cost
        """
        if kernel == 'huber':
            if error <= threshold:
                return 1.0
            else:
                return threshold / error
        
        elif kernel == 'cauchy':
            return threshold**2 / (threshold**2 + error**2)
        
        else:
            return 1.0


def batch_compute_reprojection_errors(
    observations: List[CameraObservation],
    landmarks: List[Landmark],
    camera_pose: Pose,
    calibration: CameraCalibration,
    robust_kernel: Optional[str] = None,
    kernel_threshold: float = 5.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute reprojection errors for batch of observations.
    
    Args:
        observations: List of camera observations
        landmarks: List of corresponding landmarks
        camera_pose: Camera pose
        calibration: Camera calibration
        robust_kernel: Robust cost function
        kernel_threshold: Threshold for robust kernel
    
    Returns:
        residuals: Stacked residuals (2N x 1)
        J_pose: Stacked pose Jacobian (2N x 6)
        J_landmarks: Stacked landmark Jacobians (2N x 3M)
        total_error: Total squared error
    """
    model = CameraMeasurementModel(calibration)
    
    residuals = []
    J_pose_list = []
    J_landmark_list = []
    total_error = 0.0
    
    for obs, lm in zip(observations, landmarks):
        error = model.compute_reprojection_error(
            obs, lm, camera_pose, robust_kernel, kernel_threshold
        )
        
        residuals.append(error.residual)
        J_pose_list.append(error.jacobian_pose)
        J_landmark_list.append(error.jacobian_landmark)
        total_error += error.squared_error
    
    # Stack arrays
    residuals = np.concatenate(residuals) if residuals else np.array([])
    J_pose = np.vstack(J_pose_list) if J_pose_list else np.zeros((0, 6))
    
    # Create sparse landmark Jacobian
    n_obs = len(observations)
    n_landmarks = len(set(obs.landmark_id for obs in observations))
    J_landmarks = np.zeros((2 * n_obs, 3 * n_landmarks))
    
    for i, (obs, J_lm) in enumerate(zip(observations, J_landmark_list)):
        lm_idx = obs.landmark_id
        if lm_idx < n_landmarks:
            J_landmarks[2*i:2*i+2, 3*lm_idx:3*lm_idx+3] = J_lm
    
    return residuals, J_pose, J_landmarks, total_error