"""
Stereo camera model for SLAM estimation.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

from src.common.data_structures import (
    CameraObservation, ImagePoint, CameraCalibration,
    Pose, Landmark
)
from src.estimation.camera_model import (
    CameraMeasurementModel, ReprojectionError
)
from src.utils.math_utils import (
    quaternion_to_rotation_matrix, skew
)


@dataclass
class StereoObservation:
    """
    Stereo camera observation.
    
    Attributes:
        left_pixel: Left camera pixel observation
        right_pixel: Right camera pixel observation
        landmark_id: Associated landmark ID
        timestamp: Observation timestamp
        disparity: Horizontal disparity (left.u - right.u)
        depth: Computed depth from disparity
    """
    left_pixel: ImagePoint
    right_pixel: ImagePoint
    landmark_id: int
    timestamp: float
    disparity: Optional[float] = None
    depth: Optional[float] = None
    
    def __post_init__(self):
        """Compute disparity if not provided."""
        if self.disparity is None:
            self.disparity = self.left_pixel.u - self.right_pixel.u


@dataclass
class StereoCalibration:
    """
    Stereo camera calibration.
    
    Attributes:
        left_calib: Left camera calibration
        right_calib: Right camera calibration
        baseline: Stereo baseline (distance between cameras) in meters
        T_RL: Transform from left to right camera frame
    """
    left_calib: CameraCalibration
    right_calib: CameraCalibration
    baseline: float
    T_RL: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Initialize stereo transform if not provided."""
        if self.T_RL is None:
            # Default: right camera is baseline distance along x-axis from left camera
            # T_RL transforms points from left to right camera frame
            self.T_RL = np.eye(4)
            self.T_RL[0, 3] = -self.baseline  # Right camera sees points shifted left


class StereoCameraModel:
    """
    Stereo camera measurement model for visual SLAM.
    
    Handles stereo projection, triangulation, and Jacobian computation.
    """
    
    def __init__(self, calibration: StereoCalibration):
        """
        Initialize stereo camera model.
        
        Args:
            calibration: Stereo camera calibration
        """
        self.calibration = calibration
        self.left_model = CameraMeasurementModel(calibration.left_calib)
        self.right_model = CameraMeasurementModel(calibration.right_calib)
        
        # Extract stereo transform
        self.R_RL = self.calibration.T_RL[:3, :3]
        self.t_RL = self.calibration.T_RL[:3, 3]
        
        # Precompute rectification if cameras are aligned
        self.is_rectified = self._check_rectification()
    
    def _check_rectification(self) -> bool:
        """
        Check if stereo pair is rectified (aligned horizontally).
        
        Returns:
            True if cameras are rectified
        """
        # Check if rotation is identity (aligned cameras)
        if not np.allclose(self.R_RL, np.eye(3), atol=1e-6):
            return False
        
        # Check if baseline is purely horizontal
        if abs(self.t_RL[1]) > 1e-6 or abs(self.t_RL[2]) > 1e-6:
            return False
        
        # Check if intrinsics match (common for rectified pairs)
        left_K = self.left_model.K
        right_K = self.right_model.K
        
        return np.allclose(left_K, right_K, atol=1e-3)
    
    def project_stereo(
        self,
        landmark_position: np.ndarray,
        camera_pose: Pose,
        compute_jacobian: bool = False
    ) -> Tuple[Optional[StereoObservation], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Project 3D landmark to both stereo cameras.
        
        Args:
            landmark_position: 3D position in world frame
            camera_pose: Left camera pose in world frame
            compute_jacobian: Whether to compute Jacobians
        
        Returns:
            Tuple of:
            - Stereo observation (None if not visible in both cameras)
            - Jacobian w.r.t. camera pose (4x6) if requested
            - Jacobian w.r.t. landmark (4x3) if requested
        """
        # Project to left camera
        left_pixel, J_left_pose, J_left_lm = self.left_model.project(
            landmark_position, camera_pose, compute_jacobian
        )
        
        if left_pixel is None:
            return None, None, None
        
        # Compute right camera pose
        right_pose = self._compute_right_pose(camera_pose)
        
        # Project to right camera
        right_pixel, J_right_pose, J_right_lm = self.right_model.project(
            landmark_position, right_pose, compute_jacobian
        )
        
        if right_pixel is None:
            return None, None, None
        
        # Create stereo observation
        stereo_obs = StereoObservation(
            left_pixel=left_pixel,
            right_pixel=right_pixel,
            landmark_id=-1,  # To be set by caller
            timestamp=camera_pose.timestamp
        )
        
        # Compute depth from disparity
        if self.is_rectified and stereo_obs.disparity > 0:
            fx = self.left_model.intrinsics.fx
            stereo_obs.depth = (self.calibration.baseline * fx) / stereo_obs.disparity
        
        # Combine Jacobians if requested
        if compute_jacobian:
            # Stack Jacobians for left and right observations
            J_pose = np.vstack([J_left_pose, J_right_pose]) if J_left_pose is not None else None
            J_landmark = np.vstack([J_left_lm, J_right_lm]) if J_left_lm is not None else None
            return stereo_obs, J_pose, J_landmark
        
        return stereo_obs, None, None
    
    def triangulate(
        self,
        stereo_obs: StereoObservation,
        camera_pose: Pose
    ) -> Tuple[np.ndarray, float]:
        """
        Triangulate 3D point from stereo observation.
        
        Args:
            stereo_obs: Stereo observation
            camera_pose: Left camera pose in world frame
        
        Returns:
            Tuple of:
            - 3D point in world frame
            - Uncertainty (reprojection error)
        """
        if self.is_rectified and stereo_obs.disparity is not None and stereo_obs.disparity > 0:
            # Fast triangulation for rectified cameras
            return self._triangulate_rectified(stereo_obs, camera_pose)
        else:
            # General triangulation using DLT
            return self._triangulate_general(stereo_obs, camera_pose)
    
    def _triangulate_rectified(
        self,
        stereo_obs: StereoObservation,
        camera_pose: Pose
    ) -> Tuple[np.ndarray, float]:
        """
        Fast triangulation for rectified stereo pair.
        
        Args:
            stereo_obs: Stereo observation
            camera_pose: Left camera pose
        
        Returns:
            3D point and uncertainty
        """
        # Compute depth from disparity
        fx = self.left_model.intrinsics.fx
        baseline = self.calibration.baseline
        depth = (baseline * fx) / stereo_obs.disparity
        
        # Unproject from left camera
        landmark = self.left_model.unproject(
            stereo_obs.left_pixel,
            depth,
            camera_pose
        )
        
        # Estimate uncertainty from disparity error
        disparity_std = 0.5  # pixels (typical stereo matching error)
        depth_std = (baseline * fx * disparity_std) / (stereo_obs.disparity ** 2)
        
        return landmark, depth_std
    
    def _triangulate_general(
        self,
        stereo_obs: StereoObservation,
        camera_pose: Pose
    ) -> Tuple[np.ndarray, float]:
        """
        General triangulation using DLT (Direct Linear Transform).
        
        Args:
            stereo_obs: Stereo observation
            camera_pose: Left camera pose
        
        Returns:
            3D point and uncertainty
        """
        # Get camera matrices
        P_left = self._get_projection_matrix(camera_pose, self.left_model)
        
        right_pose = self._compute_right_pose(camera_pose)
        P_right = self._get_projection_matrix(right_pose, self.right_model)
        
        # Build DLT system
        A = np.zeros((4, 4))
        
        # Left camera constraints
        u_l, v_l = stereo_obs.left_pixel.u, stereo_obs.left_pixel.v
        A[0] = u_l * P_left[2] - P_left[0]
        A[1] = v_l * P_left[2] - P_left[1]
        
        # Right camera constraints
        u_r, v_r = stereo_obs.right_pixel.u, stereo_obs.right_pixel.v
        A[2] = u_r * P_right[2] - P_right[0]
        A[3] = v_r * P_right[2] - P_right[1]
        
        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X_homo = Vt[-1]
        
        # Normalize homogeneous coordinates
        landmark = X_homo[:3] / X_homo[3]
        
        # Estimate uncertainty from reprojection error
        left_reproj, _, _ = self.left_model.project(landmark, camera_pose, False)
        right_reproj, _, _ = self.right_model.project(landmark, right_pose, False)
        
        error = 0.0
        if left_reproj is not None:
            error += np.linalg.norm([
                left_reproj.u - stereo_obs.left_pixel.u,
                left_reproj.v - stereo_obs.left_pixel.v
            ])
        if right_reproj is not None:
            error += np.linalg.norm([
                right_reproj.u - stereo_obs.right_pixel.u,
                right_reproj.v - stereo_obs.right_pixel.v
            ])
        
        return landmark, error
    
    def compute_stereo_reprojection_error(
        self,
        stereo_obs: StereoObservation,
        landmark: Landmark,
        camera_pose: Pose,
        robust_kernel: Optional[str] = None,
        kernel_threshold: float = 5.0
    ) -> Tuple[ReprojectionError, ReprojectionError]:
        """
        Compute reprojection errors for stereo observation.
        
        Args:
            stereo_obs: Stereo observation
            landmark: 3D landmark
            camera_pose: Left camera pose
            robust_kernel: Robust cost function
            kernel_threshold: Threshold for robust kernel
        
        Returns:
            Tuple of (left_error, right_error)
        """
        # Create camera observations
        left_obs = CameraObservation(
            pixel=stereo_obs.left_pixel,
            landmark_id=stereo_obs.landmark_id
        )
        
        right_obs = CameraObservation(
            pixel=stereo_obs.right_pixel,
            landmark_id=stereo_obs.landmark_id
        )
        
        # Compute left error
        left_error = self.left_model.compute_reprojection_error(
            left_obs, landmark, camera_pose, robust_kernel, kernel_threshold
        )
        
        # Compute right error with right camera pose
        right_pose = self._compute_right_pose(camera_pose)
        right_error = self.right_model.compute_reprojection_error(
            right_obs, landmark, right_pose, robust_kernel, kernel_threshold
        )
        
        return left_error, right_error
    
    def _compute_right_pose(self, left_pose: Pose) -> Pose:
        """
        Compute right camera pose from left camera pose.
        
        Args:
            left_pose: Left camera pose in world frame
        
        Returns:
            Right camera pose in world frame
        """
        # Transform from left to right camera
        R_WL = quaternion_to_rotation_matrix(left_pose.quaternion)
        t_WL = left_pose.position
        
        # Right camera in world frame
        R_WR = R_WL @ self.R_RL
        t_WR = t_WL + R_WL @ self.t_RL
        
        # Convert back to quaternion
        from src.utils.math_utils import rotation_matrix_to_quaternion
        q_WR = rotation_matrix_to_quaternion(R_WR)
        
        return Pose(
            timestamp=left_pose.timestamp,
            position=t_WR,
            quaternion=q_WR
        )
    
    def _get_projection_matrix(
        self,
        pose: Pose,
        camera_model: CameraMeasurementModel
    ) -> np.ndarray:
        """
        Get 3x4 projection matrix P = K[R|t].
        
        Args:
            pose: Camera pose in world frame
            camera_model: Camera model with intrinsics
        
        Returns:
            3x4 projection matrix
        """
        # World to camera transform
        R_WC = quaternion_to_rotation_matrix(pose.quaternion)
        t_WC = pose.position
        
        # Camera to world transform (inverse)
        R_CW = R_WC.T
        t_CW = -R_CW @ t_WC
        
        # Include body-to-camera transform
        R_CB = camera_model.R_BC.T
        t_CB = -R_CB @ camera_model.t_BC
        
        # Combined transform
        R = R_CB @ R_CW
        t = R_CB @ t_CW + t_CB
        
        # Projection matrix
        Rt = np.hstack([R, t.reshape(-1, 1)])
        P = camera_model.K @ Rt
        
        return P


def create_stereo_calibration(
    baseline: float = 0.12,
    fx: float = 500.0,
    fy: float = 500.0,
    cx: float = 320.0,
    cy: float = 240.0,
    width: int = 640,
    height: int = 480,
    distortion: Optional[np.ndarray] = None
) -> StereoCalibration:
    """
    Create a stereo calibration with identical cameras.
    
    Args:
        baseline: Stereo baseline in meters
        fx, fy: Focal lengths
        cx, cy: Principal points
        width, height: Image dimensions
        distortion: Optional distortion coefficients
    
    Returns:
        Stereo calibration
    """
    from src.common.data_structures import CameraIntrinsics, CameraExtrinsics
    
    # Create identical intrinsics
    from src.common.data_structures import CameraModel
    intrinsics = CameraIntrinsics(
        model=CameraModel.PINHOLE,
        fx=fx, fy=fy, cx=cx, cy=cy,
        width=width, height=height,
        distortion=distortion if distortion is not None else np.zeros(5)
    )
    
    # Default extrinsics (camera at body frame)
    extrinsics = CameraExtrinsics(B_T_C=np.eye(4))
    
    # Create calibrations
    left_calib = CameraCalibration(
        camera_id="left",
        intrinsics=intrinsics,
        extrinsics=extrinsics
    )
    
    right_calib = CameraCalibration(
        camera_id="right",
        intrinsics=intrinsics,
        extrinsics=extrinsics
    )
    
    return StereoCalibration(
        left_calib=left_calib,
        right_calib=right_calib,
        baseline=baseline
    )