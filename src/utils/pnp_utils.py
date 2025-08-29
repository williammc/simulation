"""
PnP (Perspective-n-Point) utilities for pose estimation.
"""

import numpy as np
from typing import Tuple, Optional
import cv2


def solve_pnp_ideal(points_3d: np.ndarray, 
                   ideal_coords: np.ndarray,
                   reprojection_threshold: float = 3.0) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Solve PnP problem using ideal coordinates.
    
    Ideal coordinates are normalized image coordinates on the z=1 plane.
    This is equivalent to using a camera with identity intrinsics.
    
    Args:
        points_3d: (N, 3) array of 3D landmark positions in world frame
        ideal_coords: (N, 2) array of ideal coordinates
        reprojection_threshold: Maximum reprojection error in pixels
        
    Returns:
        success: Whether PnP succeeded
        rotation: 3x3 rotation matrix (camera to world)
        translation: 3x1 translation vector (camera position in world)
    """
    if len(points_3d) < 4:
        return False, None, None
    
    # Convert ideal coordinates to homogeneous rays
    num_points = len(ideal_coords)
    rays = np.ones((num_points, 3))
    rays[:, :2] = ideal_coords
    
    # Use identity camera matrix for ideal coordinates
    K = np.eye(3)
    
    # Distortion coefficients (none for ideal coordinates)
    dist_coeffs = np.zeros(5)
    
    try:
        # Solve PnP
        # Note: OpenCV returns camera extrinsics (world to camera transform)
        success, rvec, tvec = cv2.solvePnP(
            points_3d.astype(np.float32),
            ideal_coords.astype(np.float32),
            K,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP  # Use EPnP algorithm
        )
        
        if not success:
            return False, None, None
        
        # Convert rotation vector to matrix
        R_cv, _ = cv2.Rodrigues(rvec)
        t_cv = tvec.flatten()
        
        # OpenCV gives world-to-camera transform
        # We want camera-to-world transform for the pose
        R_camera_to_world = R_cv.T
        t_camera_in_world = -R_cv.T @ t_cv
        
        # Verify solution with reprojection
        projected, _ = cv2.projectPoints(
            points_3d.astype(np.float32),
            rvec,
            tvec,
            K,
            dist_coeffs
        )
        projected = projected.reshape(-1, 2)
        
        # Check reprojection error
        errors = np.linalg.norm(projected - ideal_coords, axis=1)
        mean_error = np.mean(errors)
        
        if mean_error > reprojection_threshold:
            return False, None, None
        
        return True, R_camera_to_world, t_camera_in_world
        
    except Exception as e:
        print(f"PnP failed: {e}")
        return False, None, None


def solve_pnp_ransac_ideal(points_3d: np.ndarray,
                           ideal_coords: np.ndarray,
                           reprojection_threshold: float = 3.0,
                           confidence: float = 0.99) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Solve PnP with RANSAC for robust estimation.
    
    Args:
        points_3d: (N, 3) array of 3D landmark positions
        ideal_coords: (N, 2) array of ideal coordinates
        reprojection_threshold: RANSAC threshold
        confidence: RANSAC confidence level
        
    Returns:
        success: Whether PnP succeeded
        rotation: 3x3 rotation matrix
        translation: 3x1 translation vector
        inliers: Boolean mask of inlier points
    """
    if len(points_3d) < 4:
        return False, None, None, None
    
    K = np.eye(3)
    dist_coeffs = np.zeros(5)
    
    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d.astype(np.float32),
            ideal_coords.astype(np.float32),
            K,
            dist_coeffs,
            reprojectionError=reprojection_threshold,
            confidence=confidence,
            flags=cv2.SOLVEPNP_EPNP
        )
        
        if not success or inliers is None:
            return False, None, None, None
        
        # Convert to camera-to-world transform
        R_cv, _ = cv2.Rodrigues(rvec)
        R_camera_to_world = R_cv.T
        t_camera_in_world = -R_cv.T @ tvec.flatten()
        
        # Create inlier mask
        inlier_mask = np.zeros(len(points_3d), dtype=bool)
        inlier_mask[inliers.flatten()] = True
        
        return True, R_camera_to_world, t_camera_in_world, inlier_mask
        
    except Exception as e:
        print(f"PnP RANSAC failed: {e}")
        return False, None, None, None