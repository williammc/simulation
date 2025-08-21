"""
Mathematical utilities for SLAM simulation.
Uses scipy.spatial.transform.Rotation for robust implementations.
"""

import numpy as np
from typing import Tuple, Union, Optional
from scipy.spatial.transform import Rotation


# ============================================================================
# SO3 Operations (3D Rotations) - Using scipy.spatial.transform.Rotation
# ============================================================================

def so3_exp(omega: np.ndarray) -> np.ndarray:
    """
    Exponential map from so3 to SO3.
    Converts axis-angle vector to rotation matrix.
    
    Args:
        omega: 3x1 axis-angle vector (rotation vector)
    
    Returns:
        3x3 rotation matrix
    """
    omega = np.asarray(omega).flatten()
    
    # Use scipy's Rotation for robust implementation
    if np.linalg.norm(omega) < 1e-8:
        return np.eye(3)
    
    rotation = Rotation.from_rotvec(omega)
    return rotation.as_matrix()


def so3_log(R: np.ndarray) -> np.ndarray:
    """
    Logarithmic map from SO3 to so3.
    Converts rotation matrix to axis-angle vector.
    
    Args:
        R: 3x3 rotation matrix
    
    Returns:
        3x1 axis-angle vector
    """
    R = np.asarray(R)
    
    # Ensure R is a valid rotation matrix
    if not is_rotation_matrix(R):
        # Project to nearest rotation matrix using SVD
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            # Fix reflection
            Vt[-1, :] *= -1
            R = U @ Vt
    
    # Use scipy's Rotation for robust implementation
    rotation = Rotation.from_matrix(R)
    return rotation.as_rotvec()


def skew(v: np.ndarray) -> np.ndarray:
    """
    Convert 3D vector to skew-symmetric matrix (hat operator).
    
    Args:
        v: 3x1 vector
    
    Returns:
        3x3 skew-symmetric matrix
    """
    v = np.asarray(v).flatten()
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def vee(M: np.ndarray) -> np.ndarray:
    """
    Convert skew-symmetric matrix to vector (vee operator).
    
    Args:
        M: 3x3 skew-symmetric matrix
    
    Returns:
        3x1 vector
    """
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


# Aliases for compatibility
exp_so3 = so3_exp  # Alias for exponential map
log_so3 = so3_log  # Alias for logarithmic map


def is_rotation_matrix(R: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Check if matrix is a valid rotation matrix (orthogonal with det=1).
    
    Args:
        R: Matrix to check
        tol: Tolerance for numerical errors
    
    Returns:
        True if R is a valid rotation matrix
    """
    R = np.asarray(R)
    if R.shape != (3, 3):
        return False
    
    # Check orthogonality: R @ R.T = I
    should_be_I = R @ R.T
    if not np.allclose(should_be_I, np.eye(3), atol=tol):
        return False
    
    # Check determinant = 1
    if not np.isclose(np.linalg.det(R), 1.0, atol=tol):
        return False
    
    return True


# ============================================================================
# SE3 Operations (3D Rigid Transformations)
# ============================================================================

def se3_exp(xi: np.ndarray) -> np.ndarray:
    """
    Exponential map from se3 to SE3.
    Converts twist vector to transformation matrix.
    
    Args:
        xi: 6x1 twist vector [angular; linear]
    
    Returns:
        4x4 transformation matrix
    """
    xi = np.asarray(xi).flatten()
    omega = xi[:3]  # Angular part
    v = xi[3:]      # Linear part
    
    theta = np.linalg.norm(omega)
    
    if theta < 1e-8:
        # First-order approximation for small angles
        R = np.eye(3) + skew(omega)
        t = v
    else:
        R = so3_exp(omega)
        omega_hat = skew(omega)
        
        # Compute V matrix (left Jacobian of SO3)
        V = np.eye(3) + ((1 - np.cos(theta)) / (theta ** 2)) * omega_hat + \
            ((theta - np.sin(theta)) / (theta ** 3)) * (omega_hat @ omega_hat)
        
        t = V @ v
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """
    Logarithmic map from SE3 to se3.
    Converts transformation matrix to twist vector.
    
    Args:
        T: 4x4 transformation matrix
    
    Returns:
        6x1 twist vector [angular; linear]
    """
    T = np.asarray(T)
    R = T[:3, :3]
    t = T[:3, 3]
    
    omega = so3_log(R)
    theta = np.linalg.norm(omega)
    
    if theta < 1e-8:
        # Near identity
        v = t
    else:
        omega_hat = skew(omega)
        
        # Compute V inverse
        V_inv = np.eye(3) - 0.5 * omega_hat + \
                (1 / (theta ** 2)) * (1 - (theta * np.sin(theta)) / (2 * (1 - np.cos(theta)))) * \
                (omega_hat @ omega_hat)
        
        v = V_inv @ t
    
    return np.concatenate([omega, v])


def se3_inverse(T: np.ndarray) -> np.ndarray:
    """
    Compute inverse of SE3 transformation matrix.
    
    Args:
        T: 4x4 transformation matrix
    
    Returns:
        4x4 inverse transformation matrix
    """
    T = np.asarray(T)
    R = T[:3, :3]
    t = T[:3, 3]
    
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    
    return T_inv


def se3_adjoint(T: np.ndarray) -> np.ndarray:
    """
    Compute adjoint matrix of SE3 transformation.
    
    Args:
        T: 4x4 transformation matrix
    
    Returns:
        6x6 adjoint matrix
    """
    T = np.asarray(T)
    R = T[:3, :3]
    t = T[:3, 3]
    
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[3:, :3] = skew(t) @ R
    
    return Ad


# ============================================================================
# SO3 Helper Functions
# ============================================================================

def project_to_so3(R: np.ndarray) -> np.ndarray:
    """
    Project a matrix to the SO3 manifold using SVD.
    Ensures the result is a valid rotation matrix.
    
    Args:
        R: 3x3 matrix (possibly not orthogonal)
    
    Returns:
        3x3 rotation matrix on SO3 manifold
    """
    R = np.asarray(R).reshape(3, 3)
    
    # Use SVD to find closest rotation matrix
    U, _, Vt = np.linalg.svd(R)
    R_projected = U @ Vt
    
    # Ensure determinant is +1 (not -1)
    if np.linalg.det(R_projected) < 0:
        Vt[-1, :] *= -1
        R_projected = U @ Vt
    
    return R_projected


def so3_interpolate(R1: np.ndarray, R2: np.ndarray, t: float) -> np.ndarray:
    """
    Geodesic interpolation between two rotation matrices on SO3.
    
    Args:
        R1: First rotation matrix (3x3)
        R2: Second rotation matrix (3x3) 
        t: Interpolation parameter [0, 1]
    
    Returns:
        Interpolated rotation matrix
    """
    R1 = np.asarray(R1).reshape(3, 3)
    R2 = np.asarray(R2).reshape(3, 3)
    
    # Compute relative rotation
    R_rel = R1.T @ R2
    
    # Convert to axis-angle
    omega = so3_log(R_rel)
    
    # Scale by interpolation parameter
    omega_t = t * omega
    
    # Convert back to rotation matrix
    R_t = so3_exp(omega_t)
    
    # Apply to initial rotation
    return R1 @ R_t

# ============================================================================
# Legacy Quaternion Operations (TO BE REMOVED IN TASK 3)
# These are temporarily kept for estimators that still use quaternions internally
# ============================================================================

def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion [w, x, y, z].
    Temporarily kept for backward compatibility."""
    R = np.asarray(R)
    r = Rotation.from_matrix(R)
    q = r.as_quat()  # Returns [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]])

# quaternion_multiply removed - use SO3 rotation matrix multiplication instead

def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """
    Normalize quaternion to unit norm.
    
    Args:
        q: Quaternion [w, x, y, z]
    
    Returns:
        Normalized quaternion
    """
    q = np.asarray(q).flatten()
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: Quaternion [w, x, y, z]
    
    Returns:
        3x3 rotation matrix
    """
    q = quaternion_normalize(q)
    
    # Convert to scipy format [x, y, z, w] and get matrix
    r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
    return r.as_matrix()


def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Transform 3D point by SE3 transformation.
    
    Args:
        T: 4x4 transformation matrix
        p: 3x1 or Nx3 points
    
    Returns:
        Transformed points
    """
    p = np.asarray(p)
    single_point = p.ndim == 1
    
    if single_point:
        p = p.reshape(1, -1)
    
    # Convert to homogeneous coordinates
    p_hom = np.hstack([p, np.ones((p.shape[0], 1))])
    
    # Transform
    p_transformed = (T @ p_hom.T).T
    
    # Extract 3D coordinates
    result = p_transformed[:, :3]
    
    return result[0] if single_point else result


def transform_vector(R: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Transform 3D vector by rotation matrix (no translation).
    
    Args:
        R: 3x3 rotation matrix
        v: 3x1 or Nx3 vectors
    
    Returns:
        Transformed vectors
    """
    v = np.asarray(v)
    single_vector = v.ndim == 1
    
    if single_vector:
        v = v.reshape(1, -1)
    
    v_transformed = (R @ v.T).T
    
    return v_transformed[0] if single_vector else v_transformed


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float, 
                           order: str = 'xyz') -> np.ndarray:
    """
    Convert Euler angles to rotation matrix using scipy.
    
    Args:
        roll: Rotation around x-axis (radians)
        pitch: Rotation around y-axis (radians)
        yaw: Rotation around z-axis (radians)
        order: Rotation order (e.g., 'xyz', 'XYZ' for extrinsic, 'zyx' for intrinsic)
    
    Returns:
        3x3 rotation matrix
    """
    # Map our roll/pitch/yaw to the appropriate angles based on order
    if order.lower() == 'xyz':
        angles = [roll, pitch, yaw]
    elif order.lower() == 'zyx':
        angles = [yaw, pitch, roll]
    else:
        # For other orders, just use the angles in sequence
        angles = [roll, pitch, yaw]
    
    # Use scipy - uppercase for extrinsic, lowercase for intrinsic
    r = Rotation.from_euler(order, angles)
    return r.as_matrix()


def rotation_matrix_to_euler(R: np.ndarray, order: str = 'xyz') -> Tuple[float, float, float]:
    """
    Convert rotation matrix to Euler angles using scipy.
    
    Args:
        R: 3x3 rotation matrix
        order: Rotation order
    
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    # Use scipy for robust conversion
    r = Rotation.from_matrix(R)
    angles = r.as_euler(order)
    
    # Map back to roll/pitch/yaw based on order
    if order.lower() == 'xyz':
        return angles[0], angles[1], angles[2]
    elif order.lower() == 'zyx':
        return angles[2], angles[1], angles[0]
    else:
        return angles[0], angles[1], angles[2]


def interpolate_se3(T1: np.ndarray, T2: np.ndarray, t: float) -> np.ndarray:
    """
    Interpolate between two SE3 transformations.
    
    Args:
        T1: Starting 4x4 transformation matrix
        T2: Ending 4x4 transformation matrix
        t: Interpolation parameter [0, 1]
    
    Returns:
        Interpolated transformation matrix
    """
    # Extract rotation and translation
    R1, t1 = T1[:3, :3], T1[:3, 3]
    R2, t2 = T2[:3, :3], T2[:3, 3]
    
    # Use scipy for rotation interpolation
    r1 = Rotation.from_matrix(R1)
    r2 = Rotation.from_matrix(R2)
    
    # Interpolate rotation using scipy's slerp
    r_interp = r1 * (r1.inv() * r2) ** t
    R_interp = r_interp.as_matrix()
    
    # Linear interpolation for translation
    t_interp = (1 - t) * t1 + t * t2
    
    # Construct interpolated transformation
    T_interp = np.eye(4)
    T_interp[:3, :3] = R_interp
    T_interp[:3, 3] = t_interp
    
    return T_interp


# ============================================================================
# Additional Utility Functions
# ============================================================================

def random_rotation_matrix() -> np.ndarray:
    """
    Generate a random rotation matrix using scipy.
    
    Returns:
        3x3 rotation matrix
    """
    r = Rotation.random()
    return r.as_matrix()


def random_quaternion() -> np.ndarray:
    """
    Generate a random unit quaternion.
    
    Returns:
        Quaternion [w, x, y, z]
    """
    r = Rotation.random()
    q = r.as_quat()  # Returns [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]])


def rotation_matrix_from_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Find rotation matrix that rotates v1 to v2.
    
    Args:
        v1: Source vector (will be normalized)
        v2: Target vector (will be normalized)
    
    Returns:
        3x3 rotation matrix such that R @ v1 ≈ v2
    """
    v1 = np.asarray(v1).flatten()
    v2 = np.asarray(v2).flatten()
    
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Check if vectors are parallel
    cross = np.cross(v1, v2)
    if np.linalg.norm(cross) < 1e-6:
        if np.dot(v1, v2) > 0:
            return np.eye(3)  # Same direction
        else:
            # Opposite direction - find perpendicular axis
            # Find a vector not parallel to v1
            if abs(v1[0]) < 0.9:
                axis = np.cross(v1, [1, 0, 0])
            else:
                axis = np.cross(v1, [0, 1, 0])
            axis = axis / np.linalg.norm(axis)
            return Rotation.from_rotvec(axis * np.pi).as_matrix()
    
    # General case - use Rodriguez formula via scipy
    axis = cross / np.linalg.norm(cross)
    angle = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
    
    return Rotation.from_rotvec(axis * angle).as_matrix()


def pose_to_matrix(pose) -> np.ndarray:
    """
    Convert pose to 4x4 transformation matrix.
    
    Args:
        pose: Pose object with position and rotation_matrix
    
    Returns:
        4x4 transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = pose.rotation_matrix
    T[:3, 3] = pose.position
    return T