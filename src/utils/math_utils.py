"""
Mathematical utilities for SLAM simulation.
Includes SO3, SE3, quaternion operations and coordinate transformations.
"""

import numpy as np
from typing import Tuple, Union, Optional


# ============================================================================
# SO3 Operations (3D Rotations)
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
    theta = np.linalg.norm(omega)
    
    if theta < 1e-8:
        # First-order approximation for small angles
        return np.eye(3) + skew(omega)
    
    omega_hat = skew(omega)
    # Rodrigues' formula
    R = np.eye(3) + (np.sin(theta) / theta) * omega_hat + \
        ((1 - np.cos(theta)) / (theta ** 2)) * (omega_hat @ omega_hat)
    
    return R


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
        # Project to nearest rotation matrix
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
    
    trace = np.trace(R)
    
    if trace >= 3.0 - 1e-6:
        # Near identity, use first-order approximation
        return vee(0.5 * (R - R.T))
    elif trace <= -1.0 + 1e-6:
        # Rotation by π
        # Find the axis of rotation
        vals = np.diag(R)
        k = np.argmax(vals)
        axis = np.zeros(3)
        axis[k] = 1.0
        axis = axis + R[:, k]
        axis = axis / np.linalg.norm(axis)
        return np.pi * axis
    else:
        # General case
        theta = np.arccos((trace - 1) / 2)
        return (theta / (2 * np.sin(theta))) * vee(R - R.T)


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
# Quaternion Operations
# ============================================================================

def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions (Hamilton product).
    Convention: q = [w, x, y, z] where w is scalar part.
    
    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]
    
    Returns:
        Product quaternion q1 * q2
    """
    q1 = np.asarray(q1).flatten()
    q2 = np.asarray(q2).flatten()
    
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Compute quaternion conjugate.
    
    Args:
        q: Quaternion [w, x, y, z]
    
    Returns:
        Conjugate quaternion [w, -x, -y, -z]
    """
    q = np.asarray(q).flatten()
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """
    Compute quaternion inverse.
    
    Args:
        q: Quaternion [w, x, y, z]
    
    Returns:
        Inverse quaternion
    """
    q = np.asarray(q).flatten()
    norm_sq = np.sum(q ** 2)
    if norm_sq < 1e-10:
        raise ValueError("Cannot invert zero quaternion")
    return quaternion_conjugate(q) / norm_sq


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
    w, x, y, z = q
    
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    
    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion.
    Uses Shepperd's method for numerical stability.
    
    Args:
        R: 3x3 rotation matrix
    
    Returns:
        Quaternion [w, x, y, z]
    """
    R = np.asarray(R)
    
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return quaternion_normalize(np.array([w, x, y, z]))


def quaternion_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between two quaternions.
    
    Args:
        q1: Start quaternion [w, x, y, z]
        q2: End quaternion [w, x, y, z]
        t: Interpolation parameter [0, 1]
    
    Returns:
        Interpolated quaternion
    """
    q1 = quaternion_normalize(q1)
    q2 = quaternion_normalize(q2)
    
    # Compute angle between quaternions
    dot = np.dot(q1, q2)
    
    # If quaternions are nearly opposite, flip one
    if dot < 0:
        q2 = -q2
        dot = -dot
    
    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        q = q1 + t * (q2 - q1)
        return quaternion_normalize(q)
    
    # Compute slerp
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    
    return w1 * q1 + w2 * q2


def axis_angle_to_quaternion(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Convert axis-angle representation to quaternion.
    
    Args:
        axis: 3x1 rotation axis (will be normalized)
        angle: Rotation angle in radians
    
    Returns:
        Quaternion [w, x, y, z]
    """
    axis = np.asarray(axis).flatten()
    axis = axis / np.linalg.norm(axis)
    
    half_angle = angle / 2
    sin_half = np.sin(half_angle)
    
    w = np.cos(half_angle)
    x = axis[0] * sin_half
    y = axis[1] * sin_half
    z = axis[2] * sin_half
    
    return np.array([w, x, y, z])


def quaternion_to_axis_angle(q: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Convert quaternion to axis-angle representation.
    
    Args:
        q: Quaternion [w, x, y, z]
    
    Returns:
        Tuple of (axis, angle) where axis is 3x1 and angle is in radians
    """
    q = quaternion_normalize(q)
    w, x, y, z = q
    
    # Compute angle
    angle = 2 * np.arccos(np.clip(w, -1.0, 1.0))
    
    # Compute axis
    if angle < 1e-6:
        # Near identity, arbitrary axis
        axis = np.array([1.0, 0.0, 0.0])
    else:
        sin_half = np.sin(angle / 2)
        if abs(sin_half) < 1e-6:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = np.array([x, y, z]) / sin_half
    
    return axis, angle


# ============================================================================
# Coordinate Frame Transformations
# ============================================================================

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
    Convert Euler angles to rotation matrix.
    
    Args:
        roll: Rotation around x-axis (radians)
        pitch: Rotation around y-axis (radians)
        yaw: Rotation around z-axis (radians)
        order: Rotation order ('xyz', 'zyx', etc.)
    
    Returns:
        3x3 rotation matrix
    """
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    
    rotations = {'x': Rx, 'y': Ry, 'z': Rz}
    
    R = np.eye(3)
    for axis in order.lower():
        if axis in rotations:
            R = R @ rotations[axis]
    
    return R


def rotation_matrix_to_euler(R: np.ndarray, order: str = 'xyz') -> Tuple[float, float, float]:
    """
    Convert rotation matrix to Euler angles.
    Note: Multiple solutions exist, returns one valid solution.
    
    Args:
        R: 3x3 rotation matrix
        order: Rotation order (currently only 'xyz' supported)
    
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    if order.lower() == 'xyz':
        # Extract Euler angles for XYZ order
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        
        return roll, pitch, yaw
    else:
        raise NotImplementedError(f"Order {order} not implemented")


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
    
    # Convert rotations to quaternions for slerp
    q1 = rotation_matrix_to_quaternion(R1)
    q2 = rotation_matrix_to_quaternion(R2)
    
    # Interpolate rotation using slerp
    q_interp = quaternion_slerp(q1, q2, t)
    R_interp = quaternion_to_rotation_matrix(q_interp)
    
    # Linear interpolation for translation
    t_interp = (1 - t) * t1 + t * t2
    
    # Construct interpolated transformation
    T_interp = np.eye(4)
    T_interp[:3, :3] = R_interp
    T_interp[:3, 3] = t_interp
    
    return T_interp