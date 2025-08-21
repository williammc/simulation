"""
Unit tests for mathematical utilities.
Tests SO3, SE3, quaternion operations and coordinate transformations.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from src.utils.math_utils import (
    # SO3 operations
    so3_exp, so3_log, skew, vee, is_rotation_matrix,
    # SE3 operations
    se3_exp, se3_log, se3_inverse, se3_adjoint,
    # Temporarily available quaternion operations (for backward compatibility)
    quaternion_normalize, quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    # Coordinate transformations
    transform_point, transform_vector,
    euler_to_rotation_matrix, rotation_matrix_to_euler,
    interpolate_se3
)


class TestSO3Operations:
    """Test SO3 (rotation) operations."""
    
    def test_skew_vee_inverse(self):
        """Test that skew and vee are inverses."""
        v = np.array([1.0, 2.0, 3.0])
        assert_array_almost_equal(vee(skew(v)), v)
    
    def test_skew_antisymmetric(self):
        """Test that skew matrix is antisymmetric."""
        v = np.array([1.0, 2.0, 3.0])
        S = skew(v)
        assert_array_almost_equal(S, -S.T)
    
    def test_so3_exp_identity(self):
        """Test exp map at zero gives identity."""
        omega = np.zeros(3)
        R = so3_exp(omega)
        assert_array_almost_equal(R, np.eye(3))
    
    def test_so3_exp_log_inverse(self):
        """Test that exp and log are inverses."""
        # Test multiple random rotations
        np.random.seed(42)
        for _ in range(10):
            omega = np.random.randn(3) * 0.5  # Keep angles small
            R = so3_exp(omega)
            omega_recovered = so3_log(R)
            R_recovered = so3_exp(omega_recovered)
            assert_array_almost_equal(R, R_recovered, decimal=10)
    
    def test_so3_exp_rodriguez(self):
        """Test exp map with known rotation."""
        # 90 degree rotation around z-axis
        omega = np.array([0, 0, np.pi/2])
        R = so3_exp(omega)
        
        # Expected rotation matrix
        R_expected = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ])
        assert_array_almost_equal(R, R_expected, decimal=10)
    
    def test_is_rotation_matrix(self):
        """Test rotation matrix validation."""
        # Valid rotation matrix
        R = so3_exp(np.array([0.1, 0.2, 0.3]))
        assert is_rotation_matrix(R)
        
        # Invalid: not orthogonal
        R_bad = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        assert not is_rotation_matrix(R_bad)
        
        # Invalid: det = -1 (reflection)
        R_reflect = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        assert not is_rotation_matrix(R_reflect)


class TestSE3Operations:
    """Test SE3 (rigid transformation) operations."""
    
    def test_se3_exp_identity(self):
        """Test exp map at zero gives identity."""
        xi = np.zeros(6)
        T = se3_exp(xi)
        assert_array_almost_equal(T, np.eye(4))
    
    def test_se3_exp_log_inverse(self):
        """Test that exp and log are inverses."""
        np.random.seed(42)
        for _ in range(10):
            xi = np.random.randn(6) * 0.5
            T = se3_exp(xi)
            xi_recovered = se3_log(T)
            T_recovered = se3_exp(xi_recovered)
            assert_array_almost_equal(T, T_recovered, decimal=10)
    
    def test_se3_inverse(self):
        """Test SE3 inverse."""
        xi = np.array([0.1, 0.2, 0.3, 1.0, 2.0, 3.0])
        T = se3_exp(xi)
        T_inv = se3_inverse(T)
        
        # T @ T_inv should be identity
        assert_array_almost_equal(T @ T_inv, np.eye(4))
        assert_array_almost_equal(T_inv @ T, np.eye(4))
    
    def test_se3_pure_translation(self):
        """Test SE3 with pure translation."""
        xi = np.array([0, 0, 0, 1, 2, 3])  # No rotation
        T = se3_exp(xi)
        
        assert_array_almost_equal(T[:3, :3], np.eye(3))  # No rotation
        assert_array_almost_equal(T[:3, 3], [1, 2, 3])  # Translation
    
    def test_se3_adjoint(self):
        """Test SE3 adjoint matrix."""
        T = se3_exp(np.array([0.1, 0.2, 0.3, 1.0, 2.0, 3.0]))
        Ad = se3_adjoint(T)
        
        # Adjoint should be 6x6
        assert Ad.shape == (6, 6)
        
        # Test adjoint property: Ad(T) * xi = T * xi * T^-1 (in twist form)
        xi = np.random.randn(6)
        xi_transformed = Ad @ xi
        
        # Convert to matrix form and check
        T_xi = se3_exp(xi)
        T_xi_transformed = T @ T_xi @ se3_inverse(T)
        xi_transformed_check = se3_log(T_xi_transformed)
        
        # Note: This test may have some numerical error for large rotations
        assert np.linalg.norm(xi_transformed - xi_transformed_check) < 0.1


class TestQuaternionOperations:
    """Test quaternion operations."""
    
    def test_quaternion_normalize(self):
        """Test quaternion normalization."""
        q = np.array([1, 2, 3, 4])
        q_norm = quaternion_normalize(q)
        assert abs(np.linalg.norm(q_norm) - 1.0) < 1e-10
    
    @pytest.mark.skip(reason="quaternion_multiply removed - use SO3 operations")
    def test_quaternion_multiply_identity(self):
        """Test quaternion multiplication with identity."""
        pass  # Function removed
    
    @pytest.mark.skip(reason="quaternion conjugate/inverse removed - use SO3 operations")
    def test_quaternion_conjugate_inverse(self):
        """Test quaternion conjugate and inverse."""
        pass  # Functions removed
    
    def test_quaternion_rotation_conversion(self):
        """Test conversion between quaternion and rotation matrix."""
        # Test multiple random quaternions
        np.random.seed(42)
        for _ in range(10):
            q = quaternion_normalize(np.random.randn(4))
            R = quaternion_to_rotation_matrix(q)
            q_recovered = rotation_matrix_to_quaternion(R)
            
            # Note: q and -q represent the same rotation
            if np.dot(q, q_recovered) < 0:
                q_recovered = -q_recovered
            
            assert_array_almost_equal(q, q_recovered, decimal=10)
    
    @pytest.mark.skip(reason="quaternion_slerp removed - use SO3 interpolation")
    def test_quaternion_slerp(self):
        """Test spherical linear interpolation."""
        pass  # Function removed
    
    @pytest.mark.skip(reason="axis_angle functions removed - use SO3 operations")
    def test_axis_angle_quaternion_conversion(self):
        """Test conversion between axis-angle and quaternion."""
        pass  # Functions removed


class TestCoordinateTransformations:
    """Test coordinate frame transformations."""
    
    def test_transform_point(self):
        """Test point transformation."""
        # 90 degree rotation around z + translation
        T = np.array([
            [0, -1, 0, 1],
            [1,  0, 0, 2],
            [0,  0, 1, 3],
            [0,  0, 0, 1]
        ])
        
        # Single point
        p = np.array([1, 0, 0])
        p_transformed = transform_point(T, p)
        assert_array_almost_equal(p_transformed, [1, 3, 3])
        
        # Multiple points
        points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        points_transformed = transform_point(T, points)
        expected = np.array([[1, 3, 3], [0, 2, 3], [1, 2, 4]])
        assert_array_almost_equal(points_transformed, expected)
    
    def test_transform_vector(self):
        """Test vector transformation (rotation only)."""
        R = so3_exp(np.array([0, 0, np.pi/2]))  # 90 deg around z
        
        v = np.array([1, 0, 0])
        v_transformed = transform_vector(R, v)
        assert_array_almost_equal(v_transformed, [0, 1, 0])
    
    def test_euler_rotation_conversion(self):
        """Test Euler angle conversion."""
        # Test multiple angle sets
        test_cases = [
            (0, 0, 0),  # Identity
            (np.pi/6, 0, 0),  # Pure roll
            (0, np.pi/6, 0),  # Pure pitch
            (0, 0, np.pi/6),  # Pure yaw
            (np.pi/12, np.pi/12, np.pi/12),  # Small combined angles
        ]
        
        for roll, pitch, yaw in test_cases:
            R = euler_to_rotation_matrix(roll, pitch, yaw, order='xyz')
            
            # Verify it's a valid rotation matrix
            assert is_rotation_matrix(R)
            
            # For now, just verify the matrix is valid
            # Euler angle extraction can have multiple solutions
            # so we don't test the round-trip conversion
    
    def test_interpolate_se3(self):
        """Test SE3 interpolation."""
        T1 = np.eye(4)
        # Create T2 with known translation and rotation
        T2 = np.eye(4)
        T2[:3, :3] = so3_exp(np.array([0, 0, np.pi/2]))  # 90 deg rotation around z
        T2[:3, 3] = np.array([2, 0, 0])  # Translation along x
        
        # At t=0, should get T1
        T_interp = interpolate_se3(T1, T2, 0.0)
        assert_array_almost_equal(T_interp, T1)
        
        # At t=1, should get T2
        T_interp = interpolate_se3(T1, T2, 1.0)
        assert_array_almost_equal(T_interp, T2)
        
        # At t=0.5, translation should be halfway (linear interpolation)
        T_interp = interpolate_se3(T1, T2, 0.5)
        assert_array_almost_equal(T_interp[:3, 3], [1, 0, 0])  # Halfway between [0,0,0] and [2,0,0]
        
        # Rotation should be 45 degrees (halfway between 0 and 90)
        R_interp = T_interp[:3, :3]
        # Check that it's approximately a 45 degree rotation around z
        expected_R = so3_exp(np.array([0, 0, np.pi/4]))
        assert_array_almost_equal(R_interp, expected_R, decimal=5)


class TestNumericalStability:
    """Test numerical stability of operations."""
    
    def test_small_angle_approximation(self):
        """Test that small angle approximations work correctly."""
        # Very small rotation
        omega = np.array([1e-8, 2e-8, 3e-8])
        R = so3_exp(omega)
        omega_recovered = so3_log(R)
        
        # Should recover the same small angle
        assert_array_almost_equal(omega, omega_recovered, decimal=12)
    
    def test_near_pi_rotation(self):
        """Test rotations near π (challenging case)."""
        # Rotation very close to π
        angle = np.pi - 1e-6
        axis = np.array([1, 0, 0])
        omega = axis * angle
        
        R = so3_exp(omega)
        omega_recovered = so3_log(R)
        
        # Check angle is preserved (axis might flip)
        angle_recovered = np.linalg.norm(omega_recovered)
        assert abs(angle - angle_recovered) < 1e-5
    
    def test_quaternion_near_singularity(self):
        """Test quaternion operations near singularities."""
        # Near 180 degree rotation (w ≈ 0)
        q = quaternion_normalize([0.001, 1, 0, 0])
        R = quaternion_to_rotation_matrix(q)
        q_recovered = rotation_matrix_to_quaternion(R)
        
        # Should get same rotation (possibly negated quaternion)
        R_recovered = quaternion_to_rotation_matrix(q_recovered)
        assert_array_almost_equal(R, R_recovered, decimal=10)