#!/usr/bin/env python3
"""
Test suite for PnP (Perspective-n-Point) utilities.

Tests the solve_pnp_ideal and solve_pnp_ransac_ideal functions for
accuracy, robustness, and proper coordinate frame handling.
"""

import pytest
import numpy as np
from typing import Tuple

# Skip tests if OpenCV not available
cv2 = pytest.importorskip("cv2", reason="OpenCV (cv2) is required for PnP tests")

from src.utils.pnp_utils import solve_pnp_ideal, solve_pnp_ransac_ideal


class TestPnPUtils:
    """Test suite for PnP utility functions."""
    
    def generate_test_scene(self, noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a synthetic test scene with known geometry.
        
        Args:
            noise_std: Standard deviation of noise to add to ideal coordinates
            
        Returns:
            points_3d: Nx3 world points
            ideal_coords: Nx2 ideal coordinates
            R_gt: Ground truth rotation matrix (camera-to-world)  
            t_gt: Ground truth translation vector (camera position in world)
        """
        # Ground truth camera pose (camera-to-world transform)
        angle = np.radians(30)
        R_gt = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])  # 30° rotation around Z-axis
        t_gt = np.array([2.0, 1.0, 1.5])  # Camera position in world
        
        # Generate 3D points in world frame (in front of camera)
        points_3d = np.array([
            [0, 0, 5],    # Center point
            [2, 0, 5],    # Right
            [-2, 0, 5],   # Left  
            [0, 2, 5],    # Up
            [0, -2, 5],   # Down
            [1, 1, 6],    # Diagonal
            [-1, -1, 4],  # Opposite diagonal
            [3, -1, 7],   # Far right-down
        ], dtype=np.float64)
        
        # Transform to camera frame: p_C = R_gt^T @ (p_W - t_gt)
        # This is world-to-camera transform
        points_camera = np.array([R_gt.T @ (p - t_gt) for p in points_3d])
        
        # Project to ideal coordinates (normalized image plane)
        ideal_coords = np.array([
            [p[0] / p[2], p[1] / p[2]] for p in points_camera
        ])
        
        # Add noise if requested
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, ideal_coords.shape)
            ideal_coords += noise
        
        return points_3d, ideal_coords, R_gt, t_gt
    
    def pose_error(self, R_est: np.ndarray, t_est: np.ndarray, 
                   R_gt: np.ndarray, t_gt: np.ndarray) -> Tuple[float, float]:
        """
        Compute pose estimation errors.
        
        Args:
            R_est, t_est: Estimated camera-to-world pose
            R_gt, t_gt: Ground truth camera-to-world pose
            
        Returns:
            rotation_error_deg: Rotation error in degrees
            translation_error: Translation error magnitude
        """
        # Rotation error using trace formula
        R_diff = R_gt.T @ R_est  # Relative rotation
        angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        rotation_error_deg = np.degrees(angle)
        
        # Translation error
        translation_error = np.linalg.norm(t_est - t_gt)
        
        return rotation_error_deg, translation_error
    
    def test_solve_pnp_ideal_basic(self):
        """Test basic functionality of solve_pnp_ideal with perfect data."""
        points_3d, ideal_coords, R_gt, t_gt = self.generate_test_scene()
        
        # Solve PnP
        success, R_est, t_est = solve_pnp_ideal(points_3d, ideal_coords)
        
        # Should succeed
        assert success, "PnP should succeed with good data"
        assert R_est is not None, "Rotation should be returned"
        assert t_est is not None, "Translation should be returned"
        
        # Check accuracy
        rot_err, trans_err = self.pose_error(R_est, t_est, R_gt, t_gt)
        
        print(f"Perfect data errors: rotation={rot_err:.4f}°, translation={trans_err:.6f}m")
        
        # Should be very accurate for synthetic data
        assert rot_err < 0.1, f"Rotation error too high: {rot_err:.4f}° > 0.1°"
        assert trans_err < 0.01, f"Translation error too high: {trans_err:.6f}m > 0.01m"
    
    def test_solve_pnp_ideal_with_noise(self):
        """Test solve_pnp_ideal with noisy measurements."""
        # Test with different noise levels
        noise_levels = [0.001, 0.005, 0.01]  # Increasing noise in ideal coordinates
        
        for noise_std in noise_levels:
            # Set random seed for reproducibility 
            np.random.seed(42)
            points_3d, ideal_coords, R_gt, t_gt = self.generate_test_scene(noise_std=noise_std)
            
            success, R_est, t_est = solve_pnp_ideal(points_3d, ideal_coords)
            
            assert success, f"PnP should succeed with noise std={noise_std}"
            
            rot_err, trans_err = self.pose_error(R_est, t_est, R_gt, t_gt)
            
            print(f"Noise std={noise_std}: rotation={rot_err:.3f}°, translation={trans_err:.4f}m")
            
            # Errors should increase with noise but remain reasonable
            if noise_std <= 0.001:
                assert rot_err < 1.0, f"Rotation error too high for low noise: {rot_err:.3f}°"
                assert trans_err < 0.1, f"Translation error too high for low noise: {trans_err:.4f}m"
            elif noise_std <= 0.01:
                assert rot_err < 10.0, f"Rotation error too high for medium noise: {rot_err:.3f}°"
                assert trans_err < 1.0, f"Translation error too high for medium noise: {trans_err:.4f}m"
    
    def test_solve_pnp_ideal_insufficient_points(self):
        """Test handling of insufficient points."""
        # Too few points
        points_3d = np.array([[0, 0, 5], [1, 0, 5], [0, 1, 5]])  # Only 3 points
        ideal_coords = np.array([[0, 0], [0.2, 0], [0, 0.2]])
        
        success, R_est, t_est = solve_pnp_ideal(points_3d, ideal_coords)
        
        # Should fail gracefully
        assert not success, "PnP should fail with insufficient points"
        assert R_est is None, "No rotation should be returned on failure"
        assert t_est is None, "No translation should be returned on failure"
    
    def test_solve_pnp_ideal_degenerate_configuration(self):
        """Test handling of degenerate point configurations."""
        # Coplanar points (bad for PnP)
        points_3d = np.array([
            [0, 0, 5],
            [1, 0, 5], 
            [2, 0, 5],
            [3, 0, 5]
        ])  # All points on same line
        
        # Project to ideal coordinates  
        ideal_coords = np.array([
            [0, 0],
            [0.2, 0],
            [0.4, 0], 
            [0.6, 0]
        ])
        
        success, R_est, t_est = solve_pnp_ideal(points_3d, ideal_coords)
        
        # May succeed or fail, but shouldn't crash
        # If it succeeds, the result may be inaccurate
        if success:
            print("Warning: PnP succeeded on degenerate configuration")
    
    def test_solve_pnp_ideal_coordinate_frames(self):
        """Test that coordinate frame conventions are correct."""
        points_3d, ideal_coords, R_gt, t_gt = self.generate_test_scene()
        
        success, R_est, t_est = solve_pnp_ideal(points_3d, ideal_coords)
        assert success, "PnP should succeed"
        
        # Verify the returned pose transforms points correctly
        # Transform world points to camera frame using estimated pose
        points_camera_est = np.array([R_est.T @ (p - t_est) for p in points_3d])
        
        # Project these to ideal coordinates
        ideal_coords_est = np.array([
            [p[0] / p[2], p[1] / p[2]] for p in points_camera_est
        ])
        
        # Should match original ideal coordinates closely
        reprojection_error = np.mean(np.linalg.norm(ideal_coords_est - ideal_coords, axis=1))
        
        print(f"Reprojection error: {reprojection_error:.6f}")
        assert reprojection_error < 0.01, f"Reprojection error too high: {reprojection_error:.6f}"
    
    def test_solve_pnp_ransac_ideal_basic(self):
        """Test basic functionality of RANSAC PnP solver."""
        points_3d, ideal_coords, R_gt, t_gt = self.generate_test_scene()
        
        # Add one outlier
        points_3d_outlier = np.vstack([points_3d, [[10, 10, 10]]])
        ideal_coords_outlier = np.vstack([ideal_coords, [[2.0, 2.0]]])  # Bad observation
        
        success, R_est, t_est, inliers = solve_pnp_ransac_ideal(
            points_3d_outlier, ideal_coords_outlier,
            reprojection_threshold=0.1
        )
        
        assert success, "RANSAC PnP should succeed"
        assert R_est is not None and t_est is not None, "Pose should be returned"
        assert inliers is not None, "Inlier mask should be returned"
        
        # Should identify most points as inliers, outlier as outlier
        num_inliers = np.sum(inliers)
        print(f"RANSAC found {num_inliers}/{len(inliers)} inliers")
        
        assert num_inliers >= 6, f"Too few inliers: {num_inliers}"
        assert not inliers[-1], "Last point (outlier) should be rejected"
        
        # Check pose accuracy (should be good despite outlier)
        rot_err, trans_err = self.pose_error(R_est, t_est, R_gt, t_gt)
        
        print(f"RANSAC errors: rotation={rot_err:.3f}°, translation={trans_err:.4f}m")
        assert rot_err < 5.0, f"RANSAC rotation error too high: {rot_err:.3f}°"
        assert trans_err < 0.5, f"RANSAC translation error too high: {trans_err:.4f}m"
    
    def test_solve_pnp_ransac_ideal_many_outliers(self):
        """Test RANSAC with many outliers."""
        points_3d, ideal_coords, R_gt, t_gt = self.generate_test_scene()
        
        # Add multiple outliers (50% outlier ratio)
        num_outliers = len(points_3d)
        outlier_points_3d = np.random.uniform(-10, 10, (num_outliers, 3))
        outlier_ideal_coords = np.random.uniform(-2, 2, (num_outliers, 2))
        
        combined_points_3d = np.vstack([points_3d, outlier_points_3d])
        combined_ideal_coords = np.vstack([ideal_coords, outlier_ideal_coords])
        
        success, R_est, t_est, inliers = solve_pnp_ransac_ideal(
            combined_points_3d, combined_ideal_coords,
            reprojection_threshold=0.1,
            confidence=0.99
        )
        
        if success:
            num_inliers = np.sum(inliers)
            print(f"High-outlier test: {num_inliers}/{len(inliers)} inliers")
            
            # Should still find a good subset of original points
            # Check if inliers correspond to original (non-outlier) points
            original_inliers = np.sum(inliers[:len(points_3d)])
            print(f"Original points marked as inliers: {original_inliers}/{len(points_3d)}")
            
            assert original_inliers >= 4, "Should find at least 4 original points as inliers"
        else:
            print("RANSAC failed with high outlier ratio (expected behavior)")
    
    def test_reprojection_threshold_validation(self):
        """Test that reprojection threshold is properly applied."""
        points_3d, ideal_coords, R_gt, t_gt = self.generate_test_scene()
        
        # Add systematic error to make reprojection poor
        biased_ideal_coords = ideal_coords + 0.5  # Large bias
        
        # Test with strict threshold
        success_strict, _, _ = solve_pnp_ideal(
            points_3d, biased_ideal_coords, 
            reprojection_threshold=0.1
        )
        
        # Test with loose threshold
        success_loose, _, _ = solve_pnp_ideal(
            points_3d, biased_ideal_coords,
            reprojection_threshold=10.0
        )
        
        # Strict threshold should reject, loose should accept
        assert not success_strict, "Strict threshold should reject poor solution"
        assert success_loose, "Loose threshold should accept poor solution"
    
    def test_edge_cases(self):
        """Test various edge cases."""
        # Empty input
        success, R, t = solve_pnp_ideal(np.array([]), np.array([]))
        assert not success, "Empty input should fail"
        
        # Mismatched array sizes
        points_3d = np.array([[0, 0, 5], [1, 0, 5]])
        ideal_coords = np.array([[0, 0]])  # Only 1 coordinate for 2 points
        
        success, R, t = solve_pnp_ideal(points_3d, ideal_coords)
        assert not success, "Mismatched arrays should fail"
        
        # Points behind camera
        points_3d = np.array([
            [0, 0, -5],   # Behind camera
            [1, 0, -5],
            [0, 1, -5], 
            [1, 1, -5]
        ])
        ideal_coords = np.array([[0, 0], [0.2, 0], [0, 0.2], [0.2, 0.2]])
        
        success, R, t = solve_pnp_ideal(points_3d, ideal_coords)
        # May succeed or fail, but shouldn't crash
    
    def test_accuracy_comparison_standard_vs_ransac(self):
        """Compare accuracy of standard PnP vs RANSAC PnP on clean data."""
        points_3d, ideal_coords, R_gt, t_gt = self.generate_test_scene()
        
        # Standard PnP
        success1, R1, t1 = solve_pnp_ideal(points_3d, ideal_coords)
        
        # RANSAC PnP  
        success2, R2, t2, inliers = solve_pnp_ransac_ideal(points_3d, ideal_coords)
        
        assert success1 and success2, "Both methods should succeed on clean data"
        
        # Compute errors
        rot_err1, trans_err1 = self.pose_error(R1, t1, R_gt, t_gt)
        rot_err2, trans_err2 = self.pose_error(R2, t2, R_gt, t_gt)
        
        print(f"Standard PnP: rot={rot_err1:.4f}°, trans={trans_err1:.6f}m")
        print(f"RANSAC PnP:   rot={rot_err2:.4f}°, trans={trans_err2:.6f}m")
        
        # Both should be very accurate on clean data
        assert rot_err1 < 1.0 and rot_err2 < 1.0, "Both methods should be accurate"
        assert trans_err1 < 0.1 and trans_err2 < 0.1, "Both methods should be accurate"
        
        # All points should be inliers for RANSAC
        assert np.sum(inliers) == len(points_3d), "All points should be inliers on clean data"