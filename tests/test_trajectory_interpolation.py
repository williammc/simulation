"""
Tests for trajectory interpolation and smoothing.
"""

import pytest
import numpy as np

from src.simulation.trajectory_interpolation import (
    TrajectoryInterpolator, SplineTrajectoryConfig,
    smooth_trajectory, create_bezier_trajectory
)
from src.common.data_structures import Trajectory, TrajectoryState, Pose


class TestSplineInterpolation:
    """Test spline-based trajectory interpolation."""
    
    def test_basic_interpolation(self):
        """Test basic spline interpolation."""
        # Create simple trajectory with few waypoints
        trajectory = Trajectory()
        timestamps = [0.0, 1.0, 2.0, 3.0]
        positions = [
            [0, 0, 0],
            [1, 1, 0],
            [2, 0, 0],
            [3, -1, 0]
        ]
        
        for t, pos in zip(timestamps, positions):
            pose = Pose(
                timestamp=t,
                position=np.array(pos),
                rotation_matrix=np.eye(3)
            )
            trajectory.add_state(TrajectoryState(pose=pose))
        
        # Interpolate
        config = SplineTrajectoryConfig()
        interpolator = TrajectoryInterpolator(config)
        interpolator.fit(trajectory)
        
        # Generate dense trajectory
        dense_traj = interpolator.interpolate(rate=100.0)
        
        assert len(dense_traj.states) > len(trajectory.states)
        assert len(dense_traj.states) == 300  # 3 seconds at 100 Hz
    
    def test_waypoint_preservation(self):
        """Test that original waypoints are preserved in interpolation."""
        trajectory = Trajectory()
        waypoints = [
            (0.0, [0, 0, 0]),
            (1.0, [1, 0, 0]),
            (2.0, [1, 1, 0]),
            (3.0, [0, 1, 0])
        ]
        
        for t, pos in waypoints:
            pose = Pose(
                timestamp=t,
                position=np.array(pos),
                rotation_matrix=np.eye(3)
            )
            trajectory.add_state(TrajectoryState(pose=pose))
        
        config = SplineTrajectoryConfig()
        interpolator = TrajectoryInterpolator(config)
        interpolator.fit(trajectory)
        dense_traj = interpolator.interpolate(rate=100.0)
        
        # Check that original points are preserved
        for t_orig, pos_orig in waypoints:
            # Find closest point in dense trajectory
            closest = min(dense_traj.states, 
                         key=lambda s: abs(s.pose.timestamp - t_orig))
            
            pos_diff = np.linalg.norm(
                closest.pose.position - np.array(pos_orig)
            )
            assert pos_diff < 0.02  # Allow small error in spline approximation
    
    def test_smoothness(self):
        """Test that interpolated trajectory is smooth."""
        # Create trajectory with sharp corners
        trajectory = Trajectory()
        for i in range(5):
            pose = Pose(
                timestamp=float(i),
                position=np.array([i, (-1)**i, 0]),  # Zigzag pattern
                rotation_matrix=np.eye(3)
            )
            trajectory.add_state(TrajectoryState(pose=pose))
        
        config = SplineTrajectoryConfig()
        interpolator = TrajectoryInterpolator(config)
        interpolator.fit(trajectory)
        dense_traj = interpolator.interpolate(rate=50.0)
        
        # Check smoothness via second derivative
        positions = np.array([s.pose.position for s in dense_traj.states])
        first_deriv = np.diff(positions, axis=0)
        second_deriv = np.diff(first_deriv, axis=0)
        
        # Second derivative should be relatively small (smooth curve)
        assert np.all(np.linalg.norm(second_deriv, axis=1) < 2.0)
    
    def test_different_spline_degrees(self):
        """Test interpolation with different spline degrees."""
        trajectory = Trajectory()
        for i in range(6):  # Need enough points for higher degree splines
            pose = Pose(
                timestamp=float(i),
                position=np.array([i, np.sin(i), 0]),
                rotation_matrix=np.eye(3)
            )
            trajectory.add_state(TrajectoryState(pose=pose))
        
        for degree in [1, 3, 5]:
            config = SplineTrajectoryConfig(position_spline_order=degree)
            interpolator = TrajectoryInterpolator(config)
            interpolator.fit(trajectory)
            dense_traj = interpolator.interpolate(rate=20.0)
            
            assert len(dense_traj.states) > 0
            # Higher degree should give smoother results
            if degree > 1:
                positions = np.array([s.pose.position for s in dense_traj.states])
                assert positions.shape[0] > len(trajectory.states)


class TestBezierTrajectory:
    """Test Bezier curve trajectory generation."""
    
    def test_basic_bezier(self):
        """Test basic Bezier curve generation."""
        control_points = [
            np.array([0, 0, 0]),
            np.array([1, 2, 0]),
            np.array([3, 2, 0]),
            np.array([4, 0, 0])
        ]
        
        trajectory = create_bezier_trajectory(
            control_points,
            num_points=50,
            duration=5.0
        )
        
        assert isinstance(trajectory, Trajectory)
        assert len(trajectory.states) == 50
        assert trajectory.states[0].pose.timestamp == 0.0
        assert trajectory.states[-1].pose.timestamp <= 5.0
    
    def test_bezier_endpoints(self):
        """Test that Bezier curve passes through endpoints."""
        control_points = [
            np.array([0, 0, 0]),
            np.array([1, 2, 0]),
            np.array([3, 2, 0]),
            np.array([4, 0, 0])
        ]
        
        trajectory = create_bezier_trajectory(
            control_points,
            num_points=100,
            duration=5.0
        )
        
        # Check start and end points
        np.testing.assert_array_almost_equal(
            trajectory.states[0].pose.position,
            control_points[0]
        )
        np.testing.assert_array_almost_equal(
            trajectory.states[-1].pose.position,
            control_points[-1]
        )
    
    def test_bezier_smoothness(self):
        """Test that Bezier curve is smooth."""
        control_points = [
            np.array([0, 0, 0]),
            np.array([2, 1, 0]),
            np.array([2, -1, 0]),
            np.array([4, 0, 0])
        ]
        
        trajectory = create_bezier_trajectory(
            control_points,
            num_points=100,
            duration=2.0
        )
        
        positions = np.array([s.pose.position for s in trajectory.states])
        
        # Check continuity
        pos_diff = np.diff(positions, axis=0)
        step_sizes = np.linalg.norm(pos_diff, axis=1)
        
        # All steps should be small and relatively uniform
        assert np.all(step_sizes < 0.2)
        assert np.std(step_sizes) < 0.1
    
    def test_bezier_with_different_orders(self):
        """Test Bezier curves with different numbers of control points."""
        # Linear (2 points)
        linear_points = [
            np.array([0, 0, 0]),
            np.array([1, 1, 0])
        ]
        linear_traj = create_bezier_trajectory(linear_points, 10, 1.0)
        assert len(linear_traj.states) == 10
        
        # Quadratic (3 points)
        quad_points = [
            np.array([0, 0, 0]),
            np.array([1, 2, 0]),
            np.array([2, 0, 0])
        ]
        quad_traj = create_bezier_trajectory(quad_points, 20, 2.0)
        assert len(quad_traj.states) == 20
        
        # Cubic (4 points)
        cubic_points = [
            np.array([0, 0, 0]),
            np.array([1, 1, 0]),
            np.array([2, 1, 0]),
            np.array([3, 0, 0])
        ]
        cubic_traj = create_bezier_trajectory(cubic_points, 30, 3.0)
        assert len(cubic_traj.states) == 30


class TestTrajectorySmoothing:
    """Test trajectory smoothing algorithms."""
    
    def test_basic_smoothing(self):
        """Test basic trajectory smoothing."""
        # Create noisy trajectory
        trajectory = Trajectory()
        t = np.linspace(0, 2*np.pi, 50)
        
        np.random.seed(42)  # For reproducibility
        for i, ti in enumerate(t):
            # Circle with noise
            x = np.cos(ti) + 0.1 * np.random.randn()
            y = np.sin(ti) + 0.1 * np.random.randn()
            
            pose = Pose(
                timestamp=ti,
                position=np.array([x, y, 0]),
                rotation_matrix=np.eye(3)
            )
            trajectory.add_state(TrajectoryState(pose=pose))
        
        # Smooth trajectory
        smoothed = smooth_trajectory(
            trajectory,
            window_size=5,
            position_sigma=0.5
        )
        
        assert isinstance(smoothed, Trajectory)
        assert len(smoothed.states) == len(trajectory.states)
    
    def test_smoothing_reduces_noise(self):
        """Test that smoothing reduces trajectory noise."""
        # Create noisy trajectory
        trajectory = Trajectory()
        t = np.linspace(0, 10, 100)
        
        np.random.seed(42)
        for ti in t:
            # Straight line with noise
            pose = Pose(
                timestamp=ti,
                position=np.array([ti, 0.2 * np.random.randn(), 0]),
                rotation_matrix=np.eye(3)
            )
            trajectory.add_state(TrajectoryState(pose=pose))
        
        smoothed = smooth_trajectory(
            trajectory,
            window_size=7,
            position_sigma=1.0
        )
        
        # Check smoothness (reduced variance in derivatives)
        original_positions = np.array([s.pose.position for s in trajectory.states])
        smoothed_positions = np.array([s.pose.position for s in smoothed.states])
        
        original_var = np.var(np.diff(original_positions[:, 1]))  # Y variance
        smoothed_var = np.var(np.diff(smoothed_positions[:, 1]))
        
        assert smoothed_var < original_var
        # Smoothed should be closer to straight line (y ≈ 0)
        assert np.std(smoothed_positions[:, 1]) < np.std(original_positions[:, 1])
    
    def test_smoothing_preserves_shape(self):
        """Test that smoothing preserves overall trajectory shape."""
        # Create trajectory with intentional shape
        trajectory = Trajectory()
        t = np.linspace(0, 4*np.pi, 100)
        
        for ti in t:
            # Sine wave
            pose = Pose(
                timestamp=ti,
                position=np.array([ti, np.sin(ti), 0]),
                rotation_matrix=np.eye(3)
            )
            trajectory.add_state(TrajectoryState(pose=pose))
        
        smoothed = smooth_trajectory(
            trajectory,
            window_size=5,
            position_sigma=1.0  # Moderate smoothing to preserve shape
        )
        
        original_positions = np.array([s.pose.position for s in trajectory.states])
        smoothed_positions = np.array([s.pose.position for s in smoothed.states])
        
        # Check that overall shape is preserved
        # The sine wave peaks should still be roughly at the same locations
        original_peaks = np.where(original_positions[:, 1] > 0.8)[0]
        smoothed_peaks = np.where(smoothed_positions[:, 1] > 0.5)[0]  # More lenient threshold
        
        assert len(smoothed_peaks) > 0  # Should have some peaks preserved
        # Smoothing may reduce number of peaks slightly, but should preserve most
        assert len(smoothed_peaks) >= len(original_peaks) * 0.5
    
    def test_smoothing_edge_cases(self):
        """Test smoothing with edge cases."""
        # Very short trajectory
        short_traj = Trajectory()
        for i in range(3):
            pose = Pose(
                timestamp=float(i),
                position=np.array([i, 0, 0]),
                rotation_matrix=np.eye(3)
            )
            short_traj.add_state(TrajectoryState(pose=pose))
        
        smoothed = smooth_trajectory(short_traj, window_size=5)
        assert len(smoothed.states) == 3
        
        # Single point trajectory
        single_traj = Trajectory()
        single_traj.add_state(TrajectoryState(
            pose=Pose(0.0, np.zeros(3), np.eye(3))
        ))
        
        smoothed_single = smooth_trajectory(single_traj, window_size=3)
        assert len(smoothed_single.states) == 1