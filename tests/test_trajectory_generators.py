"""
Tests for advanced trajectory generators.
"""

import pytest
import numpy as np

from src.simulation.trajectory_generator import (
    Figure8Trajectory, SpiralTrajectory, LineTrajectory,
    TrajectoryParams
)
from src.common.data_structures import Trajectory, TrajectoryState, Pose


class TestFigure8Trajectory:
    """Test Figure-8 trajectory generation."""
    
    def test_basic_generation(self):
        """Test basic Figure-8 trajectory generation."""
        params = TrajectoryParams(duration=10.0, rate=100.0)
        generator = Figure8Trajectory(
            scale_x=3.0,
            scale_y=2.0,
            height=1.5,
            params=params
        )
        
        trajectory = generator.generate()
        
        assert isinstance(trajectory, Trajectory)
        assert len(trajectory.states) == 1000
        assert trajectory.states[0].pose.timestamp == 0.0
        assert trajectory.states[-1].pose.timestamp < 10.0
    
    def test_figure8_shape(self):
        """Test that Figure-8 actually forms the expected shape."""
        params = TrajectoryParams(duration=10.0, rate=100.0)
        generator = Figure8Trajectory(
            scale_x=3.0,
            scale_y=2.0,
            height=1.5,
            params=params
        )
        
        trajectory = generator.generate()
        positions = np.array([s.pose.position for s in trajectory.states])
        
        # Check for x-axis crossings (characteristic of figure-8)
        x_crossings = np.where(np.diff(np.sign(positions[:, 0])))[0]
        assert len(x_crossings) >= 1  # At least one x-axis crossing
        
        # Check that trajectory is bounded
        assert np.all(np.abs(positions[:, 0]) <= 3.5)  # Within scale_x bounds
        assert np.all(np.abs(positions[:, 1]) <= 2.5)  # Within scale_y bounds
    
    def test_figure8_velocity(self):
        """Test velocity computation for Figure-8 trajectory."""
        params = TrajectoryParams(duration=5.0, rate=50.0)
        generator = Figure8Trajectory(
            scale_x=2.0,
            scale_y=1.5,
            height=1.0,
            params=params
        )
        
        trajectory = generator.generate()
        
        # Check that velocities are computed
        velocities = [s.velocity for s in trajectory.states if s.velocity is not None]
        assert len(velocities) > 0
        
        # Check velocity continuity (no sudden jumps)
        vel_array = np.array(velocities)
        vel_diff = np.diff(vel_array, axis=0)
        assert np.all(np.linalg.norm(vel_diff, axis=1) < 1.0)  # No large jumps


class TestSpiralTrajectory:
    """Test spiral trajectory generation."""
    
    def test_basic_generation(self):
        """Test basic spiral trajectory generation."""
        params = TrajectoryParams(duration=5.0, rate=50.0)
        generator = SpiralTrajectory(
            initial_radius=0.5,
            final_radius=3.0,
            initial_height=0.5,
            final_height=3.0,
            params=params
        )
        
        trajectory = generator.generate()
        
        assert isinstance(trajectory, Trajectory)
        assert len(trajectory.states) == 250
        assert trajectory.states[0].pose.timestamp == 0.0
        assert trajectory.states[-1].pose.timestamp < 5.0
    
    def test_spiral_expansion(self):
        """Test that spiral expands properly."""
        params = TrajectoryParams(duration=5.0, rate=50.0)
        generator = SpiralTrajectory(
            initial_radius=0.5,
            final_radius=3.0,
            initial_height=0.5,
            final_height=3.0,
            params=params
        )
        
        trajectory = generator.generate()
        positions = np.array([s.pose.position for s in trajectory.states])
        
        # Check radius expansion
        radii = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
        assert radii[0] < radii[-1]  # Radius increases
        assert radii[0] < 1.0  # Starts small
        assert radii[-1] > 2.5  # Ends large
        
        # Check height increase
        assert positions[0, 2] < positions[-1, 2]  # Height increases
        assert positions[0, 2] < 1.0  # Starts low
        assert positions[-1, 2] > 2.5  # Ends high
    
    def test_spiral_smoothness(self):
        """Test that spiral is smooth without discontinuities."""
        params = TrajectoryParams(duration=3.0, rate=100.0)
        generator = SpiralTrajectory(
            initial_radius=1.0,
            final_radius=2.0,
            initial_height=0.0,
            final_height=2.0,
            params=params
        )
        
        trajectory = generator.generate()
        positions = np.array([s.pose.position for s in trajectory.states])
        
        # Check position continuity
        pos_diff = np.diff(positions, axis=0)
        step_sizes = np.linalg.norm(pos_diff, axis=1)
        
        # All steps should be reasonably small (allow for expanding spiral)
        assert np.all(step_sizes < 0.15)
        # Steps should be relatively uniform (relaxed for expanding spiral)
        assert np.std(step_sizes) < 0.03


class TestLineTrajectory:
    """Test linear trajectory generation."""
    
    def test_basic_generation(self):
        """Test basic line trajectory generation."""
        start = np.array([0, 0, 1])
        end = np.array([10, 5, 2])
        
        params = TrajectoryParams(duration=5.0, rate=20.0)
        generator = LineTrajectory(
            start_position=start,
            end_position=end,
            params=params
        )
        
        trajectory = generator.generate()
        
        assert isinstance(trajectory, Trajectory)
        assert len(trajectory.states) == 100
        assert trajectory.states[0].pose.timestamp == 0.0
        assert trajectory.states[-1].pose.timestamp < 5.0
    
    def test_linearity(self):
        """Test that trajectory is actually linear."""
        start = np.array([0, 0, 1])
        end = np.array([10, 5, 2])
        
        params = TrajectoryParams(duration=5.0, rate=20.0)
        generator = LineTrajectory(
            start_position=start,
            end_position=end,
            params=params
        )
        
        trajectory = generator.generate()
        positions = np.array([s.pose.position for s in trajectory.states])
        
        # Check start and end points (allow small discretization error)
        np.testing.assert_array_almost_equal(positions[0], start)
        np.testing.assert_array_almost_equal(positions[-1], end, decimal=1)
        
        # All points should be on the line
        direction = end - start
        direction = direction / np.linalg.norm(direction)
        
        for i in range(1, len(positions) - 1):
            vec_to_point = positions[i] - start
            # Project onto line direction
            projection = np.dot(vec_to_point, direction) * direction
            # Check perpendicular distance is small
            perpendicular = vec_to_point - projection
            assert np.linalg.norm(perpendicular) < 1e-10
    
    def test_constant_velocity(self):
        """Test that linear trajectory has constant velocity."""
        start = np.array([0, 0, 0])
        end = np.array([10, 0, 0])
        
        params = TrajectoryParams(duration=5.0, rate=20.0)
        generator = LineTrajectory(
            start_position=start,
            end_position=end,
            params=params
        )
        
        trajectory = generator.generate()
        
        # Check velocity consistency
        velocities = [s.velocity for s in trajectory.states if s.velocity is not None]
        if velocities:
            vel_array = np.array(velocities)
            expected_velocity = (end - start) / 5.0  # distance / time
            
            for vel in vel_array:
                np.testing.assert_array_almost_equal(vel, expected_velocity)
    
    def test_zero_length_trajectory(self):
        """Test trajectory when start and end are the same."""
        point = np.array([1, 2, 3])
        
        params = TrajectoryParams(duration=2.0, rate=10.0)
        generator = LineTrajectory(
            start_position=point,
            end_position=point,
            params=params
        )
        
        trajectory = generator.generate()
        positions = np.array([s.pose.position for s in trajectory.states])
        
        # All positions should be the same
        for pos in positions:
            np.testing.assert_array_almost_equal(pos, point)
        
        # Velocity should be zero
        velocities = [s.velocity for s in trajectory.states if s.velocity is not None]
        if velocities:
            for vel in velocities:
                np.testing.assert_array_almost_equal(vel, np.zeros(3))