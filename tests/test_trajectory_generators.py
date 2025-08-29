"""
Tests for advanced trajectory generators.
"""

import pytest
import numpy as np

from src.simulation.trajectory_generator import (
    CircleTrajectory, SpiralTrajectory,
    TrajectoryParams
)
from src.common.data_structures import Trajectory, TrajectoryState, Pose


class TestCircleTrajectory:
    """Test Circle trajectory generation."""
    
    def test_basic_generation(self):
        """Test basic Circle trajectory generation."""
        params = TrajectoryParams(duration=10.0, rate=100.0)
        generator = CircleTrajectory(
            radius=2.0,
            height=1.5,
            params=params
        )
        
        trajectory = generator.generate()
        
        assert isinstance(trajectory, Trajectory)
        assert len(trajectory.states) == 1000
        assert trajectory.states[0].pose.timestamp == 0.0
        assert trajectory.states[-1].pose.timestamp < 10.0
    
    def test_circle_shape(self):
        """Test that Circle actually forms the expected shape."""
        params = TrajectoryParams(duration=10.0, rate=100.0)
        generator = CircleTrajectory(
            radius=2.0,
            height=1.5,
            params=params
        )
        
        trajectory = generator.generate()
        positions = np.array([s.pose.position for s in trajectory.states])
        
        # Check that all points are at the same radius
        radii = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
        assert np.allclose(radii, 2.0, rtol=1e-6)
        
    
    def test_circle_velocity(self):
        """Test velocity computation for Circle trajectory."""
        params = TrajectoryParams(duration=5.0, rate=50.0)
        generator = CircleTrajectory(
            radius=2.0,
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


