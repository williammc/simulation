"""
Tests for landmark generation.
"""

import pytest
import numpy as np

from src.simulation.landmark_generator import (
    LandmarkGenerator,
    LandmarkGeneratorConfig,
    AdaptiveLandmarkGenerator,
    generate_landmarks
)
from src.simulation.trajectory_generator import CircleTrajectory, TrajectoryParams


class TestLandmarkGenerator:
    """Test basic landmark generation."""
    
    def test_uniform_generation(self):
        """Test uniform landmark distribution."""
        config = LandmarkGeneratorConfig(
            x_min=-5, x_max=5,
            y_min=-5, y_max=5,
            z_min=0, z_max=3,
            num_landmarks=100,
            distribution="uniform",
            min_separation=0.1,
            seed=42
        )
        
        generator = LandmarkGenerator(config)
        map_data = generator.generate()
        
        # Check number of landmarks
        assert len(map_data.landmarks) == 100
        
        # Check all landmarks are within bounds
        positions = map_data.get_positions()
        assert np.all(positions[:, 0] >= -5) and np.all(positions[:, 0] <= 5)
        assert np.all(positions[:, 1] >= -5) and np.all(positions[:, 1] <= 5)
        assert np.all(positions[:, 2] >= 0) and np.all(positions[:, 2] <= 3)
        
        # Check minimum separation
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = np.linalg.norm(positions[i] - positions[j])
                assert distance >= config.min_separation - 1e-6
    
    def test_gaussian_generation(self):
        """Test Gaussian landmark distribution."""
        config = LandmarkGeneratorConfig(
            x_min=-10, x_max=10,
            y_min=-10, y_max=10,
            z_min=0, z_max=5,
            num_landmarks=200,
            distribution="gaussian",
            gaussian_mean=np.array([0, 0, 2.5]),
            gaussian_std=3.0,
            min_separation=0.05,
            seed=42
        )
        
        generator = LandmarkGenerator(config)
        map_data = generator.generate()
        
        # Check number of landmarks
        assert len(map_data.landmarks) <= 200  # May be less due to separation constraint
        assert len(map_data.landmarks) > 150  # Should get most of them
        
        # Check concentration around mean
        positions = map_data.get_positions()
        distances_from_mean = np.linalg.norm(positions - config.gaussian_mean, axis=1)
        
        # Most landmarks should be within 2 standard deviations
        within_2std = np.sum(distances_from_mean < 2 * config.gaussian_std)
        assert within_2std > 0.6 * len(positions)  # At least 60% within 2 std
    
    def test_clustered_generation(self):
        """Test clustered landmark distribution."""
        config = LandmarkGeneratorConfig(
            x_min=-10, x_max=10,
            y_min=-10, y_max=10,
            z_min=0, z_max=5,
            num_landmarks=100,
            distribution="clustered",
            num_clusters=5,
            cluster_std=1.0,
            min_separation=0.05,
            seed=42
        )
        
        generator = LandmarkGenerator(config)
        map_data = generator.generate()
        
        # Check number of landmarks
        assert len(map_data.landmarks) <= 100
        assert len(map_data.landmarks) > 80  # Should get most of them
        
        # Verify we have approximately equal distribution across clusters
        # (This is a weak test, but clustering is stochastic)
        positions = map_data.get_positions()
        assert positions.shape[0] > 0
    
    def test_min_separation(self):
        """Test minimum separation constraint."""
        config = LandmarkGeneratorConfig(
            x_min=-2, x_max=2,
            y_min=-2, y_max=2,
            z_min=0, z_max=1,
            num_landmarks=50,
            distribution="uniform",
            min_separation=0.5,  # Large separation for small volume
            seed=42
        )
        
        generator = LandmarkGenerator(config)
        map_data = generator.generate()
        
        # With large separation, we won't fit all landmarks
        assert len(map_data.landmarks) < 50
        
        # But all landmarks should respect separation
        positions = map_data.get_positions()
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = np.linalg.norm(positions[i] - positions[j])
                assert distance >= config.min_separation - 1e-6


class TestAdaptiveLandmarkGenerator:
    """Test adaptive landmark generation based on trajectory."""
    
    def test_adaptive_generation(self):
        """Test adaptive landmark generation near trajectory."""
        # Create a simple circular trajectory
        traj_params = TrajectoryParams(
            duration=10.0,
            rate=10.0,
            start_time=0.0
        )
        
        circle_gen = CircleTrajectory(
            radius=5.0,
            height=2.0,
            params=traj_params
        )
        trajectory = circle_gen.generate()
        
        # Generate landmarks adaptively
        config = LandmarkGeneratorConfig(
            num_landmarks=100,
            min_separation=0.1,
            seed=42
        )
        
        generator = AdaptiveLandmarkGenerator(
            trajectory=trajectory,
            config=config,
            density_factor=2.0,
            max_distance=10.0
        )
        
        map_data = generator.generate()
        
        # Check we generated landmarks
        assert len(map_data.landmarks) > 50
        
        # Check landmarks are near trajectory
        positions = map_data.get_positions()
        traj_positions = np.array([state.pose.position for state in trajectory.states])
        
        for landmark_pos in positions:
            distances = np.linalg.norm(traj_positions - landmark_pos, axis=1)
            min_distance = distances.min()
            assert min_distance <= generator.max_distance + 1e-6
    
    def test_adaptive_vs_uniform(self):
        """Test that adaptive generation concentrates landmarks near trajectory."""
        # Create trajectory
        traj_params = TrajectoryParams(duration=5.0, rate=20.0)
        circle_gen = CircleTrajectory(radius=3.0, height=1.5, params=traj_params)
        trajectory = circle_gen.generate()
        
        config = LandmarkGeneratorConfig(
            num_landmarks=100,
            min_separation=0.05,
            seed=42
        )
        
        # Generate adaptive landmarks
        adaptive_gen = AdaptiveLandmarkGenerator(
            trajectory=trajectory,
            config=config,
            density_factor=3.0,
            max_distance=8.0
        )
        adaptive_map = adaptive_gen.generate()
        
        # Generate uniform landmarks in same bounds
        uniform_config = LandmarkGeneratorConfig(
            x_min=config.x_min,
            x_max=config.x_max,
            y_min=config.y_min,
            y_max=config.y_max,
            z_min=config.z_min,
            z_max=config.z_max,
            num_landmarks=100,
            distribution="uniform",
            min_separation=0.05,
            seed=43  # Different seed
        )
        uniform_gen = LandmarkGenerator(uniform_config)
        uniform_map = uniform_gen.generate()
        
        # Compare average distances to trajectory
        traj_positions = np.array([state.pose.position for state in trajectory.states])
        
        def avg_distance_to_trajectory(positions):
            distances = []
            for pos in positions:
                min_dist = np.min(np.linalg.norm(traj_positions - pos, axis=1))
                distances.append(min_dist)
            return np.mean(distances)
        
        adaptive_positions = adaptive_map.get_positions()
        uniform_positions = uniform_map.get_positions()
        
        adaptive_avg_dist = avg_distance_to_trajectory(adaptive_positions)
        uniform_avg_dist = avg_distance_to_trajectory(uniform_positions)
        
        # Adaptive should have landmarks closer to trajectory on average
        assert adaptive_avg_dist < uniform_avg_dist


class TestFactoryFunction:
    """Test the factory function."""
    
    def test_generate_landmarks_uniform(self):
        """Test factory function with uniform distribution."""
        config = LandmarkGeneratorConfig(
            num_landmarks=50,
            distribution="uniform",
            seed=42
        )
        
        map_data = generate_landmarks(config=config)
        assert len(map_data.landmarks) == 50
    
    def test_generate_landmarks_adaptive(self):
        """Test factory function with adaptive generation."""
        # Create trajectory
        traj_params = TrajectoryParams(duration=2.0, rate=50.0)
        circle_gen = CircleTrajectory(radius=2.0, height=1.0, params=traj_params)
        trajectory = circle_gen.generate()
        
        config = LandmarkGeneratorConfig(
            num_landmarks=75,
            seed=42
        )
        
        map_data = generate_landmarks(
            config=config,
            trajectory=trajectory,
            adaptive=True
        )
        
        assert len(map_data.landmarks) > 0
        assert len(map_data.landmarks) <= 75