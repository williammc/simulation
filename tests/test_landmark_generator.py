"""
Tests for landmark generation.
"""

import pytest
import numpy as np

from src.simulation.landmark_generator import (
    LandmarkGenerator,
    AdaptiveLandmarkGenerator,
    generate_landmarks
)
from src.common.config import EnvironmentConfig
from src.simulation.trajectory_generator import CircleTrajectory, TrajectoryParams


class TestLandmarkGenerator:
    """Test basic landmark generation."""
    
    def test_uniform_generation(self):
        """Test uniform landmark distribution."""
        config = EnvironmentConfig(
            landmark_range=[10.0, 10.0, 3.0],  # Range defines the total size
            num_landmarks=100,
            distribution="uniform",
            min_separation=0.1
        )
        
        generator = LandmarkGenerator(config, seed=42)
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
                dist = np.linalg.norm(positions[i] - positions[j])
                assert dist >= 0.1 - 1e-6  # Allow small numerical error
    
    def test_gaussian_generation(self):
        """Test Gaussian landmark distribution."""
        config = EnvironmentConfig(
            landmark_range=[20.0, 20.0, 5.0],
            num_landmarks=200,
            distribution="gaussian",
            gaussian_mean=[0, 0, 2.5],
            gaussian_std=3.0,
            min_separation=0.05
        )
        
        generator = LandmarkGenerator(config, seed=42)
        map_data = generator.generate()
        
        # Check number of landmarks
        assert len(map_data.landmarks) <= 200  # May be less due to separation constraint
        assert len(map_data.landmarks) > 150  # Should get most of them
        
        # Check that most landmarks are near the mean
        positions = map_data.get_positions()
        mean = np.array([0, 0, 2.5])
        distances_from_mean = np.linalg.norm(positions - mean, axis=1)
        
        # Due to clipping at boundaries, adjust expectation
        # At least 25% should be within 1 std dev (accounting for boundary effects)
        within_1_std = np.sum(distances_from_mean <= 3.0)
        assert within_1_std >= len(positions) * 0.25  # Lowered threshold due to clipping
    
    def test_clustered_generation(self):
        """Test clustered landmark distribution."""
        config = EnvironmentConfig(
            landmark_range=[20.0, 20.0, 5.0],
            num_landmarks=100,
            distribution="clustered",
            num_clusters=5,
            cluster_std=1.0,
            min_separation=0.05
        )
        
        generator = LandmarkGenerator(config, seed=42)
        map_data = generator.generate()
        
        # Check number of landmarks
        assert len(map_data.landmarks) <= 100
        assert len(map_data.landmarks) > 80  # Should get most of them
        
        # Landmarks should be clustered
        # This is hard to test precisely, but we can check
        # that the distribution is not uniform
        positions = map_data.get_positions()
        
        # Compute pairwise distances
        n = len(positions)
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                distances.append(np.linalg.norm(positions[i] - positions[j]))
        
        # In clustered distribution, we should see bimodal distance distribution
        # (small within-cluster, large between-cluster)
        distances = np.array(distances)
        assert np.std(distances) > 2.0  # High variance indicates clustering
    
    def test_min_separation(self):
        """Test minimum separation constraint."""
        config = EnvironmentConfig(
            landmark_range=[4.0, 4.0, 1.0],
            num_landmarks=50,  # Request many landmarks
            distribution="uniform",
            min_separation=1.0  # Large separation relative to space
        )
        
        generator = LandmarkGenerator(config, seed=42)
        map_data = generator.generate()
        
        # With large min_separation, we should get fewer landmarks than requested
        assert len(map_data.landmarks) < 50
        
        # Check all pairs maintain minimum separation
        positions = map_data.get_positions()
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                assert dist >= 1.0 - 1e-6  # Check the actual separation value used


class TestAdaptiveLandmarkGenerator:
    """Test adaptive landmark generation based on trajectory."""
    
    def setup_method(self):
        """Create test trajectory."""
        params = TrajectoryParams(
            duration=10.0,
            rate=10.0  # Low rate for testing
        )
        circle_generator = CircleTrajectory(
            radius=2.0,
            height=1.5,
            angular_velocity=0.5,
            params=params
        )
        self.trajectory = circle_generator.generate()  # Store the generated Trajectory object
    
    def test_adaptive_generation(self):
        """Test that landmarks are concentrated near trajectory."""
        config = EnvironmentConfig(
            landmark_range=[20.0, 20.0, 5.0],
            num_landmarks=100,
            distribution="uniform"
        )
        
        generator = AdaptiveLandmarkGenerator(
            self.trajectory,
            config,
            density_factor=2.0,
            max_distance=5.0,
            seed=42
        )
        map_data = generator.generate()
        
        # Check number of landmarks
        assert len(map_data.landmarks) == 100
        
        # Check that landmarks are near trajectory
        positions = map_data.get_positions()
        traj_positions = np.array([state.pose.position for state in self.trajectory.states])
        
        # For each landmark, find minimum distance to trajectory
        near_count = 0
        for pos in positions:
            distances = np.linalg.norm(traj_positions - pos, axis=1)
            min_dist = distances.min()
            if min_dist <= 5.0:  # Within max_distance
                near_count += 1
        
        # Most landmarks should be near trajectory
        assert near_count >= 80  # At least 80% near trajectory
    
    def test_adaptive_vs_uniform(self):
        """Test that adaptive generation differs from uniform."""
        config = EnvironmentConfig(
            landmark_range=[20.0, 20.0, 5.0],
            num_landmarks=50,
            distribution="uniform"
        )
        
        # Generate uniform landmarks
        uniform_generator = LandmarkGenerator(config, seed=42)
        uniform_map = uniform_generator.generate()
        
        # Generate adaptive landmarks
        adaptive_generator = AdaptiveLandmarkGenerator(
            self.trajectory,
            config,
            density_factor=3.0,
            max_distance=3.0,
            seed=42
        )
        adaptive_map = adaptive_generator.generate()
        
        # Compute average distance to trajectory for both
        traj_positions = np.array([state.pose.position for state in self.trajectory.states])
        
        def avg_distance_to_trajectory(positions):
            distances = []
            for pos in positions:
                min_dist = np.min(np.linalg.norm(traj_positions - pos, axis=1))
                distances.append(min_dist)
            return np.mean(distances)
        
        uniform_positions = uniform_map.get_positions()
        adaptive_positions = adaptive_map.get_positions()
        
        uniform_avg_dist = avg_distance_to_trajectory(uniform_positions)
        adaptive_avg_dist = avg_distance_to_trajectory(adaptive_positions)
        
        # Adaptive should be closer to trajectory
        assert adaptive_avg_dist < uniform_avg_dist * 0.7  # At least 30% closer


class TestFactoryFunction:
    """Test the generate_landmarks factory function."""
    
    def test_generate_landmarks_uniform(self):
        """Test factory function for uniform generation."""
        config = EnvironmentConfig(
            landmark_range=[10.0, 10.0, 3.0],
            num_landmarks=50,
            distribution="uniform"
        )
        
        map_data = generate_landmarks(config, seed=42)
        
        assert len(map_data.landmarks) == 50
        
        # Check bounds
        positions = map_data.get_positions()
        assert np.all(np.abs(positions[:, 0]) <= 5)
        assert np.all(np.abs(positions[:, 1]) <= 5)
        assert np.all(positions[:, 2] >= 0) and np.all(positions[:, 2] <= 3)
    
    def test_generate_landmarks_adaptive(self):
        """Test factory function for adaptive generation."""
        # Create trajectory
        params = TrajectoryParams(
            duration=5.0,
            rate=10.0
        )
        circle_generator = CircleTrajectory(
            radius=1.5,
            height=1.0,
            angular_velocity=1.0,
            params=params
        )
        trajectory = circle_generator.generate()  # Get the generated Trajectory object
        
        config = EnvironmentConfig(
            landmark_range=[10.0, 10.0, 3.0],
            num_landmarks=30,
            distribution="uniform"
        )
        
        map_data = generate_landmarks(
            config,
            trajectory=trajectory,
            adaptive=True,
            seed=42
        )
        
        assert len(map_data.landmarks) == 30
        
        # Check that landmarks are concentrated near trajectory path
        positions = map_data.get_positions()
        traj_positions = np.array([state.pose.position for state in trajectory.states])
        
        # For each landmark, find minimum distance to trajectory
        near_count = 0
        for pos in positions:
            distances = np.linalg.norm(traj_positions - pos, axis=1)
            min_dist = distances.min()
            if min_dist <= 5.0:  # Within reasonable distance
                near_count += 1
        
        assert near_count >= 20  # At least 2/3 should be near trajectory path