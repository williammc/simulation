"""
Tests for landmark generation.
"""

import pytest
import numpy as np

from src.simulation.landmark_generator import (
    LandmarkGenerator,
    BoundingBoxLandmarkGenerator,
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
    
    def test_unsupported_distribution_fallback(self):
        """Test that unsupported distributions fall back to uniform."""
        config = EnvironmentConfig(
            landmark_range=[10.0, 10.0, 3.0],
            num_landmarks=50,
            distribution="gaussian",  # No longer supported
            min_separation=0.1
        )
        
        generator = LandmarkGenerator(config, seed=42)
        map_data = generator.generate()
        
        # Should still generate landmarks using uniform distribution
        assert len(map_data.landmarks) == 50
        
        # Check all landmarks are within bounds (uniform distribution)
        positions = map_data.get_positions()
        assert np.all(positions[:, 0] >= -5) and np.all(positions[:, 0] <= 5)
        assert np.all(positions[:, 1] >= -5) and np.all(positions[:, 1] <= 5)
        assert np.all(positions[:, 2] >= 0) and np.all(positions[:, 2] <= 3)
    
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


class TestBoundingBoxLandmarkGenerator:
    """Test bounding box landmark generation."""
    
    def test_bounding_box_generation(self):
        """Test that landmarks are placed on bounding box faces."""
        config = EnvironmentConfig(
            landmark_range=[10.0, 10.0, 5.0],
            num_landmarks=60,  # 10 per face
            distribution="uniform"
        )
        
        generator = BoundingBoxLandmarkGenerator(
            config,
            trajectory=None,
            seed=42
        )
        map_data = generator.generate()
        
        # Check number of landmarks
        assert len(map_data.landmarks) == 60
        
        # Check that landmarks are on faces
        positions = map_data.get_positions()
        tolerance = 0.01  # Small tolerance for numerical errors
        
        on_faces = 0
        for pos in positions:
            # Check if on any face (at boundary)
            on_x_min = abs(pos[0] - (-5.0)) < tolerance
            on_x_max = abs(pos[0] - 5.0) < tolerance
            on_y_min = abs(pos[1] - (-5.0)) < tolerance
            on_y_max = abs(pos[1] - 5.0) < tolerance
            on_z_min = abs(pos[2] - 0.0) < tolerance
            on_z_max = abs(pos[2] - 5.0) < tolerance
            
            if on_x_min or on_x_max or on_y_min or on_y_max or on_z_min or on_z_max:
                on_faces += 1
        
        # All landmarks should be on faces
        assert on_faces == 60
    
    def test_face_distribution(self):
        """Test that landmarks are distributed across all faces."""
        config = EnvironmentConfig(
            landmark_range=[10.0, 10.0, 5.0],
            num_landmarks=60,  # Should be 10 per face
            distribution="uniform"
        )
        
        generator = BoundingBoxLandmarkGenerator(
            config,
            trajectory=None,
            seed=42
        )
        map_data = generator.generate()
        
        positions = map_data.get_positions()
        tolerance = 0.01
        
        # Count landmarks on each face
        faces = {
            'x_min': 0, 'x_max': 0,
            'y_min': 0, 'y_max': 0,
            'z_min': 0, 'z_max': 0
        }
        
        for pos in positions:
            if abs(pos[0] - (-5.0)) < tolerance:
                faces['x_min'] += 1
            elif abs(pos[0] - 5.0) < tolerance:
                faces['x_max'] += 1
            elif abs(pos[1] - (-5.0)) < tolerance:
                faces['y_min'] += 1
            elif abs(pos[1] - 5.0) < tolerance:
                faces['y_max'] += 1
            elif abs(pos[2] - 0.0) < tolerance:
                faces['z_min'] += 1
            elif abs(pos[2] - 5.0) < tolerance:
                faces['z_max'] += 1
        
        # Each face should have approximately 10 landmarks (60/6)
        for face, count in faces.items():
            assert count >= 8 and count <= 12, f"Face {face} has {count} landmarks, expected ~10"


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
    
    def test_generate_landmarks_bounding_box(self):
        """Test factory function for bounding box generation."""
        config = EnvironmentConfig(
            landmark_range=[10.0, 10.0, 3.0],
            num_landmarks=30,
            distribution="uniform"
        )
        
        map_data = generate_landmarks(
            config,
            bounding_box=True,
            seed=42
        )
        
        assert len(map_data.landmarks) == 30
        
        # Check that landmarks are on faces
        positions = map_data.get_positions()
        tolerance = 0.01
        
        on_faces = 0
        for pos in positions:
            # Check if on any face (at boundary)
            on_x_min = abs(pos[0] - (-5.0)) < tolerance
            on_x_max = abs(pos[0] - 5.0) < tolerance
            on_y_min = abs(pos[1] - (-5.0)) < tolerance
            on_y_max = abs(pos[1] - 5.0) < tolerance
            on_z_min = abs(pos[2] - 0.0) < tolerance
            on_z_max = abs(pos[2] - 3.0) < tolerance
            
            if on_x_min or on_x_max or on_y_min or on_y_max or on_z_min or on_z_max:
                on_faces += 1
        
        # All landmarks should be on faces when using bounding box
        assert on_faces == 30