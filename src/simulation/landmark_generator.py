"""
Landmark generation for SLAM simulation.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

from src.common.data_structures import Map, Landmark, Trajectory


@dataclass
class LandmarkGeneratorConfig:
    """Configuration for landmark generation."""
    # Bounding box for landmark generation
    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0
    z_min: float = 0.0
    z_max: float = 5.0
    
    # Number of landmarks
    num_landmarks: int = 500
    
    # Distribution type
    distribution: str = "uniform"  # "uniform", "gaussian", "clustered"
    
    # For Gaussian distribution
    gaussian_mean: np.ndarray = None
    gaussian_std: float = 5.0
    
    # For clustered distribution
    num_clusters: int = 5
    cluster_std: float = 1.0
    
    # Minimum distance between landmarks
    min_separation: float = 0.1
    
    # Random seed
    seed: Optional[int] = None


class LandmarkGenerator:
    """Generate 3D landmarks for SLAM simulation."""
    
    def __init__(self, config: Optional[LandmarkGeneratorConfig] = None):
        """
        Initialize landmark generator.
        
        Args:
            config: Generation configuration
        """
        self.config = config or LandmarkGeneratorConfig()
        
        # Set random seed for reproducibility
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        
        # Initialize Gaussian mean if not provided
        if self.config.gaussian_mean is None:
            self.config.gaussian_mean = np.array([
                (self.config.x_min + self.config.x_max) / 2,
                (self.config.y_min + self.config.y_max) / 2,
                (self.config.z_min + self.config.z_max) / 2
            ])
    
    def generate(self) -> Map:
        """
        Generate landmarks based on configuration.
        
        Returns:
            Map containing generated landmarks
        """
        if self.config.distribution == "uniform":
            return self.generate_uniform()
        elif self.config.distribution == "gaussian":
            return self.generate_gaussian()
        elif self.config.distribution == "clustered":
            return self.generate_clustered()
        else:
            raise ValueError(f"Unknown distribution: {self.config.distribution}")
    
    def generate_uniform(self) -> Map:
        """
        Generate uniformly distributed landmarks.
        
        Returns:
            Map with uniformly distributed landmarks
        """
        map_data = Map(frame_id="world")
        
        landmarks_added = 0
        max_attempts = self.config.num_landmarks * 10
        attempts = 0
        
        existing_positions = []
        
        while landmarks_added < self.config.num_landmarks and attempts < max_attempts:
            # Generate random position
            position = np.array([
                np.random.uniform(self.config.x_min, self.config.x_max),
                np.random.uniform(self.config.y_min, self.config.y_max),
                np.random.uniform(self.config.z_min, self.config.z_max)
            ])
            
            # Check minimum separation
            if self._check_separation(position, existing_positions):
                landmark = Landmark(
                    id=landmarks_added,
                    position=position
                )
                map_data.add_landmark(landmark)
                existing_positions.append(position)
                landmarks_added += 1
            
            attempts += 1
        
        if landmarks_added < self.config.num_landmarks:
            print(f"Warning: Only generated {landmarks_added}/{self.config.num_landmarks} landmarks")
        
        return map_data
    
    def generate_gaussian(self) -> Map:
        """
        Generate Gaussian distributed landmarks.
        
        Returns:
            Map with Gaussian distributed landmarks
        """
        map_data = Map(frame_id="world")
        
        landmarks_added = 0
        max_attempts = self.config.num_landmarks * 10
        attempts = 0
        
        existing_positions = []
        
        while landmarks_added < self.config.num_landmarks and attempts < max_attempts:
            # Generate position from Gaussian distribution
            position = np.random.normal(
                loc=self.config.gaussian_mean,
                scale=self.config.gaussian_std,
                size=3
            )
            
            # Clip to bounding box
            position[0] = np.clip(position[0], self.config.x_min, self.config.x_max)
            position[1] = np.clip(position[1], self.config.y_min, self.config.y_max)
            position[2] = np.clip(position[2], self.config.z_min, self.config.z_max)
            
            # Check minimum separation
            if self._check_separation(position, existing_positions):
                landmark = Landmark(
                    id=landmarks_added,
                    position=position
                )
                map_data.add_landmark(landmark)
                existing_positions.append(position)
                landmarks_added += 1
            
            attempts += 1
        
        return map_data
    
    def generate_clustered(self) -> Map:
        """
        Generate clustered landmarks.
        
        Returns:
            Map with clustered landmarks
        """
        map_data = Map(frame_id="world")
        
        # Generate cluster centers
        cluster_centers = []
        for _ in range(self.config.num_clusters):
            center = np.array([
                np.random.uniform(self.config.x_min, self.config.x_max),
                np.random.uniform(self.config.y_min, self.config.y_max),
                np.random.uniform(self.config.z_min, self.config.z_max)
            ])
            cluster_centers.append(center)
        
        # Generate landmarks around clusters
        landmarks_per_cluster = self.config.num_landmarks // self.config.num_clusters
        remaining = self.config.num_landmarks % self.config.num_clusters
        
        landmark_id = 0
        existing_positions = []
        
        for i, center in enumerate(cluster_centers):
            # Add extra landmark to first clusters if there's a remainder
            num_in_cluster = landmarks_per_cluster + (1 if i < remaining else 0)
            
            landmarks_added = 0
            max_attempts = num_in_cluster * 10
            attempts = 0
            
            while landmarks_added < num_in_cluster and attempts < max_attempts:
                # Generate position around cluster center
                position = np.random.normal(
                    loc=center,
                    scale=self.config.cluster_std,
                    size=3
                )
                
                # Clip to bounding box
                position[0] = np.clip(position[0], self.config.x_min, self.config.x_max)
                position[1] = np.clip(position[1], self.config.y_min, self.config.y_max)
                position[2] = np.clip(position[2], self.config.z_min, self.config.z_max)
                
                # Check minimum separation
                if self._check_separation(position, existing_positions):
                    landmark = Landmark(
                        id=landmark_id,
                        position=position
                    )
                    map_data.add_landmark(landmark)
                    existing_positions.append(position)
                    landmark_id += 1
                    landmarks_added += 1
                
                attempts += 1
        
        return map_data
    
    def _check_separation(self, position: np.ndarray, existing: List[np.ndarray]) -> bool:
        """
        Check if position maintains minimum separation from existing landmarks.
        
        Args:
            position: New position to check
            existing: List of existing positions
        
        Returns:
            True if separation is maintained
        """
        if self.config.min_separation <= 0 or not existing:
            return True
        
        for existing_pos in existing:
            distance = np.linalg.norm(position - existing_pos)
            if distance < self.config.min_separation:
                return False
        
        return True


class AdaptiveLandmarkGenerator:
    """
    Generate landmarks adaptively based on trajectory.
    Places more landmarks near the trajectory path.
    """
    
    def __init__(
        self,
        trajectory: Trajectory,
        config: Optional[LandmarkGeneratorConfig] = None,
        density_factor: float = 2.0,
        max_distance: float = 20.0
    ):
        """
        Initialize adaptive landmark generator.
        
        Args:
            trajectory: Reference trajectory for adaptive generation
            config: Base configuration
            density_factor: How much denser landmarks should be near trajectory
            max_distance: Maximum distance from trajectory for landmark placement
        """
        self.trajectory = trajectory
        self.config = config or LandmarkGeneratorConfig()
        self.density_factor = density_factor
        self.max_distance = max_distance
        
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
    
    def generate(self) -> Map:
        """
        Generate landmarks adaptively based on trajectory.
        
        Returns:
            Map with landmarks concentrated near trajectory
        """
        map_data = Map(frame_id="world")
        
        # Extract trajectory positions
        traj_positions = np.array([state.pose.position for state in self.trajectory.states])
        
        # Compute trajectory bounding box with margin
        margin = self.max_distance
        x_min = traj_positions[:, 0].min() - margin
        x_max = traj_positions[:, 0].max() + margin
        y_min = traj_positions[:, 1].min() - margin
        y_max = traj_positions[:, 1].max() + margin
        z_min = max(0, traj_positions[:, 2].min() - margin)
        z_max = traj_positions[:, 2].max() + margin
        
        # Update config bounds
        self.config.x_min = x_min
        self.config.x_max = x_max
        self.config.y_min = y_min
        self.config.y_max = y_max
        self.config.z_min = z_min
        self.config.z_max = z_max
        
        landmarks_added = 0
        existing_positions = []
        max_attempts = self.config.num_landmarks * 20
        attempts = 0
        
        while landmarks_added < self.config.num_landmarks and attempts < max_attempts:
            # Generate candidate position
            position = np.array([
                np.random.uniform(x_min, x_max),
                np.random.uniform(y_min, y_max),
                np.random.uniform(z_min, z_max)
            ])
            
            # Compute distance to nearest trajectory point
            distances = np.linalg.norm(traj_positions - position, axis=1)
            min_distance = distances.min()
            
            # Accept with probability based on distance
            # Closer to trajectory = higher probability
            if min_distance <= self.max_distance:
                # Probability decreases with distance
                prob = np.exp(-min_distance / (self.max_distance / self.density_factor))
                
                if np.random.random() < prob:
                    # Check minimum separation
                    if self._check_separation(position, existing_positions):
                        landmark = Landmark(
                            id=landmarks_added,
                            position=position
                        )
                        map_data.add_landmark(landmark)
                        existing_positions.append(position)
                        landmarks_added += 1
            
            attempts += 1
        
        if landmarks_added < self.config.num_landmarks:
            print(f"Warning: Only generated {landmarks_added}/{self.config.num_landmarks} landmarks")
        
        return map_data
    
    def _check_separation(self, position: np.ndarray, existing: List[np.ndarray]) -> bool:
        """Check minimum separation."""
        if self.config.min_separation <= 0 or not existing:
            return True
        
        for existing_pos in existing:
            if np.linalg.norm(position - existing_pos) < self.config.min_separation:
                return False
        
        return True


def generate_landmarks(
    config: Optional[LandmarkGeneratorConfig] = None,
    trajectory: Optional[Trajectory] = None,
    adaptive: bool = False
) -> Map:
    """
    Factory function to generate landmarks.
    
    Args:
        config: Generation configuration
        trajectory: Reference trajectory (for adaptive generation)
        adaptive: Whether to use adaptive generation
    
    Returns:
        Map containing generated landmarks
    """
    if adaptive and trajectory is not None:
        generator = AdaptiveLandmarkGenerator(trajectory, config)
    else:
        generator = LandmarkGenerator(config)
    
    return generator.generate()