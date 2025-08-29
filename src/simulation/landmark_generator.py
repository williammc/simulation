"""
Landmark generation for SLAM simulation.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any

from src.common.data_structures import Map, Landmark, Trajectory
from src.common.config import EnvironmentConfig, CameraIntrinsics, CameraExtrinsics


class LandmarkGenerator:
    """Generate 3D landmarks for SLAM simulation."""
    
    def __init__(self, config: EnvironmentConfig, seed: Optional[int] = None):
        """
        Initialize landmark generator.
        
        Args:
            config: Environment configuration including landmark settings (REQUIRED)
            seed: Random seed for reproducibility
        
        Raises:
            TypeError: If config is not provided
            ValueError: If config is not an EnvironmentConfig instance
        """
        if config is None:
            raise TypeError(
                "EnvironmentConfig is required. Backward compatibility has been removed. "
                "Please provide an EnvironmentConfig instance from src.common.config"
            )
        
        if not isinstance(config, EnvironmentConfig):
            raise ValueError(
                f"config must be an EnvironmentConfig instance, got {type(config).__name__}"
            )
        
        self.config = config
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Extract bounding box from landmark_range
        self.x_min = -self.config.landmark_range[0] / 2
        self.x_max = self.config.landmark_range[0] / 2
        self.y_min = -self.config.landmark_range[1] / 2
        self.y_max = self.config.landmark_range[1] / 2
        self.z_min = 0.0
        self.z_max = self.config.landmark_range[2]
    
    def generate(self) -> Map:
        """
        Generate landmarks based on configuration.
        
        Returns:
            Map containing generated landmarks
        """
        # Only support uniform distribution now
        # For other distributions, use BoundingBoxLandmarkGenerator
        if self.config.distribution == "uniform":
            return self.generate_uniform()
        else:
            # Default to uniform for any other distribution type
            print(f"Warning: Distribution '{self.config.distribution}' not supported, using uniform")
            return self.generate_uniform()
    
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
                np.random.uniform(self.x_min, self.x_max),
                np.random.uniform(self.y_min, self.y_max),
                np.random.uniform(self.z_min, self.z_max)
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


class BoundingBoxLandmarkGenerator:
    """
    Generate landmarks on the faces of a bounding box (walls, ceiling, floor).
    This provides a more realistic indoor environment simulation.
    """
    
    def __init__(
        self,
        config: EnvironmentConfig,
        trajectory: Optional[Trajectory] = None,
        camera_intrinsics: Optional[CameraIntrinsics] = None,
        camera_extrinsics: Optional[CameraExtrinsics] = None,
        min_visible_per_frame: int = 20,
        seed: Optional[int] = None
    ):
        """
        Initialize bounding box landmark generator.
        
        Args:
            config: Environment configuration
            trajectory: Optional trajectory to ensure landmark visibility
            camera_intrinsics: Optional camera intrinsics for visibility check
            camera_extrinsics: Optional camera extrinsics for visibility check
            min_visible_per_frame: Minimum landmarks visible per frame
            seed: Random seed for reproducibility
        """
        if config is None:
            raise TypeError(
                "EnvironmentConfig is required. Please provide an EnvironmentConfig instance"
            )
        
        if not isinstance(config, EnvironmentConfig):
            raise ValueError(
                f"config must be an EnvironmentConfig instance, got {type(config).__name__}"
            )
        
        self.config = config
        self.trajectory = trajectory
        self.camera_intrinsics = camera_intrinsics
        self.camera_extrinsics = camera_extrinsics
        self.min_visible_per_frame = min_visible_per_frame
        
        if seed is not None:
            np.random.seed(seed)
        
        # Define bounding box dimensions
        self.x_min = -self.config.landmark_range[0] / 2
        self.x_max = self.config.landmark_range[0] / 2
        self.y_min = -self.config.landmark_range[1] / 2
        self.y_max = self.config.landmark_range[1] / 2
        self.z_min = 0.0
        self.z_max = self.config.landmark_range[2]
    
    def generate(self) -> Map:
        """
        Generate landmarks on bounding box faces.
        
        Returns:
            Map with landmarks on walls, ceiling, and floor
        """
        map_data = Map(frame_id="world")
        
        # If trajectory is provided, use adaptive density based on visibility
        if self.trajectory is not None and self.camera_intrinsics is not None:
            map_data = self._generate_with_visibility_check()
        else:
            # Simple uniform distribution on faces
            map_data = self._generate_uniform_on_faces()
        
        return map_data
    
    def _generate_uniform_on_faces(self) -> Map:
        """
        Generate landmarks uniformly distributed on bounding box faces.
        
        Returns:
            Map with uniformly distributed landmarks
        """
        map_data = Map(frame_id="world")
        
        # Calculate landmarks per face (6 faces total)
        total_landmarks = self.config.num_landmarks
        landmarks_per_face = total_landmarks // 6
        remaining = total_landmarks % 6
        
        landmark_id = 0
        
        # Face definitions: (axis, value, varying_axes)
        faces = [
            ('x', self.x_min, ('y', 'z')),  # Left wall
            ('x', self.x_max, ('y', 'z')),  # Right wall
            ('y', self.y_min, ('x', 'z')),  # Front wall
            ('y', self.y_max, ('x', 'z')),  # Back wall
            ('z', self.z_min, ('x', 'y')),  # Floor
            ('z', self.z_max, ('x', 'y')),  # Ceiling
        ]
        
        for face_idx, (fixed_axis, fixed_value, varying_axes) in enumerate(faces):
            # Add extra landmark to first faces if there's a remainder
            num_on_face = landmarks_per_face + (1 if face_idx < remaining else 0)
            
            # Generate grid dimensions for this face
            if num_on_face > 0:
                landmarks = self._generate_face_landmarks(
                    fixed_axis, fixed_value, varying_axes, num_on_face, landmark_id
                )
                
                for landmark in landmarks:
                    map_data.add_landmark(landmark)
                    landmark_id += 1
        
        return map_data
    
    def _generate_with_visibility_check(self) -> Map:
        """
        Generate landmarks ensuring minimum visibility per frame.
        
        Returns:
            Map with landmarks ensuring minimum visibility
        """
        # For now, use uniform distribution
        # TODO: Implement adaptive density based on trajectory visibility
        return self._generate_uniform_on_faces()
    
    def _generate_face_landmarks(
        self,
        fixed_axis: str,
        fixed_value: float,
        varying_axes: Tuple[str, str],
        num_landmarks: int,
        start_id: int
    ) -> List[Landmark]:
        """
        Generate landmarks on a single face of the bounding box.
        
        Args:
            fixed_axis: The axis that is fixed for this face ('x', 'y', or 'z')
            fixed_value: The value of the fixed axis
            varying_axes: The two axes that vary on this face
            num_landmarks: Number of landmarks to place on this face
            start_id: Starting ID for landmarks
        
        Returns:
            List of landmarks on the face
        """
        
        # Get ranges for varying axes
        ranges = {}
        for axis in varying_axes:
            if axis == 'x':
                ranges[axis] = (self.x_min, self.x_max)
            elif axis == 'y':
                ranges[axis] = (self.y_min, self.y_max)
            elif axis == 'z':
                ranges[axis] = (self.z_min, self.z_max)
        
        # Calculate grid dimensions
        aspect_ratio = (ranges[varying_axes[0]][1] - ranges[varying_axes[0]][0]) / \
                      (ranges[varying_axes[1]][1] - ranges[varying_axes[1]][0])
        
        # Solve for grid dimensions: n_u * n_v ≈ num_landmarks
        # where n_u / n_v ≈ aspect_ratio
        n_v = int(np.sqrt(num_landmarks / aspect_ratio))
        n_u = int(num_landmarks / n_v) if n_v > 0 else 1
        
        # Ensure we have at least the required number
        while n_u * n_v < num_landmarks:
            if aspect_ratio > 1:
                n_u += 1
            else:
                n_v += 1
        
        # Generate grid points with some randomization
        grid_landmarks = []
        for i in range(n_u):
            for j in range(n_v):
                if len(grid_landmarks) >= num_landmarks:
                    break
                
                # Calculate base grid position
                u = (i + 0.5) / n_u
                v = (j + 0.5) / n_v
                
                # Add small random perturbation (±10% of grid cell)
                u += np.random.uniform(-0.1/n_u, 0.1/n_u)
                v += np.random.uniform(-0.1/n_v, 0.1/n_v)
                
                # Clamp to [0, 1]
                u = np.clip(u, 0.05, 0.95)  # Keep away from edges
                v = np.clip(v, 0.05, 0.95)
                
                # Convert to actual coordinates
                coord_u = ranges[varying_axes[0]][0] + u * (ranges[varying_axes[0]][1] - ranges[varying_axes[0]][0])
                coord_v = ranges[varying_axes[1]][0] + v * (ranges[varying_axes[1]][1] - ranges[varying_axes[1]][0])
                
                # Build position vector
                position = np.zeros(3)
                if fixed_axis == 'x':
                    position[0] = fixed_value
                    position[1] = coord_u if varying_axes[0] == 'y' else coord_v
                    position[2] = coord_v if varying_axes[1] == 'z' else coord_u
                elif fixed_axis == 'y':
                    position[0] = coord_u if varying_axes[0] == 'x' else coord_v
                    position[1] = fixed_value
                    position[2] = coord_v if varying_axes[1] == 'z' else coord_u
                elif fixed_axis == 'z':
                    position[0] = coord_u if varying_axes[0] == 'x' else coord_v
                    position[1] = coord_v if varying_axes[1] == 'y' else coord_u
                    position[2] = fixed_value
                
                landmark = Landmark(
                    id=start_id + len(grid_landmarks),
                    position=position
                )
                grid_landmarks.append(landmark)
            
            if len(grid_landmarks) >= num_landmarks:
                break
        
        return grid_landmarks[:num_landmarks]




def generate_landmarks(
    config: EnvironmentConfig,
    trajectory: Optional[Trajectory] = None,
    bounding_box: bool = False,
    camera_intrinsics: Optional[CameraIntrinsics] = None,
    camera_extrinsics: Optional[CameraExtrinsics] = None,
    min_visible_per_frame: int = 20,
    seed: Optional[int] = None
) -> Map:
    """
    Factory function to generate landmarks.
    
    Args:
        config: Environment configuration (REQUIRED)
        trajectory: Optional reference trajectory for bounding_box generation
        bounding_box: Whether to use bounding box generation (walls/ceiling/floor)
        camera_intrinsics: Optional camera intrinsics for visibility check (for bounding_box)
        camera_extrinsics: Optional camera extrinsics for visibility check (for bounding_box)
        min_visible_per_frame: Minimum landmarks visible per frame (for bounding_box)
        seed: Random seed for reproducibility
    
    Returns:
        Map containing generated landmarks
    
    Raises:
        TypeError: If config is not provided
    """
    if config is None:
        raise TypeError(
            "EnvironmentConfig is required. Please provide an EnvironmentConfig instance"
        )
    
    if bounding_box:
        generator = BoundingBoxLandmarkGenerator(
            config, 
            trajectory=trajectory,
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=camera_extrinsics,
            min_visible_per_frame=min_visible_per_frame,
            seed=seed
        )
    else:
        # Use simple uniform generator by default
        generator = LandmarkGenerator(config, seed=seed)
    
    return generator.generate()