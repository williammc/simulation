"""
Backprojection-based landmark generator for SLAM simulation.

Generates realistic landmark distributions by backprojecting pixels from camera frames
to 3D space with controlled depth distributions. Ensures landmarks are visible across
multiple frames for robust SLAM estimation.

Coordinate Frame Conventions:
    - W: World frame (global inertial reference)
    - C: Camera frame (optical center, z-forward, x-right, y-down)
    - All poses represent world-from-camera transforms (W_T_C)

Follows conventions defined in docs/conventions.md
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
import logging

from src.common.data_structures import Map, Landmark, Trajectory, Pose
from src.simulation.camera_model import PinholeCamera

logger = logging.getLogger(__name__)


@dataclass
class BackprojectionConfig:
    """Configuration for backprojection landmark generation."""
    # Number of landmarks per frame
    min_landmarks_per_frame: int = 20
    max_landmarks_per_frame: int = 50
    
    # Depth range for backprojection
    min_depth: float = 2.0
    max_depth: float = 15.0
    mean_depth: float = 8.0
    depth_std: float = 3.0
    
    # Visibility parameters
    min_visibility_count: int = 5  # Minimum frames a landmark should be visible in
    max_viewing_angle: float = 75.0  # Maximum angle (degrees) from camera optical axis
    
    # Image boundaries (avoid edges)
    image_border_margin: int = 20  # Pixels from edge to avoid
    
    # Random seed
    seed: Optional[int] = None


class BackprojectionLandmarkGenerator:
    """Generate landmarks by backprojecting from camera frames."""
    
    def __init__(
        self,
        trajectory: Trajectory,
        camera: PinholeCamera,
        config: Optional[BackprojectionConfig] = None
    ):
        """
        Initialize backprojection landmark generator.
        
        Args:
            trajectory: Camera trajectory
            camera: Camera model for projection/backprojection
            config: Generation configuration
        """
        self.trajectory = trajectory
        self.camera = camera
        self.config = config or BackprojectionConfig()
        
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        
        # Get image dimensions
        self.image_width = camera.calibration.intrinsics.width
        self.image_height = camera.calibration.intrinsics.height
        
        # Valid pixel region (avoiding borders)
        self.min_u = self.config.image_border_margin
        self.max_u = self.image_width - self.config.image_border_margin
        self.min_v = self.config.image_border_margin
        self.max_v = self.image_height - self.config.image_border_margin
    
    def generate(self) -> Tuple[Map, Dict[int, List[Tuple[int, np.ndarray]]]]:
        """
        Generate landmarks by backprojecting from frames.
        
        Returns:
            Tuple of:
            - Map containing generated landmarks
            - Dictionary mapping frame index to list of (landmark_id, pixel) observations
        """
        logger.info(f"Generating landmarks via backprojection from {len(self.trajectory.states)} frames")
        
        all_landmarks = []
        landmark_visibility = {}  # landmark_idx -> list of frame indices where visible
        
        # Step 1: Generate candidate landmarks from each frame
        for frame_idx, state in enumerate(self.trajectory.states):
            self._log_progress(frame_idx, len(self.trajectory.states), "Processing frame")
            
            # Generate candidates for this frame
            candidates = self._generate_frame_candidates(state.pose)
            
            # Add to global candidate list
            for p_W in candidates:
                landmark_idx = len(all_landmarks)
                all_landmarks.append(p_W)
                landmark_visibility[landmark_idx] = [frame_idx]
        
        logger.info(f"Generated {len(all_landmarks)} candidate landmarks")
        
        # Step 2: Check visibility across all frames
        observations = self._compute_visibility_map(all_landmarks, landmark_visibility)
        
        # Step 3: Filter and finalize landmarks
        map_data, final_observations = self._filter_and_finalize_landmarks(
            all_landmarks, landmark_visibility, observations
        )
        
        self._log_generation_summary(map_data, final_observations)
        
        return map_data, final_observations
    
    def _backproject_pixel(self, pixel: np.ndarray, depth: float, pose: Pose) -> Optional[np.ndarray]:
        """
        Backproject pixel to 3D world coordinates using proper coordinate frame notation.
        
        Args:
            pixel: [u, v] pixel coordinates in image frame
            depth: Depth along camera z-axis (meters)
            pose: Camera pose in world frame (W_T_C)
        
        Returns:
            3D position in world coordinates (p_W) or None if invalid
            
        Coordinate frames:
            - C: Camera frame (optical center, z-forward)
            - W: World frame (global reference)
            - Transform: W_T_C (world-from-camera)
        """
        # Convert pixel to normalized camera coordinates
        u, v = pixel
        intrinsics = self.camera.calibration.intrinsics
        x_norm = (u - intrinsics.cx) / intrinsics.fx
        y_norm = (v - intrinsics.cy) / intrinsics.fy
        
        # 3D point in camera frame: p_C = [x*depth, y*depth, depth]
        # Camera convention: z-forward, x-right, y-down
        p_C = np.array([x_norm * depth, y_norm * depth, depth])
        
        # Transform to world frame: p_W = W_R_C @ p_C + W_t_C
        W_R_C = pose.rotation_matrix  # World-from-camera rotation
        W_t_C = pose.position         # World-from-camera translation
        p_W = W_R_C @ p_C + W_t_C
        
        return p_W
    
    def _project_and_check_visibility(
        self, 
        p_W: np.ndarray, 
        pose: Pose
    ) -> Tuple[np.ndarray, bool]:
        """
        Project 3D world point to camera and check visibility.
        
        Args:
            p_W: 3D point in world coordinates
            pose: Camera pose in world frame (W_T_C)
        
        Returns:
            Tuple of (pixel coordinates, is_visible)
            
        Coordinate frames:
            - W: World frame (global reference)
            - C: Camera frame (optical center, z-forward)
            - Transform: C_T_W = inverse(W_T_C)
        """
        # Transform world point to camera frame: p_C = C_R_W @ p_W + C_t_W
        W_R_C = pose.rotation_matrix  # World-from-camera rotation
        W_t_C = pose.position         # World-from-camera translation
        
        # Camera-from-world transform (inverse)
        C_R_W = W_R_C.T               # Rotation inverse
        C_t_W = -C_R_W @ W_t_C        # Translation inverse
        p_C = C_R_W @ p_W + C_t_W
        
        # Check depth validity (must be in front of camera)
        if p_C[2] <= 0.1:  # Small positive threshold for numerical stability
            return np.array([0, 0]), False
        
        # Check viewing angle constraint
        viewing_angle_rad = np.arccos(p_C[2] / np.linalg.norm(p_C))
        if np.degrees(viewing_angle_rad) > self.config.max_viewing_angle:
            return np.array([0, 0]), False
        
        # Project to normalized image coordinates
        x_norm = p_C[0] / p_C[2]
        y_norm = p_C[1] / p_C[2]
        
        # Apply intrinsic calibration to get pixel coordinates
        intrinsics = self.camera.calibration.intrinsics
        u = x_norm * intrinsics.fx + intrinsics.cx
        v = y_norm * intrinsics.fy + intrinsics.cy
        pixel = np.array([u, v])
        
        # Check image bounds
        is_in_bounds = (
            0 <= u < self.image_width and 
            0 <= v < self.image_height
        )
        
        return pixel, is_in_bounds
    
    def _generate_frame_candidates(self, pose: Pose) -> List[np.ndarray]:
        """Generate candidate landmarks for a single frame."""
        candidates = []
        num_landmarks = np.random.randint(
            self.config.min_landmarks_per_frame,
            self.config.max_landmarks_per_frame + 1
        )
        
        for _ in range(num_landmarks):
            # Sample random pixel in valid region
            u = np.random.uniform(self.min_u, self.max_u)
            v = np.random.uniform(self.min_v, self.max_v)
            pixel = np.array([u, v])
            
            # Sample depth with bias toward mean for better visibility
            depth = np.random.normal(self.config.mean_depth, self.config.depth_std)
            depth = np.clip(depth, self.config.min_depth, self.config.max_depth)
            
            # Backproject to world coordinates
            p_W = self._backproject_pixel(pixel, depth, pose)
            if p_W is not None:
                candidates.append(p_W)
        
        return candidates
    
    def _compute_visibility_map(
        self, 
        landmarks: List[np.ndarray], 
        visibility: Dict[int, List[int]]
    ) -> Dict[int, List[Tuple[int, np.ndarray]]]:
        """Compute visibility of all landmarks across all frames."""
        observations = {}
        
        for landmark_idx, p_W in enumerate(landmarks):
            self._log_progress(landmark_idx, len(landmarks), "Checking visibility")
            
            visible_frames = []
            
            for frame_idx, state in enumerate(self.trajectory.states):
                pixel, is_visible = self._project_and_check_visibility(p_W, state.pose)
                
                if is_visible:
                    visible_frames.append(frame_idx)
                    
                    # Record observation
                    if frame_idx not in observations:
                        observations[frame_idx] = []
                    observations[frame_idx].append((landmark_idx, pixel))
            
            visibility[landmark_idx] = visible_frames
        
        return observations
    
    def _filter_and_finalize_landmarks(
        self,
        candidates: List[np.ndarray],
        visibility: Dict[int, List[int]],
        observations: Dict[int, List[Tuple[int, np.ndarray]]]
    ) -> Tuple[Map, Dict[int, List[Tuple[int, np.ndarray]]]]:
        """Filter landmarks by visibility and create final map."""
        map_data = Map(frame_id="world")
        id_mapping = {}  # old_idx -> new_id
        final_observations = {}
        
        # Filter by visibility count
        new_id = 0
        for old_idx, p_W in enumerate(candidates):
            visible_count = len(visibility[old_idx])
            
            if visible_count >= self.config.min_visibility_count:
                landmark = Landmark(id=new_id, position=p_W)
                map_data.add_landmark(landmark)
                id_mapping[old_idx] = new_id
                new_id += 1
        
        # Remap observations to use new IDs
        for frame_idx, frame_obs in observations.items():
            remapped_obs = [
                (id_mapping[old_idx], pixel) 
                for old_idx, pixel in frame_obs 
                if old_idx in id_mapping
            ]
            if remapped_obs:
                final_observations[frame_idx] = remapped_obs
        
        return map_data, final_observations
    
    def _log_progress(self, current: int, total: int, description: str, interval: int = 50) -> None:
        """Log progress at regular intervals."""
        if current % interval == 0:
            logger.debug(f"{description} {current}/{total}")
    
    def _log_generation_summary(self, map_data: Map, observations: Dict) -> None:
        """Log summary statistics of landmark generation."""
        logger.info(f"Final map contains {len(map_data.landmarks)} landmarks")
        logger.info("Visibility statistics:")
        
        obs_counts = [len(observations.get(i, [])) for i in range(len(self.trajectory.states))]
        if obs_counts:
            logger.info(
                f"  Observations per frame: min={min(obs_counts)}, "
                f"max={max(obs_counts)}, mean={np.mean(obs_counts):.1f}"
            )


def generate_landmarks_via_backprojection(
    trajectory: Trajectory,
    camera: PinholeCamera,
    config: Optional[BackprojectionConfig] = None
) -> Tuple[Map, Dict[int, List[Tuple[int, np.ndarray]]]]:
    """
    Factory function to generate landmarks via backprojection.
    
    Args:
        trajectory: Camera trajectory
        camera: Camera model
        config: Generation configuration
    
    Returns:
        Tuple of (Map, observations dictionary)
    """
    generator = BackprojectionLandmarkGenerator(trajectory, camera, config)
    return generator.generate()