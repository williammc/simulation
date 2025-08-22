"""
Keyframe selection strategies for SLAM optimization.

Provides different strategies for selecting which frames should be
treated as keyframes in the SLAM pipeline.
"""

import numpy as np
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
import logging

from src.common.config import KeyframeSelectionConfig, KeyframeSelectionStrategy
from src.common.data_structures import Pose, CameraFrame
from src.utils.math_utils import so3_log

logger = logging.getLogger(__name__)


class KeyframeSelector(ABC):
    """Abstract base class for keyframe selection strategies."""
    
    def __init__(self, config: KeyframeSelectionConfig):
        """
        Initialize keyframe selector.
        
        Args:
            config: Keyframe selection configuration
        """
        self.config = config
        self.last_keyframe_pose: Optional[Pose] = None
        self.last_keyframe_time: float = -float('inf')
        self.frames_since_keyframe: int = 0
        self.keyframe_count: int = 0
    
    @abstractmethod
    def should_select_keyframe(self, pose: Pose, timestamp: float) -> bool:
        """
        Determine if the current frame should be a keyframe.
        
        Args:
            pose: Current camera/robot pose
            timestamp: Current timestamp
            
        Returns:
            True if frame should be selected as keyframe
        """
        pass
    
    def reset(self):
        """Reset selector state."""
        self.last_keyframe_pose = None
        self.last_keyframe_time = -float('inf')
        self.frames_since_keyframe = 0
        self.keyframe_count = 0
    
    def mark_keyframe(self, pose: Pose, timestamp: float) -> int:
        """
        Mark current frame as keyframe and update state.
        
        Args:
            pose: Keyframe pose
            timestamp: Keyframe timestamp
            
        Returns:
            Keyframe ID
        """
        self.last_keyframe_pose = pose
        self.last_keyframe_time = timestamp
        self.frames_since_keyframe = 0
        keyframe_id = self.keyframe_count
        self.keyframe_count += 1
        return keyframe_id
    
    def compute_motion(self, pose1: Pose, pose2: Pose) -> Tuple[float, float]:
        """
        Compute translation and rotation between two poses.
        
        Args:
            pose1: First pose
            pose2: Second pose
            
        Returns:
            Tuple of (translation_distance, rotation_angle)
        """
        # Translation distance
        translation = np.linalg.norm(pose2.position - pose1.position)
        
        # Rotation angle
        R_relative = pose1.rotation_matrix.T @ pose2.rotation_matrix
        axis_angle = so3_log(R_relative)
        rotation = np.linalg.norm(axis_angle)
        
        return translation, rotation


class FixedIntervalSelector(KeyframeSelector):
    """Select keyframes at fixed intervals."""
    
    def should_select_keyframe(self, pose: Pose, timestamp: float) -> bool:
        """Select every N-th frame with minimum time gap."""
        self.frames_since_keyframe += 1
        
        # Check time gap constraint
        time_gap = timestamp - self.last_keyframe_time
        if time_gap < self.config.min_time_gap:
            return False
        
        # Check interval constraint
        if self.frames_since_keyframe >= self.config.fixed_interval:
            return True
        
        # First frame is always a keyframe
        if self.last_keyframe_pose is None:
            return True
        
        return False


class MotionBasedSelector(KeyframeSelector):
    """Select keyframes based on motion thresholds."""
    
    def should_select_keyframe(self, pose: Pose, timestamp: float) -> bool:
        """Select when motion exceeds thresholds."""
        # First frame is always a keyframe
        if self.last_keyframe_pose is None:
            return True
        
        self.frames_since_keyframe += 1
        
        # Check time gap constraint
        time_gap = timestamp - self.last_keyframe_time
        if time_gap < self.config.min_time_gap:
            return False
        
        # Compute motion since last keyframe
        translation, rotation = self.compute_motion(self.last_keyframe_pose, pose)
        
        # Check motion thresholds
        if translation >= self.config.translation_threshold:
            logger.debug(f"Translation threshold exceeded: {translation:.3f}m")
            return True
        
        if rotation >= self.config.rotation_threshold:
            logger.debug(f"Rotation threshold exceeded: {rotation:.3f}rad")
            return True
        
        return False


class HybridSelector(KeyframeSelector):
    """Combine fixed interval and motion-based selection."""
    
    def should_select_keyframe(self, pose: Pose, timestamp: float) -> bool:
        """Select based on both interval and motion criteria."""
        # First frame is always a keyframe
        if self.last_keyframe_pose is None:
            return True
        
        self.frames_since_keyframe += 1
        
        # Check time gap constraint
        time_gap = timestamp - self.last_keyframe_time
        if time_gap < self.config.min_time_gap:
            return False
        
        # Force keyframe at maximum interval
        if self.frames_since_keyframe >= self.config.max_interval:
            logger.debug(f"Maximum interval reached: {self.frames_since_keyframe}")
            return True
        
        # Check motion thresholds if configured
        if self.config.force_keyframe_on_motion:
            translation, rotation = self.compute_motion(self.last_keyframe_pose, pose)
            
            if translation >= self.config.translation_threshold:
                logger.debug(f"Motion-triggered keyframe: translation {translation:.3f}m")
                return True
            
            if rotation >= self.config.rotation_threshold:
                logger.debug(f"Motion-triggered keyframe: rotation {rotation:.3f}rad")
                return True
        
        # Regular interval check
        if self.frames_since_keyframe >= self.config.fixed_interval:
            return True
        
        return False


def create_keyframe_selector(config: KeyframeSelectionConfig) -> KeyframeSelector:
    """
    Factory function to create appropriate keyframe selector.
    
    Args:
        config: Keyframe selection configuration
        
    Returns:
        Keyframe selector instance
    """
    if config.strategy == KeyframeSelectionStrategy.FIXED_INTERVAL:
        return FixedIntervalSelector(config)
    elif config.strategy == KeyframeSelectionStrategy.MOTION_BASED:
        return MotionBasedSelector(config)
    elif config.strategy == KeyframeSelectionStrategy.HYBRID:
        return HybridSelector(config)
    else:
        raise ValueError(f"Unknown keyframe selection strategy: {config.strategy}")


def select_keyframes_from_trajectory(
    poses: List[Pose],
    timestamps: List[float],
    config: KeyframeSelectionConfig
) -> List[Tuple[int, int]]:
    """
    Select keyframes from a trajectory.
    
    Args:
        poses: List of poses
        timestamps: Corresponding timestamps
        config: Keyframe selection configuration
        
    Returns:
        List of (frame_index, keyframe_id) tuples
    """
    selector = create_keyframe_selector(config)
    keyframes = []
    
    for i, (pose, timestamp) in enumerate(zip(poses, timestamps)):
        if selector.should_select_keyframe(pose, timestamp):
            keyframe_id = selector.mark_keyframe(pose, timestamp)
            keyframes.append((i, keyframe_id))
            logger.debug(f"Selected keyframe {keyframe_id} at frame {i} (t={timestamp:.3f})")
    
    logger.info(f"Selected {len(keyframes)} keyframes from {len(poses)} frames")
    return keyframes


def mark_keyframes_in_camera_data(
    frames: List[CameraFrame],
    poses: List[Pose],
    config: KeyframeSelectionConfig
) -> None:
    """
    Mark keyframes in camera data based on selection strategy.
    
    Modifies frames in-place by setting is_keyframe and keyframe_id.
    
    Args:
        frames: Camera frames to mark
        poses: Corresponding poses for each frame
        config: Keyframe selection configuration
    """
    if len(frames) != len(poses):
        raise ValueError("Number of frames and poses must match")
    
    # Extract timestamps
    timestamps = [frame.timestamp for frame in frames]
    
    # Select keyframes
    keyframe_indices = select_keyframes_from_trajectory(poses, timestamps, config)
    
    # Mark keyframes in camera data
    for frame_idx, keyframe_id in keyframe_indices:
        frames[frame_idx].is_keyframe = True
        frames[frame_idx].keyframe_id = keyframe_id
    
    logger.info(f"Marked {len(keyframe_indices)} keyframes in {len(frames)} camera frames")