"""
Tests for keyframe selection strategies.
"""

import numpy as np
import pytest
from typing import List

from src.common.config import KeyframeSelectionConfig, KeyframeSelectionStrategy
from src.common.data_structures import Pose, CameraFrame
from src.simulation.keyframe_selector import (
    FixedIntervalSelector,
    MotionBasedSelector,
    HybridSelector,
    create_keyframe_selector,
    select_keyframes_from_trajectory,
    mark_keyframes_in_camera_data
)


def create_test_poses(n: int = 20) -> List[Pose]:
    """Create a sequence of test poses along a line."""
    poses = []
    for i in range(n):
        position = np.array([i * 0.1, 0, 0])  # Move 0.1m each step
        rotation = np.eye(3)  # No rotation
        pose = Pose(
            timestamp=i * 0.1,
            position=position,
            rotation_matrix=rotation
        )
        poses.append(pose)
    return poses


def create_rotating_poses(n: int = 20) -> List[Pose]:
    """Create poses with rotation."""
    poses = []
    for i in range(n):
        position = np.array([0, 0, 0])  # No translation
        angle = i * 0.2  # Rotate 0.2 rad each step
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        pose = Pose(
            timestamp=i * 0.1,
            position=position,
            rotation_matrix=rotation
        )
        poses.append(pose)
    return poses


class TestFixedIntervalSelector:
    """Test fixed interval keyframe selection."""
    
    def test_fixed_interval_selection(self):
        """Test selecting every N-th frame."""
        config = KeyframeSelectionConfig(
            strategy=KeyframeSelectionStrategy.FIXED_INTERVAL,
            fixed_interval=5,
            min_time_gap=0.001  # Very small but > 0
        )
        
        selector = FixedIntervalSelector(config)
        poses = create_test_poses(20)
        
        selected = []
        for i, pose in enumerate(poses):
            if selector.should_select_keyframe(pose, i * 0.1):
                selector.mark_keyframe(pose, i * 0.1)
                selected.append(i)
        
        # Should select frames 0, 5, 10, 15
        assert selected == [0, 5, 10, 15]
    
    def test_min_time_gap(self):
        """Test minimum time gap constraint."""
        config = KeyframeSelectionConfig(
            strategy=KeyframeSelectionStrategy.FIXED_INTERVAL,
            fixed_interval=3,
            min_time_gap=0.35  # Skip some frames due to time gap
        )
        
        selector = FixedIntervalSelector(config)
        poses = create_test_poses(20)
        
        selected = []
        for i, pose in enumerate(poses):
            if selector.should_select_keyframe(pose, i * 0.1):
                selector.mark_keyframe(pose, i * 0.1)
                selected.append(i)
        
        # Check that time gaps are respected
        for i in range(1, len(selected)):
            time_gap = selected[i] * 0.1 - selected[i-1] * 0.1
            assert time_gap >= 0.35


class TestMotionBasedSelector:
    """Test motion-based keyframe selection."""
    
    def test_translation_threshold(self):
        """Test selection based on translation."""
        config = KeyframeSelectionConfig(
            strategy=KeyframeSelectionStrategy.MOTION_BASED,
            translation_threshold=0.4,  # Select after 0.4m translation
            rotation_threshold=999.0,  # Very high, won't trigger
            min_time_gap=0.001  # Very small but > 0
        )
        
        selector = MotionBasedSelector(config)
        poses = create_test_poses(20)
        
        selected = []
        for i, pose in enumerate(poses):
            if selector.should_select_keyframe(pose, i * 0.1):
                selector.mark_keyframe(pose, i * 0.1)
                selected.append(i)
        
        # Should select approximately every 4 frames (0.4m / 0.1m per frame)
        # First frame is always selected
        assert selected[0] == 0
        for i in range(1, len(selected)):
            # Check translation between keyframes
            translation = np.linalg.norm(
                poses[selected[i]].position - poses[selected[i-1]].position
            )
            assert translation >= 0.4 or abs(translation - 0.4) < 0.01
    
    def test_rotation_threshold(self):
        """Test selection based on rotation."""
        config = KeyframeSelectionConfig(
            strategy=KeyframeSelectionStrategy.MOTION_BASED,
            translation_threshold=999.0,  # Very high, won't trigger
            rotation_threshold=0.5,  # Select after 0.5 rad rotation
            min_time_gap=0.001  # Very small but > 0
        )
        
        selector = MotionBasedSelector(config)
        poses = create_rotating_poses(20)
        
        selected = []
        for i, pose in enumerate(poses):
            if selector.should_select_keyframe(pose, i * 0.1):
                selector.mark_keyframe(pose, i * 0.1)
                selected.append(i)
        
        # Should select based on rotation threshold
        assert selected[0] == 0
        assert len(selected) > 1  # Should have multiple keyframes


class TestHybridSelector:
    """Test hybrid keyframe selection."""
    
    def test_hybrid_selection(self):
        """Test combination of fixed interval and motion."""
        config = KeyframeSelectionConfig(
            strategy=KeyframeSelectionStrategy.HYBRID,
            fixed_interval=5,
            max_interval=8,
            translation_threshold=0.3,
            rotation_threshold=0.4,
            force_keyframe_on_motion=True,
            min_time_gap=0.001  # Very small but > 0
        )
        
        selector = HybridSelector(config)
        
        # Create poses with varying motion
        poses = []
        for i in range(20):
            if i < 10:
                # Small motion
                position = np.array([i * 0.05, 0, 0])
            else:
                # Large motion
                position = np.array([0.5 + (i-10) * 0.2, 0, 0])
            
            pose = Pose(
                timestamp=i * 0.1,
                position=position,
                rotation_matrix=np.eye(3)
            )
            poses.append(pose)
        
        selected = []
        for i, pose in enumerate(poses):
            if selector.should_select_keyframe(pose, i * 0.1):
                selector.mark_keyframe(pose, i * 0.1)
                selected.append(i)
        
        # Should have first frame
        assert 0 in selected
        
        # Should have more keyframes during high motion period
        low_motion_keyframes = len([i for i in selected if i < 10])
        high_motion_keyframes = len([i for i in selected if i >= 10])
        
        # High motion period should have more frequent keyframes
        assert high_motion_keyframes >= low_motion_keyframes
    
    def test_max_interval_enforcement(self):
        """Test that max interval is enforced."""
        config = KeyframeSelectionConfig(
            strategy=KeyframeSelectionStrategy.HYBRID,
            fixed_interval=10,
            max_interval=12,  # Must be >= fixed_interval
            translation_threshold=999.0,
            rotation_threshold=999.0,
            min_time_gap=0.001  # Very small but > 0
        )
        
        selector = HybridSelector(config)
        poses = create_test_poses(20)
        
        selected = []
        for i, pose in enumerate(poses):
            if selector.should_select_keyframe(pose, i * 0.1):
                selector.mark_keyframe(pose, i * 0.1)
                selected.append(i)
        
        # Check that no interval exceeds max_interval
        for i in range(1, len(selected)):
            interval = selected[i] - selected[i-1]
            assert interval <= config.max_interval


class TestFactoryAndHelpers:
    """Test factory function and helper utilities."""
    
    def test_create_keyframe_selector(self):
        """Test selector factory."""
        # Fixed interval
        config = KeyframeSelectionConfig(
            strategy=KeyframeSelectionStrategy.FIXED_INTERVAL
        )
        selector = create_keyframe_selector(config)
        assert isinstance(selector, FixedIntervalSelector)
        
        # Motion based
        config = KeyframeSelectionConfig(
            strategy=KeyframeSelectionStrategy.MOTION_BASED
        )
        selector = create_keyframe_selector(config)
        assert isinstance(selector, MotionBasedSelector)
        
        # Hybrid
        config = KeyframeSelectionConfig(
            strategy=KeyframeSelectionStrategy.HYBRID
        )
        selector = create_keyframe_selector(config)
        assert isinstance(selector, HybridSelector)
    
    def test_select_keyframes_from_trajectory(self):
        """Test trajectory keyframe selection."""
        config = KeyframeSelectionConfig(
            strategy=KeyframeSelectionStrategy.FIXED_INTERVAL,
            fixed_interval=4
        )
        
        poses = create_test_poses(12)
        timestamps = [p.timestamp for p in poses]
        
        keyframes = select_keyframes_from_trajectory(poses, timestamps, config)
        
        # Should return list of (frame_index, keyframe_id) tuples
        assert len(keyframes) == 3  # Frames 0, 4, 8
        assert keyframes[0] == (0, 0)
        assert keyframes[1] == (4, 1)
        assert keyframes[2] == (8, 2)
    
    def test_mark_keyframes_in_camera_data(self):
        """Test marking keyframes in camera frames."""
        config = KeyframeSelectionConfig(
            strategy=KeyframeSelectionStrategy.FIXED_INTERVAL,
            fixed_interval=3
        )
        
        # Create camera frames
        frames = []
        poses = []
        for i in range(9):
            frame = CameraFrame(
                timestamp=i * 0.1,
                camera_id="cam0",
                observations=[]
            )
            frames.append(frame)
            
            pose = Pose(
                timestamp=i * 0.1,
                position=np.array([i * 0.1, 0, 0]),
                rotation_matrix=np.eye(3)
            )
            poses.append(pose)
        
        # Mark keyframes
        mark_keyframes_in_camera_data(frames, poses, config)
        
        # Check that correct frames are marked
        keyframe_indices = [i for i, f in enumerate(frames) if f.is_keyframe]
        assert keyframe_indices == [0, 3, 6]
        
        # Check keyframe IDs
        assert frames[0].keyframe_id == 0
        assert frames[3].keyframe_id == 1
        assert frames[6].keyframe_id == 2
        
        # Non-keyframes should not have IDs
        assert frames[1].keyframe_id is None
        assert frames[2].keyframe_id is None


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_valid_configs(self):
        """Test that valid configs are accepted."""
        # Fixed interval
        config = KeyframeSelectionConfig(
            strategy=KeyframeSelectionStrategy.FIXED_INTERVAL,
            fixed_interval=5
        )
        assert config.fixed_interval == 5
        
        # Motion based
        config = KeyframeSelectionConfig(
            strategy=KeyframeSelectionStrategy.MOTION_BASED,
            translation_threshold=0.5,
            rotation_threshold=0.3
        )
        assert config.translation_threshold == 0.5
        
        # Hybrid
        config = KeyframeSelectionConfig(
            strategy=KeyframeSelectionStrategy.HYBRID,
            fixed_interval=5,
            max_interval=10
        )
        assert config.max_interval == 10
    
    def test_invalid_hybrid_config(self):
        """Test that invalid hybrid config is rejected."""
        with pytest.raises(ValueError):
            # max_interval < fixed_interval should fail
            KeyframeSelectionConfig(
                strategy=KeyframeSelectionStrategy.HYBRID,
                fixed_interval=10,
                max_interval=5  # Invalid: less than fixed_interval
            )