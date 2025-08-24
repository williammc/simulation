"""
Utility functions for IMU preintegration between keyframes.

Provides helpers for converting raw IMU measurements to preintegrated
data and managing keyframe associations.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

from src.common.data_structures import (
    IMUMeasurement, CameraFrame, PreintegratedIMUData
)
from src.estimation.imu_integration import IMUPreintegrator

logger = logging.getLogger(__name__)


class PreintegrationCache:
    """
    Cache for preintegrated IMU data between keyframes.
    
    Stores preintegrated results to avoid recomputation and
    manages keyframe associations.
    """
    
    def __init__(self):
        """Initialize cache."""
        self.cache: Dict[Tuple[int, int], PreintegratedIMUData] = {}
        self.keyframe_times: Dict[int, float] = {}
        
    def add_keyframe(self, keyframe_id: int, timestamp: float) -> None:
        """
        Register a keyframe with its timestamp.
        
        Args:
            keyframe_id: Keyframe identifier
            timestamp: Keyframe timestamp
        """
        self.keyframe_times[keyframe_id] = timestamp
        
    def get(self, from_id: int, to_id: int) -> Optional[PreintegratedIMUData]:
        """
        Retrieve cached preintegration between keyframes.
        
        Args:
            from_id: Source keyframe ID
            to_id: Target keyframe ID
            
        Returns:
            Cached PreintegratedIMUData or None if not cached
        """
        return self.cache.get((from_id, to_id))
    
    def put(self, from_id: int, to_id: int, data: PreintegratedIMUData) -> None:
        """
        Store preintegrated data in cache.
        
        Args:
            from_id: Source keyframe ID
            to_id: Target keyframe ID
            data: Preintegrated IMU data
        """
        self.cache[(from_id, to_id)] = data
        
    def clear(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        self.keyframe_times.clear()


def preintegrate_between_keyframes(
    imu_measurements: List[IMUMeasurement],
    keyframe_ids: List[int],
    keyframe_times: List[float],
    preintegrator: Optional[IMUPreintegrator] = None,
    cache: Optional[PreintegrationCache] = None,
    keyframe_orientations: Optional[List[np.ndarray]] = None
) -> Dict[int, PreintegratedIMUData]:
    """
    Preintegrate IMU measurements between consecutive keyframes.
    
    Args:
        imu_measurements: All IMU measurements
        keyframe_ids: List of keyframe IDs in order
        keyframe_times: Corresponding keyframe timestamps
        preintegrator: IMU preintegrator (creates default if None)
        cache: Optional cache for storing results
        keyframe_orientations: Optional list of rotation matrices (3x3) at keyframes
        
    Returns:
        Dictionary mapping target keyframe ID to preintegrated data
    """
    if len(keyframe_ids) != len(keyframe_times):
        raise ValueError("Keyframe IDs and times must have same length")
    
    if len(keyframe_ids) < 2:
        return {}
    
    # Create preintegrator if not provided
    if preintegrator is None:
        preintegrator = IMUPreintegrator()
    
    # Sort keyframes by time
    sorted_indices = np.argsort(keyframe_times)
    sorted_ids = [keyframe_ids[i] for i in sorted_indices]
    sorted_times = [keyframe_times[i] for i in sorted_indices]
    
    # Update cache with keyframe times
    if cache is not None:
        for kf_id, kf_time in zip(sorted_ids, sorted_times):
            cache.add_keyframe(kf_id, kf_time)
    
    # Group IMU measurements by time intervals
    result = {}
    
    for i in range(len(sorted_ids) - 1):
        from_id = sorted_ids[i]
        to_id = sorted_ids[i + 1]
        from_time = sorted_times[i]
        to_time = sorted_times[i + 1]
        
        # Check cache first
        if cache is not None:
            cached_data = cache.get(from_id, to_id)
            if cached_data is not None:
                result[to_id] = cached_data
                continue
        
        # Extract measurements in this interval
        interval_measurements = []
        for meas in imu_measurements:
            if from_time <= meas.timestamp < to_time:
                interval_measurements.append(meas)
        
        if not interval_measurements:
            logger.warning(f"No IMU measurements between keyframes {from_id} and {to_id}")
            continue
        
        # Get initial orientation for this interval if available
        initial_orientation = None
        if keyframe_orientations is not None and i < len(keyframe_orientations):
            initial_orientation = keyframe_orientations[i]
        
        # Preintegrate measurements with initial orientation
        preintegrated_data = preintegrator.batch_process(
            interval_measurements,
            from_id,
            to_id,
            initial_orientation
        )
        
        # Store in cache
        if cache is not None:
            cache.put(from_id, to_id, preintegrated_data)
        
        # Store result
        result[to_id] = preintegrated_data
    
    return result


def attach_preintegrated_to_frames(
    camera_frames: List[CameraFrame],
    preintegrated_data: Dict[int, PreintegratedIMUData]
) -> None:
    """
    Attach preintegrated IMU data to corresponding camera frames.
    
    Modifies camera frames in-place by setting their preintegrated_imu field.
    
    Args:
        camera_frames: List of camera frames to modify
        preintegrated_data: Dictionary mapping keyframe ID to preintegrated data
    """
    for frame in camera_frames:
        if frame.is_keyframe and frame.keyframe_id in preintegrated_data:
            frame.preintegrated_imu = preintegrated_data[frame.keyframe_id]
            logger.debug(f"Attached preintegrated IMU to keyframe {frame.keyframe_id}")


def create_keyframe_schedule(
    timestamps: List[float],
    interval: int = 10,
    min_time_gap: float = 0.1
) -> List[Tuple[int, float]]:
    """
    Create a keyframe selection schedule based on fixed intervals.
    
    Args:
        timestamps: All frame timestamps
        interval: Select every N-th frame as keyframe
        min_time_gap: Minimum time gap between keyframes
        
    Returns:
        List of (keyframe_id, timestamp) tuples
    """
    if not timestamps:
        return []
    
    keyframes = []
    last_kf_time = -float('inf')
    keyframe_id = 0
    
    for i, timestamp in enumerate(timestamps):
        # Check if this should be a keyframe
        is_keyframe = (i % interval == 0) and (timestamp - last_kf_time >= min_time_gap)
        
        if is_keyframe:
            keyframes.append((keyframe_id, timestamp))
            last_kf_time = timestamp
            keyframe_id += 1
    
    return keyframes


def split_measurements_by_keyframes(
    measurements: List[IMUMeasurement],
    keyframe_times: List[float]
) -> List[List[IMUMeasurement]]:
    """
    Split IMU measurements into segments between keyframes.
    
    Args:
        measurements: All IMU measurements
        keyframe_times: Keyframe timestamps (sorted)
        
    Returns:
        List of measurement lists, one for each keyframe interval
    """
    if len(keyframe_times) < 2:
        return []
    
    segments = []
    
    for i in range(len(keyframe_times) - 1):
        start_time = keyframe_times[i]
        end_time = keyframe_times[i + 1]
        
        segment = [
            m for m in measurements
            if start_time <= m.timestamp < end_time
        ]
        segments.append(segment)
    
    return segments