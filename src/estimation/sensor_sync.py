"""
Sensor synchronization and temporal alignment.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import bisect

from src.common.data_structures import (
    IMUMeasurement, CameraFrame, Pose
)


class SyncMethod(Enum):
    """Synchronization methods."""
    NEAREST = "nearest"
    LINEAR_INTERPOLATION = "linear_interpolation"
    CUBIC_INTERPOLATION = "cubic_interpolation"
    TRIGGER = "hardware_trigger"


@dataclass
class SensorTiming:
    """Timing information for a sensor."""
    sensor_id: str
    frequency: float  # Hz
    latency: float  # seconds
    jitter: float  # seconds (std dev)
    clock_offset: float = 0.0  # Offset from master clock
    drift_rate: float = 0.0  # Clock drift in ppm
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.frequency <= 0:
            raise ValueError(f"Frequency must be positive, got {self.frequency}")
        if self.latency < 0:
            raise ValueError(f"Latency must be non-negative, got {self.latency}")
        if self.jitter < 0:
            raise ValueError(f"Jitter must be non-negative, got {self.jitter}")
    
    @property
    def period(self) -> float:
        """Get the period (1/frequency) in seconds."""
        return 1.0 / self.frequency if self.frequency > 0 else float('inf')
    
    def is_valid(self) -> bool:
        """Check if timing parameters are valid."""
        return (
            self.frequency > 0 and
            self.latency >= 0 and
            self.jitter >= 0
        )


@dataclass
class SyncedMeasurement:
    """Synchronized measurement bundle."""
    timestamp: float  # Master clock timestamp
    imu_data: Optional[IMUMeasurement] = None
    camera_data: Optional[CameraFrame] = None
    auxiliary_data: Dict[str, Any] = field(default_factory=dict)
    interpolated: bool = False
    sync_quality: float = 1.0  # 0-1 quality metric


class SensorSynchronizer:
    """
    Synchronizes measurements from multiple sensors with different rates and delays.
    
    Handles:
    - Different sensor frequencies
    - Clock skew and drift
    - Measurement latency
    - Temporal interpolation
    - Buffer management
    """
    
    def __init__(
        self,
        sensor_timings: Dict[str, SensorTiming],
        master_clock: str = "imu",
        buffer_size: int = 1000,
        sync_method: SyncMethod = SyncMethod.NEAREST
    ):
        """
        Initialize sensor synchronizer.
        
        Args:
            sensor_timings: Timing configuration for each sensor
            master_clock: Reference sensor for synchronization
            buffer_size: Size of measurement buffers
            sync_method: Method for temporal alignment
        """
        self.sensor_timings = sensor_timings
        self.master_clock = master_clock
        self.buffer_size = buffer_size
        self.sync_method = sync_method
        
        # Measurement buffers
        self.buffers = {
            sensor_id: deque(maxlen=buffer_size)
            for sensor_id in sensor_timings.keys()
        }
        
        # Clock correction estimates
        self.clock_corrections = {
            sensor_id: ClockCorrection()
            for sensor_id in sensor_timings.keys()
        }
        
        # Synchronization statistics
        self.sync_stats = {
            "total_syncs": 0,
            "successful_syncs": 0,
            "interpolation_count": 0,
            "dropped_measurements": 0
        }
    
    @property
    def max_buffer_size(self):
        """Get maximum buffer size."""
        return self.buffer_size
    
    @max_buffer_size.setter
    def max_buffer_size(self, value: int):
        """Set maximum buffer size and resize buffers."""
        self.buffer_size = value
        # Recreate buffers with new size
        for sensor_id in self.buffers.keys():
            old_buffer = list(self.buffers[sensor_id])
            self.buffers[sensor_id] = deque(old_buffer, maxlen=value)
    
    def add_measurement(
        self,
        sensor_id: str,
        measurement: Any,
        timestamp: float
    ):
        """
        Add a measurement to the buffer.
        
        Args:
            sensor_id: Sensor identifier
            measurement: Sensor measurement
            timestamp: Measurement timestamp (in sensor clock)
        """
        if sensor_id not in self.buffers:
            return
        
        # Apply clock correction
        corrected_timestamp = self._correct_timestamp(sensor_id, timestamp)
        
        # Store with corrected timestamp
        self.buffers[sensor_id].append((corrected_timestamp, measurement))
    
    def get_synced_measurements(
        self,
        target_timestamp: float,
        tolerance: float = 0.01
    ) -> Optional[SyncedMeasurement]:
        """
        Get synchronized measurements at target timestamp.
        
        Args:
            target_timestamp: Desired timestamp (in master clock)
            tolerance: Maximum time difference for valid sync
        
        Returns:
            Synchronized measurement bundle or None if sync fails
        """
        self.sync_stats["total_syncs"] += 1
        
        synced = SyncedMeasurement(timestamp=target_timestamp)
        
        # Process each sensor
        for sensor_id, buffer in self.buffers.items():
            if not buffer:
                continue
            
            # Get measurement at target time
            measurement, quality, interpolated = self._get_measurement_at_time(
                sensor_id, buffer, target_timestamp, tolerance
            )
            
            if measurement is not None:
                # Store based on sensor type
                if "imu" in sensor_id.lower():
                    synced.imu_data = measurement
                elif "camera" in sensor_id.lower() or "cam" in sensor_id.lower():
                    synced.camera_data = measurement
                else:
                    synced.auxiliary_data[sensor_id] = measurement
                
                if interpolated:
                    self.sync_stats["interpolation_count"] += 1
                
                synced.interpolated = synced.interpolated or interpolated
                synced.sync_quality = min(synced.sync_quality, quality)
        
        # Check if we have minimum required sensors
        if synced.imu_data is not None or synced.camera_data is not None:
            self.sync_stats["successful_syncs"] += 1
            return synced
        
        return None
    
    def _get_measurement_at_time(
        self,
        sensor_id: str,
        buffer: deque,
        target_time: float,
        tolerance: float
    ) -> Tuple[Optional[Any], float, bool]:
        """
        Get measurement at specific time using configured method.
        
        Args:
            sensor_id: Sensor identifier
            buffer: Measurement buffer
            target_time: Target timestamp
            tolerance: Time tolerance
        
        Returns:
            Tuple of (measurement, quality, interpolated)
        """
        if not buffer:
            return None, 0.0, False
        
        # Convert buffer to lists for easier processing
        timestamps = [t for t, _ in buffer]
        measurements = [m for _, m in buffer]
        
        if self.sync_method == SyncMethod.NEAREST:
            return self._nearest_neighbor(
                timestamps, measurements, target_time, tolerance
            )
        elif self.sync_method == SyncMethod.LINEAR_INTERPOLATION:
            return self._linear_interpolation(
                sensor_id, timestamps, measurements, target_time, tolerance
            )
        elif self.sync_method == SyncMethod.CUBIC_INTERPOLATION:
            return self._cubic_interpolation(
                sensor_id, timestamps, measurements, target_time, tolerance
            )
        else:
            # Hardware trigger - assume exact sync
            idx = bisect.bisect_left(timestamps, target_time)
            if idx < len(timestamps) and abs(timestamps[idx] - target_time) < tolerance:
                return measurements[idx], 1.0, False
            return None, 0.0, False
    
    def _nearest_neighbor(
        self,
        timestamps: List[float],
        measurements: List[Any],
        target_time: float,
        tolerance: float
    ) -> Tuple[Optional[Any], float, bool]:
        """Find nearest measurement in time."""
        if not timestamps:
            return None, 0.0, False
        
        # Find nearest timestamp
        idx = bisect.bisect_left(timestamps, target_time)
        
        candidates = []
        if idx > 0:
            candidates.append((idx - 1, abs(timestamps[idx - 1] - target_time)))
        if idx < len(timestamps):
            candidates.append((idx, abs(timestamps[idx] - target_time)))
        
        if not candidates:
            return None, 0.0, False
        
        # Choose closest
        best_idx, best_diff = min(candidates, key=lambda x: x[1])
        
        if best_diff > tolerance:
            return None, 0.0, False
        
        # Quality based on time difference
        quality = 1.0 - (best_diff / tolerance)
        
        return measurements[best_idx], quality, False
    
    def _linear_interpolation(
        self,
        sensor_id: str,
        timestamps: List[float],
        measurements: List[Any],
        target_time: float,
        tolerance: float
    ) -> Tuple[Optional[Any], float, bool]:
        """Linearly interpolate between measurements."""
        if len(timestamps) < 2:
            return self._nearest_neighbor(timestamps, measurements, target_time, tolerance)
        
        # Find bracketing measurements
        idx = bisect.bisect_left(timestamps, target_time)
        
        if idx == 0:
            # Before first measurement
            if target_time >= timestamps[0] - tolerance:
                return measurements[0], 0.5, False
            return None, 0.0, False
        
        if idx >= len(timestamps):
            # After last measurement
            if target_time <= timestamps[-1] + tolerance:
                return measurements[-1], 0.5, False
            return None, 0.0, False
        
        # Interpolate between idx-1 and idx
        t0, t1 = timestamps[idx - 1], timestamps[idx]
        m0, m1 = measurements[idx - 1], measurements[idx]
        
        # Check if gap is too large
        if t1 - t0 > 2 * tolerance:
            return None, 0.0, False
        
        # Interpolation weight
        alpha = (target_time - t0) / (t1 - t0)
        
        # Interpolate based on measurement type
        if isinstance(m0, IMUMeasurement):
            interpolated = self._interpolate_imu(m0, m1, alpha, target_time)
        elif isinstance(m0, CameraFrame):
            # Can't interpolate camera frames - use nearest
            interpolated = m0 if alpha < 0.5 else m1
        else:
            # Generic interpolation for numeric data
            interpolated = self._interpolate_generic(m0, m1, alpha)
        
        # Quality based on interpolation distance
        quality = 1.0 - abs(alpha - 0.5)
        
        return interpolated, quality, True
    
    def _cubic_interpolation(
        self,
        sensor_id: str,
        timestamps: List[float],
        measurements: List[Any],
        target_time: float,
        tolerance: float
    ) -> Tuple[Optional[Any], float, bool]:
        """Cubic interpolation using 4 points."""
        if len(timestamps) < 4:
            return self._linear_interpolation(
                sensor_id, timestamps, measurements, target_time, tolerance
            )
        
        # For now, fall back to linear
        # Full implementation would use cubic splines
        return self._linear_interpolation(
            sensor_id, timestamps, measurements, target_time, tolerance
        )
    
    def _interpolate_imu(
        self,
        m0: IMUMeasurement,
        m1: IMUMeasurement,
        alpha: float,
        timestamp: float
    ) -> IMUMeasurement:
        """Interpolate IMU measurements."""
        return IMUMeasurement(
            timestamp=timestamp,
            accelerometer=(1 - alpha) * m0.accelerometer + alpha * m1.accelerometer,
            gyroscope=(1 - alpha) * m0.gyroscope + alpha * m1.gyroscope
        )
    
    def _interpolate_generic(self, m0: Any, m1: Any, alpha: float) -> Any:
        """Generic interpolation for numeric data."""
        if hasattr(m0, '__mul__') and hasattr(m0, '__add__'):
            return (1 - alpha) * m0 + alpha * m1
        return m0 if alpha < 0.5 else m1
    
    def _correct_timestamp(self, sensor_id: str, timestamp: float) -> float:
        """
        Apply clock correction to timestamp.
        
        Args:
            sensor_id: Sensor identifier
            timestamp: Raw timestamp
        
        Returns:
            Corrected timestamp in master clock
        """
        timing = self.sensor_timings[sensor_id]
        correction = self.clock_corrections[sensor_id]
        
        # Apply static offset and latency
        corrected = timestamp + timing.clock_offset + timing.latency
        
        # Apply estimated drift correction
        if correction.drift_estimate is not None:
            corrected += correction.drift_estimate * timestamp
        
        return corrected
    
    def estimate_clock_drift(
        self,
        sensor_id: str,
        sensor_timestamps: List[float],
        reference_timestamps: List[float]
    ):
        """
        Estimate clock drift between sensors.
        
        Uses least squares to estimate linear drift.
        
        Args:
            sensor_id: Sensor to calibrate
            sensor_timestamps: Timestamps from sensor
            reference_timestamps: Corresponding reference timestamps
        """
        if len(sensor_timestamps) < 2:
            return
        
        # Least squares estimation of drift
        # reference_time = sensor_time * (1 + drift) + offset
        
        X = np.array(sensor_timestamps)
        y = np.array(reference_timestamps)
        
        # Fit linear model
        A = np.vstack([X, np.ones(len(X))]).T
        drift_plus_one, offset = np.linalg.lstsq(A, y, rcond=None)[0]
        
        drift = drift_plus_one - 1.0
        
        # Update correction
        self.clock_corrections[sensor_id].drift_estimate = drift
        self.clock_corrections[sensor_id].offset_estimate = offset
    
    def get_sync_statistics(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        stats = self.sync_stats.copy()
        
        # Add success rate
        if stats["total_syncs"] > 0:
            stats["success_rate"] = stats["successful_syncs"] / stats["total_syncs"]
            stats["interpolation_rate"] = stats["interpolation_count"] / stats["successful_syncs"]
        else:
            stats["success_rate"] = 0.0
            stats["interpolation_rate"] = 0.0
        
        # Add buffer status
        stats["buffer_status"] = {
            sensor_id: len(buffer)
            for sensor_id, buffer in self.buffers.items()
        }
        
        return stats
    
    def clear_old_measurements(self, current_time: float, window: float = 1.0):
        """
        Remove old measurements from buffers.
        
        Args:
            current_time: Current timestamp
            window: Time window to keep (seconds)
        """
        cutoff_time = current_time - window
        
        for sensor_id, buffer in self.buffers.items():
            # Remove old measurements
            while buffer and buffer[0][0] < cutoff_time:
                buffer.popleft()
                self.sync_stats["dropped_measurements"] += 1


@dataclass
class ClockCorrection:
    """Clock correction parameters."""
    drift_estimate: Optional[float] = None
    offset_estimate: Optional[float] = None
    last_update: Optional[float] = None


class HardwareTrigger:
    """
    Hardware trigger synchronization for cameras and IMUs.
    
    Simulates hardware triggering for perfect synchronization.
    """
    
    def __init__(
        self,
        trigger_rate: float = 30.0,  # Hz
        camera_delay: float = 0.005,  # Camera exposure delay
        imu_delay: float = 0.001,  # IMU sampling delay
        jitter: float = 0.0  # Trigger jitter (std dev in seconds)
    ):
        """
        Initialize hardware trigger.
        
        Args:
            trigger_rate: Trigger frequency in Hz
            camera_delay: Camera trigger to exposure delay
            imu_delay: IMU trigger to sample delay
            jitter: Standard deviation of trigger timing jitter
        """
        self.trigger_rate = trigger_rate
        self.trigger_period = 1.0 / trigger_rate
        self.camera_delay = camera_delay
        self.imu_delay = imu_delay
        self.jitter = jitter
        
        self.last_trigger = 0.0
    
    def get_next_trigger_time(self, current_time: float) -> float:
        """
        Get next trigger time.
        
        Args:
            current_time: Current timestamp
        
        Returns:
            Next trigger timestamp
        """
        next_trigger = self.last_trigger + self.trigger_period
        if next_trigger <= current_time:
            # Missed trigger, snap to next one
            n_missed = int((current_time - self.last_trigger) / self.trigger_period)
            next_trigger = self.last_trigger + (n_missed + 1) * self.trigger_period
        
        return next_trigger
    
    def trigger(self, timestamp: float, add_jitter: bool = False) -> Dict[str, float]:
        """
        Generate trigger signal.
        
        Args:
            timestamp: Trigger timestamp
            add_jitter: Whether to add timing jitter
        
        Returns:
            Dictionary of sensor timestamps after delays
        """
        self.last_trigger = timestamp
        
        # Add jitter if requested
        jitter_offset = 0.0
        if add_jitter and self.jitter > 0:
            jitter_offset = np.random.normal(0, self.jitter)
        
        return {
            "camera": timestamp + self.camera_delay + jitter_offset,
            "imu": timestamp + self.imu_delay + jitter_offset
        }
    
    def align_to_trigger(self, timestamp: float) -> float:
        """
        Align timestamp to nearest trigger.
        
        Args:
            timestamp: Raw timestamp
        
        Returns:
            Aligned timestamp
        """
        n = round(timestamp / self.trigger_period)
        return n * self.trigger_period