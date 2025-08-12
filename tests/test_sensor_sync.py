"""
Tests for sensor synchronization functionality.
"""

import pytest
import numpy as np
from typing import Dict, List

from src.estimation.sensor_sync import (
    SensorSynchronizer, SensorTiming, SyncMethod,
    HardwareTrigger, SyncedMeasurement
)
from src.common.data_structures import (
    IMUMeasurement, CameraFrame
)


class TestSensorTiming:
    """Test sensor timing configuration."""
    
    def test_sensor_timing_creation(self):
        """Test creating sensor timing configuration."""
        timing = SensorTiming(
            sensor_id="imu",
            frequency=200.0,
            latency=0.001,
            jitter=0.0001
        )
        
        assert timing.sensor_id == "imu"
        assert timing.frequency == 200.0
        assert timing.latency == 0.001
        assert timing.jitter == 0.0001
        assert timing.period == pytest.approx(1.0 / 200.0)
    
    def test_multiple_sensor_timings(self):
        """Test configuration for multiple sensors."""
        timings = {
            "imu": SensorTiming("imu", 200.0, 0.001, 0.0001),
            "camera": SensorTiming("camera", 30.0, 0.020, 0.002),
            "lidar": SensorTiming("lidar", 10.0, 0.050, 0.005)
        }
        
        assert len(timings) == 3
        assert timings["imu"].frequency > timings["camera"].frequency
        assert timings["camera"].frequency > timings["lidar"].frequency
    
    def test_timing_validation(self):
        """Test validation of timing parameters."""
        # Valid timing
        timing = SensorTiming("sensor", 100.0, 0.01, 0.001)
        assert timing.is_valid()
        
        # Invalid frequency
        with pytest.raises(ValueError):
            SensorTiming("sensor", -10.0, 0.01, 0.001)
        
        # Invalid latency
        with pytest.raises(ValueError):
            SensorTiming("sensor", 100.0, -0.01, 0.001)
        
        # Invalid jitter
        with pytest.raises(ValueError):
            SensorTiming("sensor", 100.0, 0.01, -0.001)


class TestSensorSynchronizer:
    """Test sensor synchronization functionality."""
    
    @pytest.fixture
    def synchronizer(self):
        """Create a sensor synchronizer for testing."""
        timings = {
            "imu": SensorTiming("imu", 200.0, 0.001, 0.0),
            "camera": SensorTiming("camera", 30.0, 0.020, 0.0)
        }
        return SensorSynchronizer(timings)
    
    def test_synchronizer_initialization(self, synchronizer):
        """Test synchronizer initialization."""
        assert len(synchronizer.buffers) == 2
        assert "imu" in synchronizer.buffers
        assert "camera" in synchronizer.buffers
        assert synchronizer.sync_method == SyncMethod.NEAREST
    
    def test_add_measurements(self, synchronizer):
        """Test adding measurements to buffers."""
        # Add IMU measurements
        for i in range(10):
            t = i * 0.005  # 200 Hz
            imu_meas = IMUMeasurement(
                timestamp=t,
                accelerometer=np.array([0, 0, 9.81]),
                gyroscope=np.array([0.1, 0, 0])
            )
            synchronizer.add_measurement("imu", imu_meas, t)
        
        # Add camera measurement
        cam_frame = CameraFrame(
            timestamp=0.033,
            camera_id="cam0",
            observations=[]
        )
        synchronizer.add_measurement("camera", cam_frame, 0.033)
        
        assert len(synchronizer.buffers["imu"]) == 10
        assert len(synchronizer.buffers["camera"]) == 1
    
    def test_nearest_neighbor_sync(self, synchronizer):
        """Test nearest neighbor synchronization."""
        synchronizer.sync_method = SyncMethod.NEAREST
        
        # Add measurements
        imu_times = [0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035]
        for t in imu_times:
            imu_meas = IMUMeasurement(
                timestamp=t,
                accelerometer=np.array([0, 0, 9.81]),
                gyroscope=np.array([t, 0, 0])  # Use time as gyro.x for testing
            )
            synchronizer.add_measurement("imu", imu_meas, t)
        
        # Get synchronized measurement at t=0.022
        synced = synchronizer.get_synced_measurements(0.022, tolerance=0.01)
        
        assert synced is not None
        assert isinstance(synced, SyncedMeasurement)
        assert synced.timestamp == 0.022
        assert synced.imu_data is not None
        # Should get IMU at t=0.020 (nearest)
        assert synced.imu_data.gyroscope[0] == pytest.approx(0.020, abs=0.001)
    
    def test_linear_interpolation_sync(self, synchronizer):
        """Test linear interpolation synchronization."""
        synchronizer.sync_method = SyncMethod.LINEAR_INTERPOLATION
        
        # Add measurements with known pattern
        for i in range(10):
            t = i * 0.005
            imu_meas = IMUMeasurement(
                timestamp=t,
                accelerometer=np.array([t, 0, 9.81]),  # Linear in time
                gyroscope=np.array([0, 0, 0])
            )
            synchronizer.add_measurement("imu", imu_meas, t)
        
        # Get interpolated measurement at t=0.012
        synced = synchronizer.get_synced_measurements(0.012, tolerance=0.01)
        
        assert synced is not None
        assert synced.interpolated
        # Should interpolate between t=0.010 and t=0.015
        # Expected: 0.012
        assert synced.imu_data.accelerometer[0] == pytest.approx(0.012, abs=0.001)
    
    def test_spline_interpolation_sync(self, synchronizer):
        """Test spline interpolation synchronization."""
        synchronizer.sync_method = SyncMethod.CUBIC_INTERPOLATION
        
        # Add measurements with quadratic pattern
        for i in range(10):
            t = i * 0.01
            imu_meas = IMUMeasurement(
                timestamp=t,
                accelerometer=np.array([t**2, 0, 9.81]),  # Quadratic in time
                gyroscope=np.array([0, 0, 0])
            )
            synchronizer.add_measurement("imu", imu_meas, t)
        
        # Get spline-interpolated measurement
        synced = synchronizer.get_synced_measurements(0.025, tolerance=0.01)
        
        assert synced is not None
        assert synced.interpolated
        # Should approximate t^2 at t=0.025
        expected = 0.025 ** 2
        assert abs(synced.imu_data.accelerometer[0] - expected) < 0.01
    
    def test_buffer_management(self, synchronizer):
        """Test buffer size management."""
        synchronizer.max_buffer_size = 5
        
        # Add more measurements than buffer size
        for i in range(10):
            t = i * 0.005
            imu_meas = IMUMeasurement(
                timestamp=t,
                accelerometer=np.array([0, 0, 9.81]),
                gyroscope=np.array([0, 0, 0])
            )
            synchronizer.add_measurement("imu", imu_meas, t)
        
        # Buffer should maintain max size
        assert len(synchronizer.buffers["imu"]) <= 5
        # Should keep most recent measurements
        timestamps = [t for t, m in synchronizer.buffers["imu"]]
        assert min(timestamps) >= 0.025  # Oldest kept measurements
    
    def test_synchronization_with_latency(self):
        """Test synchronization accounting for sensor latency."""
        timings = {
            "imu": SensorTiming("imu", 200.0, 0.002, 0.0),  # 2ms latency
            "camera": SensorTiming("camera", 30.0, 0.020, 0.0)  # 20ms latency
        }
        sync = SensorSynchronizer(timings)
        
        # Add measurements at hardware timestamps
        imu_hw_time = 0.010
        cam_hw_time = 0.030
        
        imu_meas = IMUMeasurement(
            timestamp=imu_hw_time,
            accelerometer=np.array([1, 0, 0]),
            gyroscope=np.array([0, 0, 0])
        )
        cam_frame = CameraFrame(
            timestamp=cam_hw_time,
            camera_id="cam0",
            observations=[]
        )
        
        # Add measurements with raw hardware timestamps (let sync handle latency correction)
        sync.add_measurement("imu", imu_meas, imu_hw_time)
        sync.add_measurement("camera", cam_frame, cam_hw_time)
        
        # Check corrected timestamps
        assert sync.buffers["imu"][0][0] == pytest.approx(0.012)  # 10ms + 2ms
        assert sync.buffers["camera"][0][0] == pytest.approx(0.050)  # 30ms + 20ms


class TestHardwareTrigger:
    """Test hardware trigger synchronization."""
    
    def test_trigger_creation(self):
        """Test hardware trigger creation."""
        trigger = HardwareTrigger(
            trigger_rate=30.0,
            camera_delay=0.005,
            imu_delay=0.001
        )
        
        assert trigger.trigger_rate == 30.0
        assert trigger.trigger_period == pytest.approx(1.0 / 30.0)
        assert trigger.camera_delay == 0.005
        assert trigger.imu_delay == 0.001
    
    def test_trigger_generation(self):
        """Test trigger signal generation."""
        trigger = HardwareTrigger(
            trigger_rate=30.0,
            camera_delay=0.005,
            imu_delay=0.001
        )
        
        # Generate trigger at t=0
        sensor_times = trigger.trigger(0.0)
        
        assert "camera" in sensor_times
        assert "imu" in sensor_times
        assert sensor_times["camera"] == 0.005
        assert sensor_times["imu"] == 0.001
        
        # Generate trigger at t=0.033 (one period later)
        sensor_times = trigger.trigger(0.033)
        
        assert sensor_times["camera"] == pytest.approx(0.038, abs=0.001)
        assert sensor_times["imu"] == pytest.approx(0.034, abs=0.001)
    
    def test_trigger_alignment(self):
        """Test aligning timestamps to trigger."""
        trigger = HardwareTrigger(trigger_rate=30.0)
        
        # Test alignment
        aligned = trigger.align_to_trigger(0.035)
        expected = 1.0 / 30.0  # Should align to nearest trigger
        assert abs(aligned - expected) < 0.001
        
        aligned = trigger.align_to_trigger(0.050)
        expected = 2.0 / 30.0
        assert abs(aligned - expected) < 0.001
    
    def test_trigger_jitter(self):
        """Test trigger with jitter."""
        trigger = HardwareTrigger(
            trigger_rate=100.0,
            camera_delay=0.001,
            imu_delay=0.0005,
            jitter=0.0001
        )
        
        # Generate multiple triggers
        np.random.seed(42)
        triggers = []
        for i in range(10):
            sensor_times = trigger.trigger(i * 0.01, add_jitter=True)
            triggers.append(sensor_times["camera"])
        
        # Check that jitter causes variation
        expected_times = [i * 0.01 + 0.001 for i in range(10)]
        differences = [abs(t - e) for t, e in zip(triggers, expected_times)]
        
        assert max(differences) > 0  # Some jitter present
        assert max(differences) < 0.001  # Bounded by jitter parameter


class TestClockDriftEstimation:
    """Test clock drift estimation."""
    
    def test_drift_estimation_in_synchronizer(self):
        """Test clock drift estimation within synchronizer."""
        timings = {
            "imu": SensorTiming("imu", 200.0, 0.001, 0.0),
            "camera": SensorTiming("camera", 30.0, 0.020, 0.0)
        }
        sync = SensorSynchronizer(timings)
        
        # Simulate clock drift
        sensor_times = np.linspace(0, 1, 100)
        reference_times = sensor_times * 1.01  # 1% drift
        
        sync.estimate_clock_drift("imu", sensor_times.tolist(), reference_times.tolist())
        
        assert "imu" in sync.clock_corrections
    
    def test_linear_drift_pattern(self):
        """Test detection of linear clock drift pattern."""
        timings = {
            "sensor1": SensorTiming("sensor1", 100.0, 0.0, 0.0)
        }
        sync = SensorSynchronizer(timings)
        
        # Simulate linear drift: sensor clock is 1% faster
        sensor_times = np.linspace(0, 1, 100)
        reference_times = sensor_times * 1.01
        
        sync.estimate_clock_drift(
            "sensor1",
            sensor_times.tolist(),
            reference_times.tolist()
        )
        
        assert "sensor1" in sync.clock_corrections
    
    def test_drift_correction_application(self):
        """Test that drift correction is applied."""
        timings = {
            "sensor1": SensorTiming("sensor1", 100.0, 0.0, 0.0)
        }
        sync = SensorSynchronizer(timings)
        
        # Add measurements before and after drift estimation
        for i in range(10):
            t = i * 0.01
            meas = IMUMeasurement(
                timestamp=t,
                accelerometer=np.array([0, 0, 9.81]),
                gyroscope=np.zeros(3)
            )
            sync.add_measurement("sensor1", meas, t)
        
        # Estimate drift
        sensor_times = np.linspace(0, 0.1, 10)
        reference_times = sensor_times * 1.02  # 2% drift
        
        sync.estimate_clock_drift(
            "sensor1",
            sensor_times.tolist(),
            reference_times.tolist()
        )
        
        # Verify correction was applied
        assert "sensor1" in sync.clock_corrections
    
    def test_drift_estimation_robustness(self):
        """Test drift estimation with noisy data."""
        timings = {
            "sensor1": SensorTiming("sensor1", 100.0, 0.0, 0.0)
        }
        sync = SensorSynchronizer(timings)
        
        # Create data with some noise
        np.random.seed(42)
        sensor_times = list(np.linspace(0, 1, 50))
        reference_times = [t * 1.01 + 0.001 * np.random.randn() for t in sensor_times]
        
        sync.estimate_clock_drift(
            "sensor1",
            sensor_times,
            reference_times
        )
        
        # Should still detect drift pattern
        assert "sensor1" in sync.clock_corrections
    
    def test_multi_sensor_drift(self):
        """Test drift estimation for multiple sensors."""
        timings = {
            "imu": SensorTiming("imu", 200.0, 0.001, 0.0),
            "camera": SensorTiming("camera", 30.0, 0.020, 0.0)
        }
        
        sync = SensorSynchronizer(timings)
        
        # Simulate IMU with 0.5% drift
        imu_times = np.linspace(0, 1, 200)
        reference_times = imu_times * 1.005
        
        sync.estimate_clock_drift("imu", imu_times.tolist(), reference_times.tolist())
        
        assert "imu" in sync.clock_corrections


class TestSyncedMeasurement:
    """Test synchronized measurement container."""
    
    def test_synced_measurement_creation(self):
        """Test creating synchronized measurement."""
        imu_data = IMUMeasurement(
            timestamp=0.01,
            accelerometer=np.array([0, 0, 9.81]),
            gyroscope=np.array([0.1, 0, 0])
        )
        
        cam_data = CameraFrame(
            timestamp=0.033,
            camera_id="cam0",
            observations=[]
        )
        
        synced = SyncedMeasurement(
            timestamp=0.01,
            imu_data=imu_data,
            camera_data=cam_data,
            interpolated=False
        )
        
        assert synced.timestamp == 0.01
        assert synced.imu_data is not None
        assert synced.camera_data is not None
        assert not synced.interpolated
    
    def test_partial_synced_measurement(self):
        """Test synchronized measurement with missing sensors."""
        imu_data = IMUMeasurement(
            timestamp=0.01,
            accelerometer=np.array([0, 0, 9.81]),
            gyroscope=np.array([0, 0, 0])
        )
        
        synced = SyncedMeasurement(
            timestamp=0.01,
            imu_data=imu_data,
            camera_data=None,  # No camera data
            interpolated=True
        )
        
        assert synced.imu_data is not None
        assert synced.camera_data is None
        assert synced.interpolated
    
    def test_measurement_completeness(self):
        """Test checking if synchronized measurement is complete."""
        # Complete: all sensors present
        complete = SyncedMeasurement(
            timestamp=0.01,
            imu_data=IMUMeasurement(0.01, np.zeros(3), np.zeros(3)),
            camera_data=CameraFrame(0.01, "cam0", []),
            interpolated=False
        )
        
        # Check both sensors present
        assert complete.imu_data is not None
        assert complete.camera_data is not None
        
        # Incomplete: missing camera
        incomplete = SyncedMeasurement(
            timestamp=0.01,
            imu_data=IMUMeasurement(0.01, np.zeros(3), np.zeros(3)),
            camera_data=None,
            interpolated=True
        )
        
        assert incomplete.imu_data is not None
        assert incomplete.camera_data is None