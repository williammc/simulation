"""
Tests for IMU preintegration functionality.

Tests the preintegration pipeline, data structures, and estimator support
for preintegrated IMU measurements.
"""

import numpy as np
import pytest
from typing import List

from src.common.data_structures import (
    IMUMeasurement, CameraFrame, PreintegratedIMUData,
    Pose, TrajectoryState, Trajectory
)
from src.estimation.imu_integration import IMUPreintegrator, IMUState
from src.utils.preintegration_utils import (
    PreintegrationCache,
    preintegrate_between_keyframes,
    attach_preintegrated_to_frames,
    create_keyframe_schedule,
    split_measurements_by_keyframes
)


class TestPreintegratedIMUData:
    """Test PreintegratedIMUData structure."""
    
    def test_creation(self):
        """Test creating preintegrated IMU data."""
        data = PreintegratedIMUData(
            delta_position=np.array([1, 2, 3]),
            delta_velocity=np.array([0.1, 0.2, 0.3]),
            delta_rotation=np.eye(3),
            covariance=np.eye(15),
            dt=0.1,
            from_keyframe_id=0,
            to_keyframe_id=1,
            num_measurements=10
        )
        
        assert data.from_keyframe_id == 0
        assert data.to_keyframe_id == 1
        assert data.num_measurements == 10
        assert data.dt == 0.1
        np.testing.assert_array_equal(data.delta_position, [1, 2, 3])
        np.testing.assert_array_equal(data.delta_velocity, [0.1, 0.2, 0.3])
        assert data.delta_rotation.shape == (3, 3)
        assert data.covariance.shape == (15, 15)
    
    def test_validation(self):
        """Test dimension validation."""
        # Should reshape vectors
        data = PreintegratedIMUData(
            delta_position=np.array([[1], [2], [3]]),  # Column vector
            delta_velocity=np.array([0.1, 0.2, 0.3]),
            delta_rotation=np.eye(3),
            covariance=np.eye(15),
            dt=0.1,
            from_keyframe_id=0,
            to_keyframe_id=1,
            num_measurements=10
        )
        assert data.delta_position.shape == (3,)
        
    def test_so3_projection(self):
        """Test SO3 projection of rotation matrix."""
        # Create a non-orthogonal matrix
        R_bad = np.array([
            [1.1, 0.1, 0],
            [0, 0.9, 0],
            [0, 0, 1.2]
        ])
        
        data = PreintegratedIMUData(
            delta_position=np.zeros(3),
            delta_velocity=np.zeros(3),
            delta_rotation=R_bad,
            covariance=np.eye(15),
            dt=0.1,
            from_keyframe_id=0,
            to_keyframe_id=1,
            num_measurements=10
        )
        
        # Check that rotation was projected to SO3
        R = data.delta_rotation
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


class TestIMUPreintegrator:
    """Test IMU preintegrator functionality."""
    
    def test_reset(self):
        """Test preintegrator reset."""
        preintegrator = IMUPreintegrator()
        
        # Add some measurements
        meas = IMUMeasurement(
            timestamp=0.1,
            accelerometer=np.array([0, 0, 9.81]),
            gyroscope=np.array([0, 0, 0.1])
        )
        preintegrator.add_measurement(meas, 0.01)
        
        # Reset should clear everything
        preintegrator.reset()
        
        np.testing.assert_array_equal(preintegrator.delta_p, np.zeros(3))
        np.testing.assert_array_equal(preintegrator.delta_v, np.zeros(3))
        np.testing.assert_array_equal(preintegrator.delta_R, np.eye(3))
        assert preintegrator.dt == 0.0
        assert len(preintegrator.measurements) == 0
    
    def test_batch_process(self):
        """Test batch processing of measurements."""
        preintegrator = IMUPreintegrator()
        
        # Create a sequence of measurements
        measurements = []
        for i in range(10):
            t = i * 0.01
            meas = IMUMeasurement(
                timestamp=t,
                accelerometer=np.array([0.1, 0, 9.81]),
                gyroscope=np.array([0, 0, 0.1])
            )
            measurements.append(meas)
        
        # Batch process
        result = preintegrator.batch_process(
            measurements,
            from_keyframe_id=0,
            to_keyframe_id=1
        )
        
        assert isinstance(result, PreintegratedIMUData)
        assert result.from_keyframe_id == 0
        assert result.to_keyframe_id == 1
        assert result.num_measurements == 10
        assert result.dt > 0
        assert result.covariance.shape == (15, 15)
        
        # Check that jacobian was created
        assert result.jacobian is not None
        assert result.jacobian.shape == (15, 6)
        
        # Check that source measurements were stored
        assert result.source_measurements is not None
        assert len(result.source_measurements) == 10
    
    def test_constant_acceleration(self):
        """Test preintegration with constant acceleration."""
        preintegrator = IMUPreintegrator(gravity=np.array([0, 0, 0]))  # No gravity
        
        # Constant acceleration in x direction
        measurements = []
        dt = 0.01
        accel = np.array([1.0, 0, 0])
        
        for i in range(100):
            meas = IMUMeasurement(
                timestamp=i * dt,
                accelerometer=accel,
                gyroscope=np.zeros(3)
            )
            measurements.append(meas)
        
        result = preintegrator.batch_process(measurements, 0, 1)
        
        # With constant acceleration a=1, after t=1:
        # v = a*t = 1.0
        # p = 0.5*a*t^2 = 0.5
        total_time = dt * (len(measurements) - 1)
        expected_velocity = accel * total_time
        expected_position = 0.5 * accel * total_time**2
        
        np.testing.assert_allclose(result.delta_velocity, expected_velocity, rtol=0.1)
        np.testing.assert_allclose(result.delta_position, expected_position, rtol=0.1)
        np.testing.assert_allclose(result.delta_rotation, np.eye(3), atol=1e-10)
    
    def test_constant_rotation(self):
        """Test preintegration with constant rotation."""
        preintegrator = IMUPreintegrator(gravity=np.array([0, 0, 0]))
        
        # Constant rotation around z-axis
        measurements = []
        dt = 0.01
        omega = np.array([0, 0, np.pi/2])  # 90 deg/s
        
        for i in range(100):
            meas = IMUMeasurement(
                timestamp=i * dt,
                accelerometer=np.zeros(3),
                gyroscope=omega
            )
            measurements.append(meas)
        
        result = preintegrator.batch_process(measurements, 0, 1)
        
        # After ~1 second rotating at pi/2 rad/s, should have ~90 degree rotation
        total_angle = omega[2] * dt * (len(measurements) - 1)
        expected_R = np.array([
            [np.cos(total_angle), -np.sin(total_angle), 0],
            [np.sin(total_angle), np.cos(total_angle), 0],
            [0, 0, 1]
        ])
        
        np.testing.assert_allclose(result.delta_rotation, expected_R, atol=0.1)


class TestPreintegrationUtils:
    """Test preintegration utility functions."""
    
    def test_cache(self):
        """Test preintegration cache."""
        cache = PreintegrationCache()
        
        # Add keyframes
        cache.add_keyframe(0, 0.0)
        cache.add_keyframe(1, 0.1)
        
        # Create and store preintegrated data
        data = PreintegratedIMUData(
            delta_position=np.ones(3),
            delta_velocity=np.ones(3),
            delta_rotation=np.eye(3),
            covariance=np.eye(15),
            dt=0.1,
            from_keyframe_id=0,
            to_keyframe_id=1,
            num_measurements=10
        )
        
        cache.put(0, 1, data)
        
        # Retrieve
        retrieved = cache.get(0, 1)
        assert retrieved is not None
        assert retrieved.from_keyframe_id == 0
        assert retrieved.to_keyframe_id == 1
        
        # Non-existent should return None
        assert cache.get(1, 2) is None
        
        # Clear
        cache.clear()
        assert cache.get(0, 1) is None
    
    def test_keyframe_schedule(self):
        """Test keyframe selection schedule."""
        # Create timestamps
        timestamps = [i * 0.01 for i in range(100)]
        
        # Create schedule
        schedule = create_keyframe_schedule(
            timestamps,
            interval=10,
            min_time_gap=0.05
        )
        
        assert len(schedule) > 0
        
        # Check that keyframes are properly spaced
        for i in range(1, len(schedule)):
            time_gap = schedule[i][1] - schedule[i-1][1]
            assert time_gap >= 0.05
        
        # Check IDs are sequential
        for i, (kf_id, _) in enumerate(schedule):
            assert kf_id == i
    
    def test_split_measurements(self):
        """Test splitting measurements by keyframes."""
        # Create measurements
        measurements = []
        for i in range(100):
            meas = IMUMeasurement(
                timestamp=i * 0.01,
                accelerometer=np.zeros(3),
                gyroscope=np.zeros(3)
            )
            measurements.append(meas)
        
        # Define keyframe times
        keyframe_times = [0.0, 0.3, 0.6, 0.9]
        
        # Split
        segments = split_measurements_by_keyframes(measurements, keyframe_times)
        
        assert len(segments) == len(keyframe_times) - 1
        
        # Check that all measurements are accounted for
        total_measurements = sum(len(seg) for seg in segments)
        assert total_measurements <= len(measurements)
        
        # Check time ranges
        for i, segment in enumerate(segments):
            if segment:
                assert segment[0].timestamp >= keyframe_times[i]
                assert segment[-1].timestamp < keyframe_times[i + 1]
    
    def test_preintegrate_between_keyframes(self):
        """Test full preintegration pipeline."""
        # Create IMU measurements
        measurements = []
        for i in range(100):
            t = i * 0.01
            meas = IMUMeasurement(
                timestamp=t,
                accelerometer=np.array([0, 0, 9.81]),
                gyroscope=np.array([0, 0, 0.1])
            )
            measurements.append(meas)
        
        # Define keyframes
        keyframe_ids = [0, 1, 2]
        keyframe_times = [0.0, 0.3, 0.6]
        
        # Preintegrate
        result = preintegrate_between_keyframes(
            measurements,
            keyframe_ids,
            keyframe_times
        )
        
        # Should have preintegrated data for keyframes 1 and 2
        assert 1 in result
        assert 2 in result
        
        # Check that preintegrated data has correct IDs
        assert result[1].from_keyframe_id == 0
        assert result[1].to_keyframe_id == 1
        assert result[2].from_keyframe_id == 1
        assert result[2].to_keyframe_id == 2
        
        # Check that measurements were processed
        assert result[1].num_measurements > 0
        assert result[2].num_measurements > 0
    
    def test_attach_to_frames(self):
        """Test attaching preintegrated data to camera frames."""
        # Create camera frames
        frames = []
        for i in range(3):
            frame = CameraFrame(
                timestamp=i * 0.1,
                camera_id="cam0",
                observations=[],
                is_keyframe=(i > 0),
                keyframe_id=i if i > 0 else None
            )
            frames.append(frame)
        
        # Create preintegrated data
        preintegrated = {
            1: PreintegratedIMUData(
                delta_position=np.ones(3),
                delta_velocity=np.ones(3),
                delta_rotation=np.eye(3),
                covariance=np.eye(15),
                dt=0.1,
                from_keyframe_id=0,
                to_keyframe_id=1,
                num_measurements=10
            ),
            2: PreintegratedIMUData(
                delta_position=np.ones(3) * 2,
                delta_velocity=np.ones(3) * 2,
                delta_rotation=np.eye(3),
                covariance=np.eye(15),
                dt=0.1,
                from_keyframe_id=1,
                to_keyframe_id=2,
                num_measurements=10
            )
        }
        
        # Attach
        attach_preintegrated_to_frames(frames, preintegrated)
        
        # Check that keyframes have preintegrated data
        assert frames[0].preintegrated_imu is None  # Not a keyframe or no data
        assert frames[1].preintegrated_imu is not None
        assert frames[2].preintegrated_imu is not None
        
        # Check that data is correct
        np.testing.assert_array_equal(frames[1].preintegrated_imu.delta_position, np.ones(3))
        np.testing.assert_array_equal(frames[2].preintegrated_imu.delta_position, np.ones(3) * 2)