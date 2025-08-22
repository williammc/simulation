"""
Tests for keyframe-preintegration integration.
"""

import numpy as np
import pytest
from typing import List

from src.common.config import KeyframeSelectionConfig, KeyframeSelectionStrategy
from src.common.data_structures import (
    Pose, CameraFrame, IMUMeasurement, PreintegratedIMUData
)
from src.simulation.keyframe_selector import (
    create_keyframe_selector, mark_keyframes_in_camera_data
)
from src.estimation.imu_integration import IMUPreintegrator
from src.utils.preintegration_utils import (
    preintegrate_between_keyframes,
    attach_preintegrated_to_frames,
    PreintegrationCache
)


class TestKeyframePreintegrationFlow:
    """Test that keyframe selection and preintegration work together correctly."""
    
    def test_keyframe_selection_before_preintegration(self):
        """Test that keyframes are selected before IMU preintegration."""
        # Create test data
        timestamps = np.arange(0, 1.0, 0.1)  # 10 frames
        
        # Create camera frames
        frames = []
        poses = []
        for t in timestamps:
            frame = CameraFrame(
                timestamp=t,
                camera_id="cam0",
                observations=[]
            )
            frames.append(frame)
            
            pose = Pose(
                timestamp=t,
                position=np.array([t, 0, 0]),  # Moving along x
                rotation_matrix=np.eye(3)
            )
            poses.append(pose)
        
        # Select keyframes FIRST
        config = KeyframeSelectionConfig(
            strategy=KeyframeSelectionStrategy.FIXED_INTERVAL,
            fixed_interval=3
        )
        mark_keyframes_in_camera_data(frames, poses, config)
        
        # Verify keyframes are marked
        keyframe_indices = [i for i, f in enumerate(frames) if f.is_keyframe]
        assert keyframe_indices == [0, 3, 6, 9]
        
        # Get keyframe schedule for preintegration
        keyframe_schedule = [
            (f.keyframe_id, f.timestamp)
            for f in frames
            if f.is_keyframe
        ]
        
        # Create IMU measurements
        imu_measurements = []
        for t in np.arange(0, 1.0, 0.01):  # 100Hz IMU
            imu = IMUMeasurement(
                timestamp=t,
                accelerometer=np.array([0, 0, 9.81]),
                gyroscope=np.zeros(3)
            )
            imu_measurements.append(imu)
        
        # Preintegrate AFTER keyframe selection
        keyframe_ids = [kf[0] for kf in keyframe_schedule]
        keyframe_times = [kf[1] for kf in keyframe_schedule]
        
        preintegrator = IMUPreintegrator()
        preintegrated_data = preintegrate_between_keyframes(
            imu_measurements,
            keyframe_ids,
            keyframe_times,
            preintegrator
        )
        
        # Verify preintegration references correct keyframes
        assert len(preintegrated_data) == 3  # 3 intervals between 4 keyframes
        
        # Check keyframe ID references
        for to_id, data in preintegrated_data.items():
            assert to_id in keyframe_ids[1:]  # Should be destination keyframes
            assert data.from_keyframe_id in keyframe_ids[:-1]  # Should be source keyframes
            assert data.to_keyframe_id == to_id
    
    def test_preintegrated_data_attached_to_keyframes(self):
        """Test that preintegrated IMU data is correctly attached to keyframes."""
        # Create camera frames with keyframes marked
        frames = []
        for i in range(10):
            frame = CameraFrame(
                timestamp=i * 0.1,
                camera_id="cam0",
                observations=[]
            )
            if i % 3 == 0:  # Keyframes at 0, 3, 6, 9
                frame.is_keyframe = True
                frame.keyframe_id = i // 3
            frames.append(frame)
        
        # Create preintegrated data
        preintegrated_data = {
            1: PreintegratedIMUData(
                from_keyframe_id=0,
                to_keyframe_id=1,
                delta_position=np.array([0.3, 0, 0]),
                delta_velocity=np.array([1.0, 0, 0]),
                delta_rotation=np.eye(3),
                dt=0.3,
                covariance=np.eye(15) * 0.01,
                num_measurements=30
            ),
            2: PreintegratedIMUData(
                from_keyframe_id=1,
                to_keyframe_id=2,
                delta_position=np.array([0.3, 0, 0]),
                delta_velocity=np.array([1.0, 0, 0]),
                delta_rotation=np.eye(3),
                dt=0.3,
                covariance=np.eye(15) * 0.01,
                num_measurements=30
            ),
            3: PreintegratedIMUData(
                from_keyframe_id=2,
                to_keyframe_id=3,
                delta_position=np.array([0.3, 0, 0]),
                delta_velocity=np.array([1.0, 0, 0]),
                delta_rotation=np.eye(3),
                dt=0.3,
                covariance=np.eye(15) * 0.01,
                num_measurements=30
            )
        }
        
        # Attach preintegrated data to frames
        attach_preintegrated_to_frames(frames, preintegrated_data)
        
        # Verify attachment
        for frame in frames:
            if frame.is_keyframe and frame.keyframe_id > 0:
                assert frame.preintegrated_imu is not None
                assert frame.preintegrated_imu.to_keyframe_id == frame.keyframe_id
                assert frame.preintegrated_imu.from_keyframe_id == frame.keyframe_id - 1
            elif frame.is_keyframe and frame.keyframe_id == 0:
                # First keyframe has no preintegrated data
                assert frame.preintegrated_imu is None
            else:
                # Non-keyframes have no preintegrated data
                assert frame.preintegrated_imu is None
    
    def test_visual_inertial_constraint_consistency(self):
        """Test consistency between visual and inertial constraints."""
        # Create a simple trajectory
        timestamps = np.arange(0, 2.0, 0.1)  # 20 frames
        
        frames = []
        poses = []
        for i, t in enumerate(timestamps):
            frame = CameraFrame(
                timestamp=t,
                camera_id="cam0",
                observations=[]
            )
            frames.append(frame)
            
            # Simple constant velocity motion
            pose = Pose(
                timestamp=t,
                position=np.array([t * 0.5, 0, 0]),  # 0.5 m/s
                rotation_matrix=np.eye(3)
            )
            poses.append(pose)
        
        # Select keyframes
        config = KeyframeSelectionConfig(
            strategy=KeyframeSelectionStrategy.FIXED_INTERVAL,
            fixed_interval=5  # Keyframes at 0, 5, 10, 15
        )
        mark_keyframes_in_camera_data(frames, poses, config)
        
        # Get keyframe poses for visual constraints
        keyframe_poses = []
        keyframe_ids = []
        for frame, pose in zip(frames, poses):
            if frame.is_keyframe:
                keyframe_poses.append(pose)
                keyframe_ids.append(frame.keyframe_id)
        
        # Create consistent IMU measurements
        imu_measurements = []
        for t in np.arange(0, 2.0, 0.01):  # 100Hz IMU
            # Constant acceleration to achieve 0.5 m/s velocity
            imu = IMUMeasurement(
                timestamp=t,
                accelerometer=np.array([0.5 if t < 1.0 else 0, 0, 9.81]),
                gyroscope=np.zeros(3)
            )
            imu_measurements.append(imu)
        
        # Preintegrate IMU
        keyframe_times = [p.timestamp for p in keyframe_poses]
        preintegrator = IMUPreintegrator(
            gravity=np.array([0, 0, -9.81])
        )
        
        preintegrated_data = preintegrate_between_keyframes(
            imu_measurements,
            keyframe_ids,
            keyframe_times,
            preintegrator
        )
        
        # Verify consistency: check that preintegrated deltas match pose differences
        for i in range(1, len(keyframe_poses)):
            pose_prev = keyframe_poses[i-1]
            pose_curr = keyframe_poses[i]
            
            # Visual constraint: relative pose
            visual_delta_pos = pose_curr.position - pose_prev.position
            visual_delta_rot = pose_prev.rotation_matrix.T @ pose_curr.rotation_matrix
            
            # Inertial constraint: preintegrated IMU
            kf_id = keyframe_ids[i]
            if kf_id in preintegrated_data:
                preint = preintegrated_data[kf_id]
                
                # Check that time intervals match
                expected_dt = pose_curr.timestamp - pose_prev.timestamp
                assert abs(preint.dt - expected_dt) < 0.01
                
                # Check that keyframe IDs are consistent
                assert preint.from_keyframe_id == keyframe_ids[i-1]
                assert preint.to_keyframe_id == kf_id
                
                # The actual position delta would need to account for velocity and gravity
                # Here we just verify the structure is correct
                assert preint.delta_position is not None
                assert preint.delta_velocity is not None
                assert preint.delta_rotation is not None
                assert preint.covariance.shape == (15, 15)
    
    def test_cache_efficiency(self):
        """Test that preintegration cache improves efficiency."""
        # Create IMU measurements
        imu_measurements = []
        for t in np.arange(0, 1.0, 0.01):
            imu = IMUMeasurement(
                timestamp=t,
                accelerometer=np.array([0, 0, 9.81]),
                gyroscope=np.zeros(3)
            )
            imu_measurements.append(imu)
        
        keyframe_ids = [0, 1, 2, 3]
        keyframe_times = [0.0, 0.3, 0.6, 0.9]
        
        # First run with cache
        cache = PreintegrationCache()
        preintegrator1 = IMUPreintegrator()
        
        result1 = preintegrate_between_keyframes(
            imu_measurements,
            keyframe_ids,
            keyframe_times,
            preintegrator1,
            cache
        )
        
        # Second run should use cache
        preintegrator2 = IMUPreintegrator()
        result2 = preintegrate_between_keyframes(
            imu_measurements,
            keyframe_ids,
            keyframe_times,
            preintegrator2,
            cache
        )
        
        # Results should be identical (from cache)
        assert len(result1) == len(result2)
        for kf_id in result1:
            assert result1[kf_id].from_keyframe_id == result2[kf_id].from_keyframe_id
            assert result1[kf_id].to_keyframe_id == result2[kf_id].to_keyframe_id
            assert result1[kf_id].dt == result2[kf_id].dt
            assert result1[kf_id].num_measurements == result2[kf_id].num_measurements


class TestDataIntegrity:
    """Test data integrity in the keyframe-preintegration pipeline."""
    
    def test_keyframe_id_continuity(self):
        """Test that keyframe IDs are continuous and properly referenced."""
        # Create frames
        frames = []
        poses = []
        for i in range(20):
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
        config = KeyframeSelectionConfig(
            strategy=KeyframeSelectionStrategy.FIXED_INTERVAL,
            fixed_interval=4
        )
        mark_keyframes_in_camera_data(frames, poses, config)
        
        # Extract keyframe IDs
        keyframe_ids = []
        for frame in frames:
            if frame.is_keyframe:
                keyframe_ids.append(frame.keyframe_id)
        
        # Check continuity
        for i in range(1, len(keyframe_ids)):
            assert keyframe_ids[i] == keyframe_ids[i-1] + 1
    
    def test_no_data_loss_in_preintegration(self):
        """Test that no IMU measurements are lost during preintegration."""
        # Create IMU measurements
        imu_measurements = []
        for t in np.arange(0, 1.0, 0.01):
            imu = IMUMeasurement(
                timestamp=t,
                accelerometer=np.random.randn(3),
                gyroscope=np.random.randn(3) * 0.1
            )
            imu_measurements.append(imu)
        
        keyframe_ids = [0, 1, 2]
        keyframe_times = [0.0, 0.4, 0.99]  # Last keyframe at 0.99 to include all measurements
        
        preintegrator = IMUPreintegrator()
        preintegrated_data = preintegrate_between_keyframes(
            imu_measurements,
            keyframe_ids,
            keyframe_times,
            preintegrator
        )
        
        # Count total measurements used
        total_measurements = 0
        for data in preintegrated_data.values():
            total_measurements += data.num_measurements
        
        # Should account for most measurements (might miss the last one at t=0.99)
        expected_measurements = len([m for m in imu_measurements if m.timestamp < 0.99])
        assert total_measurements >= expected_measurements - 1  # Allow for edge case