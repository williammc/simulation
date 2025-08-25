"""
Simplified integration tests for keyframe-preintegration pipeline.
"""

import numpy as np
import pytest
from typing import List

from src.common.config import (
    KeyframeSelectionConfig, KeyframeSelectionStrategy,
    EKFConfig, SWBAConfig
)
from src.common.data_structures import (
    Pose, CameraFrame, CameraObservation, ImagePoint,
    IMUMeasurement, Landmark, Map,
    CameraCalibration, CameraIntrinsics, CameraExtrinsics, CameraModel,
    IMUCalibration, PreintegratedIMUData, Trajectory, TrajectoryState
)
from src.estimation.legacy.ekf_slam import EKFSlam
from src.estimation.legacy.swba_slam import SlidingWindowBA
from src.simulation.keyframe_selector import mark_keyframes_in_camera_data
from src.utils.preintegration_utils import preintegrate_between_keyframes
from src.estimation.imu_integration import IMUPreintegrator


class TestSimplifiedIntegration:
    """Simplified integration tests without external dependencies."""
    
    @pytest.fixture
    def camera_calibration(self):
        """Create test camera calibration."""
        intrinsics = CameraIntrinsics(
            model=CameraModel.PINHOLE,
            width=640,
            height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            distortion=np.zeros(5)
        )
        extrinsics = CameraExtrinsics(B_T_C=np.eye(4))
        return CameraCalibration(
            camera_id="cam0",
            intrinsics=intrinsics,
            extrinsics=extrinsics
        )
    
    @pytest.fixture
    def imu_calibration(self):
        """Create test IMU calibration."""
        return IMUCalibration(
            imu_id="imu0",
            accelerometer_noise_density=0.01,
            accelerometer_random_walk=0.001,
            gyroscope_noise_density=0.001,
            gyroscope_random_walk=0.0001,
            rate=200.0
        )
    
    def test_keyframe_selection_and_preintegration(self):
        """Test keyframe selection followed by IMU preintegration."""
        # Create synthetic trajectory
        timestamps = np.arange(0, 2.0, 0.1)
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
                position=np.array([t * 0.5, 0, 0]),
                rotation_matrix=np.eye(3)
            )
            poses.append(pose)
        
        # Select keyframes
        config = KeyframeSelectionConfig(
            strategy=KeyframeSelectionStrategy.FIXED_INTERVAL,
            fixed_interval=5
        )
        mark_keyframes_in_camera_data(frames, poses, config)
        
        # Verify keyframes
        keyframes = [f for f in frames if f.is_keyframe]
        assert len(keyframes) == 4  # At indices 0, 5, 10, 15
        
        # Create IMU measurements
        imu_measurements = []
        for t in np.arange(0, 2.0, 0.01):  # 100Hz IMU
            imu = IMUMeasurement(
                timestamp=t,
                accelerometer=np.array([0.5, 0, 9.81]),
                gyroscope=np.zeros(3)
            )
            imu_measurements.append(imu)
        
        # Preintegrate between keyframes
        keyframe_ids = [f.keyframe_id for f in frames if f.is_keyframe]
        keyframe_times = [f.timestamp for f in frames if f.is_keyframe]
        
        preintegrator = IMUPreintegrator()
        preintegrated_data = preintegrate_between_keyframes(
            imu_measurements,
            keyframe_ids,
            keyframe_times,
            preintegrator
        )
        
        # Verify preintegration
        assert len(preintegrated_data) == 3  # 3 intervals between 4 keyframes
        for data in preintegrated_data.values():
            assert data.dt > 0
            assert data.num_measurements > 0
    
    def test_ekf_with_keyframes(self, camera_calibration):
        """Test EKF processing only keyframes."""
        config = EKFConfig(
            use_keyframes_only=True,
            use_preintegrated_imu=True
        )
        
        ekf = EKFSlam(config, camera_calibration)
        
        # Initialize
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            rotation_matrix=np.eye(3)
        )
        ekf.initialize(initial_pose)
        
        # Create map
        map_data = Map()
        map_data.add_landmark(Landmark(id=0, position=np.array([2, 0, 3])))
        
        # Process keyframe
        frame = CameraFrame(
            timestamp=0.5,
            camera_id="cam0",
            observations=[
                CameraObservation(
                    landmark_id=0,
                    pixel=ImagePoint(u=320, v=240)
                )
            ]
        )
        frame.is_keyframe = True
        frame.keyframe_id = 1
        
        # Add preintegrated IMU
        frame.preintegrated_imu = PreintegratedIMUData(
            from_keyframe_id=0,
            to_keyframe_id=1,
            delta_position=np.array([0.25, 0, 0]),
            delta_velocity=np.array([0.5, 0, 0]),
            delta_rotation=np.eye(3),
            dt=0.5,
            covariance=np.eye(15) * 0.01,
            num_measurements=50
        )
        
        # Process
        if frame.preintegrated_imu:
            ekf.predict(frame.preintegrated_imu)
        ekf.update(frame, map_data)
        
        # Verify state was updated
        result = ekf.get_result()
        assert result.trajectory is not None
        assert len(result.trajectory.states) > 0
    
    def test_swba_with_keyframes(self, camera_calibration, imu_calibration):
        """Test SWBA processing keyframes with preintegrated IMU."""
        config = SWBAConfig(
            window_size=3,
            use_preintegrated_imu=True,
            use_keyframes_only=True
        )
        
        swba = SlidingWindowBA(config, camera_calibration, imu_calibration)
        
        # Initialize
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            rotation_matrix=np.eye(3)
        )
        swba.initialize(initial_pose)
        
        # Create map
        map_data = Map()
        map_data.add_landmark(Landmark(id=0, position=np.array([2, 0, 3])))
        map_data.add_landmark(Landmark(id=1, position=np.array([2, 1, 3])))
        
        # Process multiple keyframes
        for i in range(1, 4):
            frame = CameraFrame(
                timestamp=i * 0.5,
                camera_id="cam0",
                observations=[
                    CameraObservation(
                        landmark_id=0,
                        pixel=ImagePoint(u=320, v=240)
                    ),
                    CameraObservation(
                        landmark_id=1,
                        pixel=ImagePoint(u=340, v=240)
                    )
                ]
            )
            frame.is_keyframe = True
            frame.keyframe_id = i
            
            # Add preintegrated IMU
            frame.preintegrated_imu = PreintegratedIMUData(
                from_keyframe_id=i-1,
                to_keyframe_id=i,
                delta_position=np.array([0.25, 0.05, 0]),
                delta_velocity=np.array([0.5, 0.1, 0]),
                delta_rotation=np.eye(3),
                dt=0.5,
                covariance=np.eye(15) * 0.01,
                num_measurements=50
            )
            
            # Update state
            swba.current_state.position = np.array([i*0.25, i*0.05, 0])
            swba.current_state.timestamp = i * 0.5
            
            # Process
            swba.update(frame, map_data)
        
        # Verify optimization ran
        assert swba.num_optimizations > 0
        
        # Get result
        result = swba.get_result()
        assert result.trajectory is not None
        assert len(result.trajectory.states) > 0
    
    def test_pipeline_consistency(self, camera_calibration, imu_calibration):
        """Test that the pipeline maintains consistency."""
        # Create frames and poses
        frames = []
        poses = []
        for i in range(10):
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
            fixed_interval=3
        )
        mark_keyframes_in_camera_data(frames, poses, config)
        
        # Get keyframe info
        keyframe_indices = [i for i, f in enumerate(frames) if f.is_keyframe]
        keyframe_ids = [f.keyframe_id for f in frames if f.is_keyframe]
        keyframe_times = [f.timestamp for f in frames if f.is_keyframe]
        
        # Create IMU measurements
        imu_measurements = []
        for t in np.arange(0, 1.0, 0.01):
            imu = IMUMeasurement(
                timestamp=t,
                accelerometer=np.array([0.1, 0, 9.81]),
                gyroscope=np.zeros(3)
            )
            imu_measurements.append(imu)
        
        # Preintegrate
        preintegrator = IMUPreintegrator()
        preintegrated_data = preintegrate_between_keyframes(
            imu_measurements,
            keyframe_ids,
            keyframe_times,
            preintegrator
        )
        
        # Verify consistency
        assert len(keyframe_indices) == 4  # 0, 3, 6, 9
        assert len(preintegrated_data) == 3  # 3 intervals
        
        # Check that each preintegrated segment connects correct keyframes
        for to_id, data in preintegrated_data.items():
            assert data.to_keyframe_id == to_id
            assert data.from_keyframe_id == to_id - 1
            assert data.dt > 0
            assert data.num_measurements > 0