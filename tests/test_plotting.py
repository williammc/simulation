"""
Legacy plotting tests - kept for backward compatibility.
Most functionality has been moved to test_visualization.py
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from src.common.data_structures import (
    Trajectory, TrajectoryState, Pose,
    Map, Landmark,
    IMUData, IMUMeasurement,
    CameraData, CameraFrame, CameraObservation, ImagePoint,
    CameraCalibration, CameraIntrinsics, CameraExtrinsics, CameraModel
)


class TestLegacyPlotting:
    """Legacy plotting tests."""
    
    @pytest.fixture
    def sample_trajectory(self):
        """Create a sample trajectory for testing."""
        traj = Trajectory()
        for i in range(10):
            t = i * 0.1
            pose = Pose(
                timestamp=t,
                position=np.array([np.cos(t), np.sin(t), 0.5]),
                rotation_matrix=np.eye(3)
            )
            state = TrajectoryState(
                pose=pose,
                velocity=np.array([-np.sin(t), np.cos(t), 0])
            )
            traj.add_state(state)
        return traj
    
    def test_trajectory_creation(self, sample_trajectory):
        """Test that sample trajectory is created correctly."""
        assert len(sample_trajectory.states) == 10
        assert sample_trajectory.states[0].pose.timestamp == 0
        assert sample_trajectory.states[-1].pose.timestamp == pytest.approx(0.9, rel=1e-5)
    
    @pytest.fixture
    def sample_imu_data(self):
        """Create sample IMU data."""
        imu_data = IMUData()
        imu_data.imu_id = "imu0"
        imu_data.rate = 100.0
        for i in range(50):
            t = i * 0.01
            meas = IMUMeasurement(
                timestamp=t,
                accelerometer=np.array([0.1, 0.2, 9.81]),
                gyroscope=np.array([0.01, 0.02, 0.03])
            )
            imu_data.add_measurement(meas)
        return imu_data
    
    def test_imu_data_creation(self, sample_imu_data):
        """Test that sample IMU data is created correctly."""
        assert len(sample_imu_data.measurements) == 50
        assert sample_imu_data.imu_id == "imu0"
        assert sample_imu_data.rate == 100.0
    
    @pytest.fixture
    def sample_camera_data(self):
        """Create sample camera data."""
        camera_data = CameraData(camera_id="cam0", rate=30.0)
        
        for i in range(10):
            frame = CameraFrame(
                timestamp=i * 0.033,
                camera_id="cam0",
                observations=[
                    CameraObservation(
                        landmark_id=j,
                        pixel=ImagePoint(u=100 + j*10 + i*2, v=200 + j*5 - i)
                    ) for j in range(5)
                ]
            )
            camera_data.add_frame(frame)
        
        return camera_data
    
    def test_camera_data_creation(self, sample_camera_data):
        """Test that sample camera data is created correctly."""
        assert len(sample_camera_data.frames) == 10
        assert sample_camera_data.camera_id == "cam0"
        assert len(sample_camera_data.frames[0].observations) == 5
    
    @pytest.fixture
    def sample_landmarks(self):
        """Create sample landmarks."""
        map_data = Map(frame_id="world")
        for i in range(20):
            landmark = Landmark(
                id=i,
                position=np.random.randn(3) * 5
            )
            map_data.add_landmark(landmark)
        return map_data
    
    def test_landmarks_creation(self, sample_landmarks):
        """Test that sample landmarks are created correctly."""
        assert len(sample_landmarks.landmarks) == 20
        assert sample_landmarks.frame_id == "world"
        # Check that landmarks are stored in dict
        assert isinstance(sample_landmarks.landmarks, dict)
        assert 0 in sample_landmarks.landmarks
        assert 19 in sample_landmarks.landmarks