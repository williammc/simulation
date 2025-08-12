"""
Unit tests for data structures.
"""

import pytest
import numpy as np
from pathlib import Path

from src.common.data_structures import (
    IMUMeasurement, IMUData,
    ImagePoint, CameraObservation, CameraFrame, CameraData,
    Pose, TrajectoryState, Trajectory,
    Landmark, Map,
    CameraModel, CameraIntrinsics, CameraExtrinsics, CameraCalibration, IMUCalibration
)


class TestIMUDataStructures:
    """Test IMU-related data structures."""
    
    def test_imu_measurement_creation(self):
        """Test IMU measurement creation and validation."""
        meas = IMUMeasurement(
            timestamp=1.0,
            accelerometer=np.array([0, 0, 9.81]),
            gyroscope=np.array([0.1, 0.2, 0.3])
        )
        
        assert meas.timestamp == 1.0
        assert np.allclose(meas.accelerometer, [0, 0, 9.81])
        assert np.allclose(meas.gyroscope, [0.1, 0.2, 0.3])
    
    def test_imu_measurement_invalid_dimensions(self):
        """Test that invalid dimensions raise errors."""
        with pytest.raises(ValueError, match="Accelerometer must be 3D"):
            IMUMeasurement(
                timestamp=1.0,
                accelerometer=np.array([0, 0]),  # 2D instead of 3D
                gyroscope=np.array([0, 0, 0])
            )
        
        with pytest.raises(ValueError, match="Gyroscope must be 3D"):
            IMUMeasurement(
                timestamp=1.0,
                accelerometer=np.array([0, 0, 0]),
                gyroscope=np.array([0, 0, 0, 0])  # 4D instead of 3D
            )
    
    def test_imu_data_collection(self):
        """Test IMU data collection."""
        imu_data = IMUData(sensor_id="imu0", rate=200.0)
        
        # Add measurements
        for t in [0.0, 0.1, 0.2]:
            meas = IMUMeasurement(
                timestamp=t,
                accelerometer=np.array([0, 0, 9.81]),
                gyroscope=np.array([0, 0, 0])
            )
            imu_data.add_measurement(meas)
        
        assert len(imu_data.measurements) == 3
        assert imu_data.get_time_range() == (0.0, 0.2)
    
    def test_imu_data_chronological_order(self):
        """Test that measurements must be added in chronological order."""
        imu_data = IMUData()
        
        meas1 = IMUMeasurement(timestamp=1.0, accelerometer=np.zeros(3), gyroscope=np.zeros(3))
        meas2 = IMUMeasurement(timestamp=0.5, accelerometer=np.zeros(3), gyroscope=np.zeros(3))
        
        imu_data.add_measurement(meas1)
        
        with pytest.raises(ValueError, match="chronological order"):
            imu_data.add_measurement(meas2)


class TestCameraDataStructures:
    """Test camera-related data structures."""
    
    def test_image_point_creation(self):
        """Test image point creation."""
        point = ImagePoint(u=320.5, v=240.5)
        
        assert point.u == 320.5
        assert point.v == 240.5
        assert np.allclose(point.to_array(), [320.5, 240.5])
    
    def test_camera_observation(self):
        """Test camera observation creation."""
        obs = CameraObservation(
            landmark_id=42,
            pixel=ImagePoint(u=100, v=200),
            descriptor=np.random.randn(128)
        )
        
        assert obs.landmark_id == 42
        assert obs.pixel.u == 100
        assert obs.pixel.v == 200
        assert obs.descriptor is not None
        assert len(obs.descriptor) == 128
    
    def test_camera_frame(self):
        """Test camera frame with observations."""
        observations = [
            CameraObservation(
                landmark_id=i,
                pixel=ImagePoint(u=i*10, v=i*20)
            )
            for i in range(5)
        ]
        
        frame = CameraFrame(
            timestamp=1.0,
            camera_id="cam0",
            observations=observations,
            image_path="/path/to/image.png"
        )
        
        assert frame.timestamp == 1.0
        assert frame.camera_id == "cam0"
        assert len(frame.observations) == 5
        assert frame.image_path == "/path/to/image.png"
    
    def test_camera_data_collection(self):
        """Test camera data collection."""
        camera_data = CameraData(camera_id="cam0", rate=30.0)
        
        for t in [0.0, 0.033, 0.066]:
            frame = CameraFrame(
                timestamp=t,
                camera_id="cam0",
                observations=[]
            )
            camera_data.add_frame(frame)
        
        assert len(camera_data.frames) == 3
        assert camera_data.get_time_range() == (0.0, 0.066)


class TestTrajectoryDataStructures:
    """Test trajectory-related data structures."""
    
    def test_pose_creation(self):
        """Test pose creation and quaternion normalization."""
        pose = Pose(
            timestamp=1.0,
            position=np.array([1, 2, 3]),
            quaternion=np.array([2, 0, 0, 0])  # Will be normalized
        )
        
        assert pose.timestamp == 1.0
        assert np.allclose(pose.position, [1, 2, 3])
        assert np.allclose(pose.quaternion, [1, 0, 0, 0])  # Normalized
        assert np.allclose(np.linalg.norm(pose.quaternion), 1.0)
    
    def test_pose_to_matrix(self):
        """Test pose to transformation matrix conversion."""
        pose = Pose(
            timestamp=1.0,
            position=np.array([1, 2, 3]),
            quaternion=np.array([1, 0, 0, 0])  # Identity rotation
        )
        
        T = pose.to_matrix()
        
        assert T.shape == (4, 4)
        assert np.allclose(T[:3, :3], np.eye(3))  # Identity rotation
        assert np.allclose(T[:3, 3], [1, 2, 3])  # Position
        assert np.allclose(T[3, :], [0, 0, 0, 1])  # Homogeneous row
    
    def test_trajectory_state(self):
        """Test trajectory state with velocities."""
        pose = Pose(
            timestamp=1.0,
            position=np.array([1, 2, 3]),
            quaternion=np.array([1, 0, 0, 0])
        )
        
        state = TrajectoryState(
            pose=pose,
            velocity=np.array([0.5, 1.0, 0.0]),
            angular_velocity=np.array([0, 0, 0.1])
        )
        
        assert state.pose.timestamp == 1.0
        assert np.allclose(state.velocity, [0.5, 1.0, 0.0])
        assert np.allclose(state.angular_velocity, [0, 0, 0.1])
    
    def test_trajectory_collection(self):
        """Test trajectory collection and interpolation."""
        trajectory = Trajectory(frame_id="world")
        
        # Add states
        for t in [0.0, 1.0, 2.0]:
            pose = Pose(
                timestamp=t,
                position=np.array([t, 0, 0]),
                quaternion=np.array([1, 0, 0, 0])
            )
            state = TrajectoryState(pose=pose)
            trajectory.add_state(state)
        
        assert len(trajectory.states) == 3
        assert trajectory.get_time_range() == (0.0, 2.0)
        
        # Test interpolation
        interp_pose = trajectory.get_pose_at_time(0.5)
        assert interp_pose is not None
        assert interp_pose.timestamp == 0.5
        assert np.allclose(interp_pose.position, [0.5, 0, 0])  # Linear interpolation


class TestLandmarkDataStructures:
    """Test landmark/map data structures."""
    
    def test_landmark_creation(self):
        """Test landmark creation."""
        landmark = Landmark(
            id=1,
            position=np.array([10, 20, 30]),
            descriptor=np.random.randn(256),
            covariance=np.eye(3) * 0.01
        )
        
        assert landmark.id == 1
        assert np.allclose(landmark.position, [10, 20, 30])
        assert landmark.descriptor is not None
        assert landmark.descriptor.shape == (256,)
        assert landmark.covariance is not None
        assert landmark.covariance.shape == (3, 3)
    
    def test_map_collection(self):
        """Test map landmark collection."""
        map_data = Map(frame_id="world")
        
        # Add landmarks
        for i in range(10):
            landmark = Landmark(
                id=i,
                position=np.array([i, i*2, i*3])
            )
            map_data.add_landmark(landmark)
        
        assert len(map_data.landmarks) == 10
        assert map_data.get_landmark(5) is not None
        assert map_data.get_landmark(5).id == 5
        
        # Test get all positions
        positions = map_data.get_positions()
        assert positions.shape == (10, 3)
        assert np.allclose(positions[5], [5, 10, 15])


class TestCalibrationDataStructures:
    """Test calibration data structures."""
    
    def test_camera_intrinsics(self):
        """Test camera intrinsics."""
        intrinsics = CameraIntrinsics(
            model=CameraModel.PINHOLE_RADTAN,
            width=640,
            height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            distortion=np.array([0.1, -0.2, 0.001, 0.002, 0.05])
        )
        
        assert intrinsics.model == CameraModel.PINHOLE_RADTAN
        assert intrinsics.width == 640
        assert intrinsics.height == 480
        
        # Test simple projection
        point_3d = np.array([0, 0, 1])  # 1 meter in front
        pixel = intrinsics.project(point_3d)
        assert pixel is not None
        assert np.allclose(pixel, [320, 240])  # Principal point
    
    def test_camera_extrinsics(self):
        """Test camera extrinsics."""
        B_T_C = np.eye(4)
        B_T_C[:3, 3] = [0.1, 0.0, 0.05]  # Camera offset from body
        
        extrinsics = CameraExtrinsics(B_T_C=B_T_C)
        
        assert extrinsics.B_T_C.shape == (4, 4)
        assert np.allclose(extrinsics.B_T_C[:3, 3], [0.1, 0.0, 0.05])
    
    def test_imu_calibration(self):
        """Test IMU calibration."""
        calib = IMUCalibration(
            imu_id="imu0",
            accelerometer_noise_density=0.01,
            accelerometer_random_walk=0.001,
            gyroscope_noise_density=0.0001,
            gyroscope_random_walk=0.00001,
            rate=200.0
        )
        
        assert calib.imu_id == "imu0"
        assert calib.rate == 200.0
        assert calib.accelerometer_noise_density == 0.01


class TestSerialization:
    """Test to_dict and from_dict methods."""
    
    def test_imu_measurement_serialization(self):
        """Test IMU measurement serialization."""
        meas = IMUMeasurement(
            timestamp=1.0,
            accelerometer=np.array([0, 0, 9.81]),
            gyroscope=np.array([0.1, 0.2, 0.3])
        )
        
        data = meas.to_dict()
        meas2 = IMUMeasurement.from_dict(data)
        
        assert meas2.timestamp == meas.timestamp
        assert np.allclose(meas2.accelerometer, meas.accelerometer)
        assert np.allclose(meas2.gyroscope, meas.gyroscope)
    
    def test_pose_serialization(self):
        """Test pose serialization."""
        pose = Pose(
            timestamp=1.0,
            position=np.array([1, 2, 3]),
            quaternion=np.array([0.7071, 0, 0, 0.7071])
        )
        
        data = pose.to_dict()
        pose2 = Pose.from_dict(data)
        
        assert pose2.timestamp == pose.timestamp
        assert np.allclose(pose2.position, pose.position)
        assert np.allclose(pose2.quaternion, pose.quaternion, atol=1e-4)
    
    def test_landmark_serialization(self):
        """Test landmark serialization."""
        landmark = Landmark(
            id=42,
            position=np.array([10, 20, 30]),
            descriptor=np.random.randn(128),
            covariance=np.eye(3) * 0.01
        )
        
        data = landmark.to_dict()
        landmark2 = Landmark.from_dict(data)
        
        assert landmark2.id == landmark.id
        assert np.allclose(landmark2.position, landmark.position)
        assert np.allclose(landmark2.descriptor, landmark.descriptor)
        assert np.allclose(landmark2.covariance, landmark.covariance)