"""
Tests for sensor models (camera and IMU).
"""

import pytest
import numpy as np

from src.simulation.camera_model import (
    PinholeCamera, CameraViewConfig,
    generate_camera_observations
)
from src.simulation.imu_model import (
    IMUModel, IMUNoiseConfig
)
from src.common.data_structures import (
    CameraCalibration, CameraIntrinsics, CameraExtrinsics, CameraModel,
    IMUCalibration,
    Pose, Map, Landmark,
    Trajectory, TrajectoryState
)
from src.simulation.trajectory_generator import CircleTrajectory, TrajectoryParams


class TestPinholeCamera:
    """Test pinhole camera model."""
    
    def setup_method(self):
        """Setup test camera."""
        # Create camera calibration
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
        
        # Camera looking forward from body
        extrinsics = CameraExtrinsics(
            B_T_C=np.eye(4)  # Camera at body origin
        )
        
        self.calibration = CameraCalibration(
            camera_id="cam0",
            intrinsics=intrinsics,
            extrinsics=extrinsics
        )
        
        self.camera = PinholeCamera(self.calibration)
    
    def test_project_point(self):
        """Test point projection."""
        # Camera at origin looking forward
        W_T_B = np.eye(4)
        
        # Point directly in front of camera
        point_world = np.array([0, 0, 5])  # 5 meters in front
        
        pixel, depth = self.camera.project_point(point_world, W_T_B)
        
        assert pixel is not None
        assert np.isclose(pixel.u, 320.0)  # Should be at principal point
        assert np.isclose(pixel.v, 240.0)
        assert np.isclose(depth, 5.0)
    
    def test_visibility_checking(self):
        """Test visibility checking."""
        W_T_B = np.eye(4)
        
        # Visible point
        visible_point = np.array([0, 0, 5])
        assert self.camera.is_visible(visible_point, W_T_B)
        
        # Point behind camera
        behind_point = np.array([0, 0, -5])
        assert not self.camera.is_visible(behind_point, W_T_B)
        
        # Point outside FOV
        far_side_point = np.array([100, 0, 5])
        assert not self.camera.is_visible(far_side_point, W_T_B)
    
    def test_frustum_culling(self):
        """Test frustum culling."""
        W_T_B = np.eye(4)
        
        # Create points
        points = np.array([
            [0, 0, 5],      # In front, visible
            [0, 0, -5],     # Behind camera
            [100, 0, 5],    # Far to the side
            [0.5, 0.5, 5],  # In front, visible
            [0, 0, 100],    # Too far
        ])
        
        # Configure max depth
        self.camera.view_config.max_depth = 50.0
        
        mask = self.camera.frustum_culling(points, W_T_B)
        
        assert mask[0] == True   # In front
        assert mask[1] == False  # Behind
        assert mask[2] == False  # Outside FOV
        assert mask[3] == True   # In front
        assert mask[4] == False  # Too far
    
    def test_field_of_view(self):
        """Test FOV calculation."""
        h_fov, v_fov = self.camera.compute_field_of_view()
        
        # Expected FOV based on focal length and image size
        # FOV = 2 * atan(size / (2 * f))
        expected_h_fov = 2 * np.arctan(640 / (2 * 500))
        expected_v_fov = 2 * np.arctan(480 / (2 * 500))
        
        assert np.isclose(h_fov, expected_h_fov)
        assert np.isclose(v_fov, expected_v_fov)
    
    def test_generate_observations(self):
        """Test observation generation."""
        # Create landmarks
        map_data = Map()
        for i in range(5):
            landmark = Landmark(
                id=i,
                position=np.array([i-2, 0, 10])  # Line of landmarks
            )
            map_data.add_landmark(landmark)
        
        # Camera pose
        pose = Pose(
            timestamp=0.0,
            position=np.array([0, 0, 0]),
            quaternion=np.array([1, 0, 0, 0])
        )
        
        # Generate observations
        frame = generate_camera_observations(
            self.camera,
            map_data,
            pose,
            timestamp=0.0,
            camera_id="cam0"
        )
        
        assert len(frame.observations) > 0
        assert frame.timestamp == 0.0
        assert frame.camera_id == "cam0"


class TestIMUModel:
    """Test IMU measurement model."""
    
    def setup_method(self):
        """Setup test IMU."""
        self.calibration = IMUCalibration(
            imu_id="imu0",
            accelerometer_noise_density=0.01,
            accelerometer_random_walk=0.001,
            gyroscope_noise_density=0.001,
            gyroscope_random_walk=0.0001,
            rate=100.0  # 100 Hz
        )
        
        self.noise_config = IMUNoiseConfig(
            accel_noise_density=0.01,
            gyro_noise_density=0.001,
            gravity_magnitude=9.81,
            seed=42
        )
        
        self.imu = IMUModel(self.calibration, self.noise_config)
    
    def test_perfect_measurements_stationary(self):
        """Test perfect IMU measurements for stationary trajectory."""
        # Create stationary trajectory
        trajectory = Trajectory()
        for t in np.arange(0, 1.0, 0.01):  # 1 second
            pose = Pose(
                timestamp=t,
                position=np.array([0, 0, 0]),
                quaternion=np.array([1, 0, 0, 0])  # Identity
            )
            state = TrajectoryState(
                pose=pose,
                velocity=np.zeros(3),
                angular_velocity=np.zeros(3)
            )
            trajectory.add_state(state)
        
        # Generate measurements
        imu_data = self.imu.generate_perfect_measurements(trajectory)
        
        assert len(imu_data.measurements) > 0
        
        # For stationary case, should measure only gravity
        for meas in imu_data.measurements:
            # Accelerometer should measure gravity in body frame
            # With identity rotation, gravity is [0, 0, -9.81] in world
            # So accelerometer measures [0, 0, 9.81] (opposite)
            assert np.allclose(meas.accelerometer, [0, 0, 9.81], atol=1e-6)
            # Gyroscope should be zero
            assert np.allclose(meas.gyroscope, [0, 0, 0], atol=1e-6)
    
    def test_perfect_measurements_rotating(self):
        """Test perfect IMU measurements for rotating trajectory."""
        # Create trajectory with constant angular velocity
        trajectory = Trajectory()
        angular_vel = np.array([0, 0, 1.0])  # 1 rad/s around z
        
        for t in np.arange(0, 1.0, 0.01):
            angle = t * 1.0  # Rotation angle
            # Rotation around z-axis
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])
            
            from src.utils.math_utils import rotation_matrix_to_quaternion
            quat = rotation_matrix_to_quaternion(R)
            
            pose = Pose(
                timestamp=t,
                position=np.array([0, 0, 0]),
                quaternion=quat
            )
            state = TrajectoryState(
                pose=pose,
                velocity=np.zeros(3),
                angular_velocity=angular_vel
            )
            trajectory.add_state(state)
        
        # Generate measurements
        imu_data = self.imu.generate_perfect_measurements(trajectory)
        
        # Check gyroscope measures the angular velocity
        for meas in imu_data.measurements:
            assert np.allclose(meas.gyroscope, angular_vel, atol=0.01)
    
    def test_noisy_measurements(self):
        """Test noisy IMU measurement generation."""
        # Create simple trajectory
        traj_params = TrajectoryParams(duration=0.5, rate=100.0)
        circle_gen = CircleTrajectory(radius=1.0, height=1.0, params=traj_params)
        trajectory = circle_gen.generate()
        
        # Generate noisy measurements
        imu_data = self.imu.generate_noisy_measurements(trajectory)
        
        assert len(imu_data.measurements) > 0
        
        # Collect all measurements
        accels = np.array([m.accelerometer for m in imu_data.measurements])
        gyros = np.array([m.gyroscope for m in imu_data.measurements])
        
        # Check that measurements have noise (variance > 0)
        assert np.std(accels[:, 0]) > 0
        assert np.std(gyros[:, 0]) > 0
        
        # But should be bounded
        assert np.all(np.abs(accels) < 100)  # Reasonable bounds
        assert np.all(np.abs(gyros) < 10)
    
    def test_measurement_rate(self):
        """Test that IMU generates at correct rate."""
        # Create 1 second trajectory
        trajectory = Trajectory()
        for t in [0.0, 0.5, 1.0]:
            pose = Pose(
                timestamp=t,
                position=np.array([t, 0, 0]),
                quaternion=np.array([1, 0, 0, 0])
            )
            state = TrajectoryState(pose=pose)
            trajectory.add_state(state)
        
        # Generate at 100 Hz
        imu_data = self.imu.generate_perfect_measurements(trajectory)
        
        # Should have approximately 100 measurements for 1 second
        expected_samples = int(1.0 * self.calibration.rate)
        assert abs(len(imu_data.measurements) - expected_samples) <= 1
        
        # Check timestamps
        dt = 1.0 / self.calibration.rate
        for i in range(1, len(imu_data.measurements)):
            time_diff = imu_data.measurements[i].timestamp - imu_data.measurements[i-1].timestamp
            assert np.isclose(time_diff, dt, atol=1e-6)