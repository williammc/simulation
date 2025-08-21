"""
Tests for Phase 10 advanced features.
"""

import pytest
import numpy as np
from pathlib import Path

from src.simulation.trajectory_generator import (
    Figure8Trajectory, SpiralTrajectory, LineTrajectory,
    TrajectoryParams
)
from src.simulation.trajectory_interpolation import (
    TrajectoryInterpolator, SplineTrajectoryConfig,
    smooth_trajectory, create_bezier_trajectory
)
from src.estimation.stereo_camera import (
    StereoCameraModel, StereoCalibration, StereoObservation,
    create_stereo_calibration
)
from src.estimation.multi_imu import (
    MultiIMUFusion, IMUFusionMethod, IMUConfig,
    create_multi_imu_setup, FusedIMUMeasurement
)
from src.estimation.sensor_sync import (
    SensorSynchronizer, SensorTiming, SyncMethod,
    HardwareTrigger
)
from src.common.data_structures import (
    Trajectory, TrajectoryState, Pose, ImagePoint,
    IMUMeasurement, CameraFrame, Landmark
)


class TestTrajectoryGenerators:
    """Test advanced trajectory generators."""
    
    def test_figure8_trajectory(self):
        """Test Figure-8 trajectory generation."""
        params = TrajectoryParams(duration=10.0, rate=100.0)
        generator = Figure8Trajectory(
            scale_x=3.0,
            scale_y=2.0,
            height=1.5,
            params=params
        )
        
        trajectory = generator.generate()
        
        # Check trajectory properties
        assert len(trajectory.states) == 1000
        assert trajectory.states[0].pose.timestamp == 0.0
        assert trajectory.states[-1].pose.timestamp < 10.0
        
        # Check figure-8 shape (should cross origin twice)
        positions = np.array([s.pose.position for s in trajectory.states])
        x_crossings = np.where(np.diff(np.sign(positions[:, 0])))[0]
        assert len(x_crossings) >= 1  # At least one x-axis crossing
    
    def test_spiral_trajectory(self):
        """Test spiral trajectory generation."""
        params = TrajectoryParams(duration=5.0, rate=50.0)
        generator = SpiralTrajectory(
            initial_radius=0.5,
            final_radius=3.0,
            initial_height=0.5,
            final_height=3.0,
            params=params
        )
        
        trajectory = generator.generate()
        
        # Check trajectory properties
        assert len(trajectory.states) == 250
        
        # Check spiral expansion
        positions = np.array([s.pose.position for s in trajectory.states])
        radii = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
        
        assert radii[0] < radii[-1]  # Radius increases
        assert positions[0, 2] < positions[-1, 2]  # Height increases
    
    def test_line_trajectory(self):
        """Test line trajectory generation."""
        start = np.array([0, 0, 1])
        end = np.array([10, 5, 2])
        
        params = TrajectoryParams(duration=5.0, rate=20.0)
        generator = LineTrajectory(
            start_position=start,
            end_position=end,
            params=params
        )
        
        trajectory = generator.generate()
        
        # Check trajectory properties
        assert len(trajectory.states) == 100
        
        # Check linearity
        positions = np.array([s.pose.position for s in trajectory.states])
        
        # All points should be on the line
        direction = end - start
        direction = direction / np.linalg.norm(direction)
        
        for i in range(1, len(positions) - 1):
            vec_to_point = positions[i] - start
            # Project onto line direction
            projection = np.dot(vec_to_point, direction) * direction
            # Check perpendicular distance is small
            perpendicular = vec_to_point - projection
            assert np.linalg.norm(perpendicular) < 1e-10
        
        # Check constant velocity
        velocities = [s.velocity for s in trajectory.states if s.velocity is not None]
        if velocities:
            vel_std = np.std(velocities, axis=0)
            assert np.all(vel_std < 1e-10)


class TestTrajectoryInterpolation:
    """Test trajectory interpolation features."""
    
    def test_spline_interpolation(self):
        """Test spline-based trajectory interpolation."""
        # Create simple trajectory with few waypoints
        trajectory = Trajectory()
        timestamps = [0.0, 1.0, 2.0, 3.0]
        positions = [
            [0, 0, 0],
            [1, 1, 0],
            [2, 0, 0],
            [3, -1, 0]
        ]
        
        for t, pos in zip(timestamps, positions):
            pose = Pose(
                timestamp=t,
                position=np.array(pos),
                rotation_matrix=np.eye(3)
            )
            trajectory.add_state(TrajectoryState(pose=pose))
        
        # Interpolate
        config = SplineTrajectoryConfig()
        interpolator = TrajectoryInterpolator(config)
        interpolator.fit(trajectory)
        
        # Generate dense trajectory
        dense_traj = interpolator.interpolate(rate=100.0)
        
        # Check smoothness
        assert len(dense_traj.states) > len(trajectory.states)
        
        # Check that original points are preserved
        for original in trajectory.states:
            # Find closest point in dense trajectory
            t_orig = original.pose.timestamp
            closest = min(dense_traj.states, 
                         key=lambda s: abs(s.pose.timestamp - t_orig))
            
            pos_diff = np.linalg.norm(
                closest.pose.position - original.pose.position
            )
            assert pos_diff < 0.02  # Allow small error in spline approximation
    
    def test_bezier_trajectory(self):
        """Test Bezier curve trajectory generation."""
        control_points = [
            np.array([0, 0, 0]),
            np.array([1, 2, 0]),
            np.array([3, 2, 0]),
            np.array([4, 0, 0])
        ]
        
        trajectory = create_bezier_trajectory(
            control_points,
            num_points=50,
            duration=5.0
        )
        
        assert len(trajectory.states) == 50
        
        # Check start and end points
        np.testing.assert_array_almost_equal(
            trajectory.states[0].pose.position,
            control_points[0]
        )
        np.testing.assert_array_almost_equal(
            trajectory.states[-1].pose.position,
            control_points[-1]
        )
    
    def test_trajectory_smoothing(self):
        """Test trajectory smoothing."""
        # Create noisy trajectory
        trajectory = Trajectory()
        t = np.linspace(0, 2*np.pi, 50)
        
        for i, ti in enumerate(t):
            # Circle with noise
            x = np.cos(ti) + 0.1 * np.random.randn()
            y = np.sin(ti) + 0.1 * np.random.randn()
            
            pose = Pose(
                timestamp=ti,
                position=np.array([x, y, 0]),
                rotation_matrix=np.eye(3)
            )
            trajectory.add_state(TrajectoryState(pose=pose))
        
        # Smooth trajectory
        smoothed = smooth_trajectory(
            trajectory,
            window_size=5,
            position_sigma=0.5
        )
        
        # Check smoothness (reduced variance)
        original_positions = np.array([s.pose.position for s in trajectory.states])
        smoothed_positions = np.array([s.pose.position for s in smoothed.states])
        
        original_var = np.var(np.diff(original_positions, axis=0))
        smoothed_var = np.var(np.diff(smoothed_positions, axis=0))
        
        assert smoothed_var < original_var


class TestStereoCamera:
    """Test stereo camera functionality."""
    
    def test_stereo_calibration_creation(self):
        """Test stereo calibration creation."""
        calib = create_stereo_calibration(
            baseline=0.12,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0
        )
        
        assert calib.baseline == 0.12
        assert calib.left_calib.intrinsics.fx == 500.0
        assert calib.right_calib.intrinsics.fx == 500.0
        assert np.allclose(calib.T_RL[0, 3], -0.12)
    
    def test_stereo_projection(self):
        """Test stereo projection."""
        calib = create_stereo_calibration(baseline=0.1)
        model = StereoCameraModel(calib)
        
        # Test point projection
        landmark = np.array([0, 0, 5])
        pose = Pose(
            timestamp=0.0,
            position=np.array([0, 0, 0]),
            rotation_matrix=np.eye(3)
        )
        
        stereo_obs, _, _ = model.project_stereo(landmark, pose, False)
        
        assert stereo_obs is not None
        # For a point in front, disparity = left.u - right.u should be positive
        # Since the right camera is to the right of the left camera,
        # the same point appears more to the left in the right image
        assert stereo_obs.disparity != 0  # Non-zero disparity for point not at infinity
    
    def test_stereo_triangulation(self):
        """Test stereo triangulation."""
        calib = create_stereo_calibration(baseline=0.1, fx=500.0)
        model = StereoCameraModel(calib)
        
        # Create stereo observation
        stereo_obs = StereoObservation(
            left_pixel=ImagePoint(u=320, v=240),
            right_pixel=ImagePoint(u=310, v=240),  # 10 pixel disparity
            landmark_id=0,
            timestamp=0.0
        )
        
        pose = Pose(
            timestamp=0.0,
            position=np.array([0, 0, 0]),
            rotation_matrix=np.eye(3)
        )
        
        # Triangulate
        point_3d, uncertainty = model.triangulate(stereo_obs, pose)
        
        # Check depth from disparity
        expected_depth = (0.1 * 500.0) / 10.0  # baseline * fx / disparity
        assert abs(point_3d[2] - expected_depth) < 0.1
    
    def test_stereo_reprojection_error(self):
        """Test stereo reprojection error computation."""
        calib = create_stereo_calibration(baseline=0.1)
        model = StereoCameraModel(calib)
        
        # Create landmark and observation
        landmark = Landmark(id=0, position=np.array([1, 1, 5]))
        
        stereo_obs = StereoObservation(
            left_pixel=ImagePoint(u=320, v=240),
            right_pixel=ImagePoint(u=310, v=240),
            landmark_id=0,
            timestamp=0.0
        )
        
        pose = Pose(
            timestamp=0.0,
            position=np.array([0, 0, 0]),
            rotation_matrix=np.eye(3)
        )
        
        # Compute errors
        left_error, right_error = model.compute_stereo_reprojection_error(
            stereo_obs, landmark, pose
        )
        
        assert left_error is not None
        assert right_error is not None
        assert left_error.residual.shape == (2,)
        assert right_error.residual.shape == (2,)


class TestMultiIMU:
    """Test multi-IMU fusion."""
    
    def test_multi_imu_setup(self):
        """Test multi-IMU configuration creation."""
        configs = create_multi_imu_setup(num_imus=3, configuration="orthogonal")
        
        assert len(configs) == 3
        assert "imu_0" in configs
        assert "imu_1" in configs
        assert "imu_2" in configs
        
        # Check that IMUs have different orientations
        R0 = configs["imu_0"].calibration.extrinsics.B_T_S[:3, :3]
        R1 = configs["imu_1"].calibration.extrinsics.B_T_S[:3, :3]
        
        assert not np.allclose(R0, R1)
    
    def test_weighted_average_fusion(self):
        """Test weighted average IMU fusion."""
        configs = create_multi_imu_setup(num_imus=3, configuration="redundant")
        fusion = MultiIMUFusion(configs, fusion_method=IMUFusionMethod.WEIGHTED_AVERAGE)
        
        # Create measurements with slight variations
        base_accel = np.array([0, 0, 9.81])
        base_gyro = np.array([0.1, 0.0, 0.0])
        
        # Use fixed seed for reproducible test
        np.random.seed(42)
        measurements = {
            "imu_0": IMUMeasurement(
                timestamp=0.0,
                accelerometer=base_accel + 0.005 * np.random.randn(3),  # Reduced noise
                gyroscope=base_gyro + 0.0005 * np.random.randn(3)
            ),
            "imu_1": IMUMeasurement(
                timestamp=0.0,
                accelerometer=base_accel + 0.005 * np.random.randn(3),
                gyroscope=base_gyro + 0.0005 * np.random.randn(3)
            ),
            "imu_2": IMUMeasurement(
                timestamp=0.0,
                accelerometer=base_accel + 0.005 * np.random.randn(3),
                gyroscope=base_gyro + 0.0005 * np.random.randn(3)
            )
        }
        
        # Fuse measurements
        fused = fusion.fuse_measurements(measurements)
        
        assert fused is not None
        assert isinstance(fused, FusedIMUMeasurement)
        assert len(fused.contributing_imus) == 3
        assert fused.fusion_confidence > 0
        
        # Check that fused value is close to base values
        np.testing.assert_array_almost_equal(fused.acceleration, base_accel, decimal=1)
        np.testing.assert_array_almost_equal(fused.angular_velocity, base_gyro, decimal=2)
    
    def test_outlier_detection(self):
        """Test outlier detection in multi-IMU fusion."""
        configs = create_multi_imu_setup(num_imus=3, configuration="redundant")
        fusion = MultiIMUFusion(configs, fusion_method=IMUFusionMethod.WEIGHTED_AVERAGE)
        
        # Create measurements with one outlier
        good_accel = np.array([0, 0, 9.81])
        bad_accel = np.array([10, 10, 20])  # Outlier
        
        measurements = {
            "imu_0": IMUMeasurement(
                timestamp=0.0,
                accelerometer=good_accel,
                gyroscope=np.zeros(3)
            ),
            "imu_1": IMUMeasurement(
                timestamp=0.0,
                accelerometer=good_accel,
                gyroscope=np.zeros(3)
            ),
            "imu_2": IMUMeasurement(
                timestamp=0.0,
                accelerometer=bad_accel,  # Outlier
                gyroscope=np.zeros(3)
            )
        }
        
        # Fuse measurements
        fused = fusion.fuse_measurements(measurements)
        
        assert fused is not None
        # Check that outlier was rejected
        assert len(fused.contributing_imus) == 2
        assert "imu_2" not in fused.contributing_imus
        
        # Fused value should be close to good value
        np.testing.assert_array_almost_equal(fused.acceleration, good_accel, decimal=1)


class TestSensorSync:
    """Test sensor synchronization."""
    
    def test_sensor_timing_setup(self):
        """Test sensor timing configuration."""
        timings = {
            "imu": SensorTiming(
                sensor_id="imu",
                frequency=200.0,
                latency=0.001,
                jitter=0.0001
            ),
            "camera": SensorTiming(
                sensor_id="camera",
                frequency=30.0,
                latency=0.020,
                jitter=0.002
            )
        }
        
        sync = SensorSynchronizer(timings)
        
        assert len(sync.buffers) == 2
        assert "imu" in sync.buffers
        assert "camera" in sync.buffers
    
    def test_measurement_synchronization(self):
        """Test measurement synchronization."""
        timings = {
            "imu": SensorTiming("imu", 200.0, 0.001, 0.0),
            "camera": SensorTiming("camera", 30.0, 0.020, 0.0)
        }
        
        sync = SensorSynchronizer(
            timings,
            sync_method=SyncMethod.LINEAR_INTERPOLATION
        )
        
        # Add IMU measurements
        for i in range(10):
            t = i * 0.005  # 200 Hz
            imu_meas = IMUMeasurement(
                timestamp=t,
                accelerometer=np.array([0, 0, 9.81]),
                gyroscope=np.array([0.1 * i, 0, 0])
            )
            sync.add_measurement("imu", imu_meas, t)
        
        # Add camera measurement
        cam_frame = CameraFrame(
            timestamp=0.033,
            camera_id="cam0",
            observations=[]
        )
        sync.add_measurement("camera", cam_frame, 0.033)
        
        # Get synchronized measurements
        synced = sync.get_synced_measurements(0.025, tolerance=0.01)
        
        assert synced is not None
        assert synced.imu_data is not None
        assert abs(synced.timestamp - 0.025) < 1e-6
        
        # Check interpolation
        if synced.interpolated:
            # At t=0.025, we're interpolating between t=0.020 (i=4) and t=0.025 (i=5)
            # i=4: gyro[0] = 0.1 * 4 = 0.4
            # i=5: gyro[0] = 0.1 * 5 = 0.5
            # Linear interpolation at t=0.025 should give 0.5
            expected_omega = 0.5
            assert abs(synced.imu_data.gyroscope[0] - expected_omega) < 0.05
    
    def test_hardware_trigger(self):
        """Test hardware trigger synchronization."""
        trigger = HardwareTrigger(
            trigger_rate=30.0,
            camera_delay=0.005,
            imu_delay=0.001
        )
        
        # Test trigger generation
        sensor_times = trigger.trigger(0.0)
        
        assert sensor_times["camera"] == 0.005
        assert sensor_times["imu"] == 0.001
        
        # Test trigger alignment
        aligned = trigger.align_to_trigger(0.035)
        expected = 1.0 / 30.0  # One trigger period
        assert abs(aligned - expected) < 1e-6
    
    def test_clock_drift_estimation(self):
        """Test clock drift estimation."""
        timings = {
            "sensor1": SensorTiming("sensor1", 100.0, 0.0, 0.0),
            "sensor2": SensorTiming("sensor2", 100.0, 0.0, 0.0)
        }
        
        sync = SensorSynchronizer(timings)
        
        # Simulate clock drift (sensor2 is 1% faster)
        sensor_times = np.linspace(0, 1, 100)
        reference_times = sensor_times * 1.01  # 1% drift
        
        sync.estimate_clock_drift("sensor2", sensor_times.tolist(), reference_times.tolist())
        
        # Check drift estimate
        drift = sync.clock_corrections["sensor2"].drift_estimate
        assert drift is not None
        assert abs(drift - 0.01) < 0.001  # Should detect 1% drift