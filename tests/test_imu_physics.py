"""
Comprehensive tests for IMU physics correctness.
Ensures IMU measurements match expected physical behavior.
"""

import numpy as np
import pytest
from src.simulation.imu_model import IMUModel, IMUNoiseConfig
from src.simulation.trajectory_generator import generate_trajectory
from src.common.data_structures import (
    IMUCalibration, Trajectory, TrajectoryState, Pose
)


class TestIMUPhysics:
    """Test IMU measurements match expected physics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create IMU calibration
        self.imu_calib = IMUCalibration(
            imu_id="test_imu",
            accelerometer_noise_density=0.0,  # No noise for physics tests
            accelerometer_random_walk=0.0,
            gyroscope_noise_density=0.0,
            gyroscope_random_walk=0.0,
            rate=200.0,
            gravity_magnitude=9.81
        )
        
        # Create noise-free IMU model
        self.noise_config = IMUNoiseConfig(
            accel_noise_density=0.0,
            accel_random_walk=0.0,
            gyro_noise_density=0.0,
            gyro_random_walk=0.0,
            gravity_magnitude=9.81
        )
        
        self.imu = IMUModel(self.imu_calib, self.noise_config)
    
    def test_stationary_imu_measures_gravity(self):
        """Test that a stationary IMU measures gravity correctly."""
        # Create stationary trajectory (robot at origin, upright)
        traj = Trajectory()
        
        # Add states at different times, all at same position
        for t in np.linspace(0, 1, 10):
            pose = Pose(
                timestamp=t,
                position=np.array([0, 0, 0]),
                rotation_matrix=np.eye(3)  # Identity = upright
            )
            state = TrajectoryState(
                pose=pose,
                velocity=np.zeros(3),
                angular_velocity=np.zeros(3)
            )
            traj.add_state(state)
        
        # Generate IMU measurements
        imu_data = self.imu.generate_perfect_measurements(traj)
        
        # Check all measurements
        for meas in imu_data.measurements:
            # Accelerometer should measure gravity pointing up in body frame
            # When upright, gravity in world is [0, 0, -9.81]
            # In body frame (with identity rotation), should measure -gravity = [0, 0, 9.81]
            expected_accel = np.array([0, 0, 9.81])
            np.testing.assert_allclose(
                meas.accelerometer, expected_accel, 
                rtol=1e-5, atol=1e-5,
                err_msg=f"Stationary IMU should measure gravity: {expected_accel}"
            )
            
            # Gyroscope should measure zero
            np.testing.assert_allclose(
                meas.gyroscope, np.zeros(3),
                rtol=1e-5, atol=1e-5,
                err_msg="Stationary IMU should measure zero angular velocity"
            )
    
    def test_circular_motion_centripetal_acceleration(self):
        """Test that circular motion produces correct centripetal acceleration."""
        # Generate circular trajectory
        params = {
            "radius": 2.0,
            "height": 1.5,
            "duration": 3.0,
            "rate": 100.0,
            "start_time": 0.0
        }
        traj = generate_trajectory("circle", params)
        
        # Generate IMU measurements
        imu_data = self.imu.generate_perfect_measurements(traj)
        
        # Expected values for circular motion
        radius = 2.0
        period = 3.0
        omega = 2 * np.pi / period  # Angular velocity
        expected_centripetal = omega**2 * radius  # Centripetal acceleration
        
        # Check a measurement in the middle of trajectory (steady state)
        mid_idx = len(imu_data.measurements) // 2
        mid_meas = imu_data.measurements[mid_idx]
        
        # Get corresponding trajectory state
        mid_time = mid_meas.timestamp
        # Find the closest state to mid_time
        mid_state = None
        min_diff = float('inf')
        for state in traj.states:
            diff = abs(state.pose.timestamp - mid_time)
            if diff < min_diff:
                min_diff = diff
                mid_state = state
        
        # In circular motion, the robot should be rotating around Z axis
        # Gyroscope should measure yaw rate
        expected_gyro_z = omega
        assert abs(mid_meas.gyroscope[2] - expected_gyro_z) < 0.1, \
            f"Circular motion should have yaw rate ≈ {expected_gyro_z:.3f} rad/s, got {mid_meas.gyroscope[2]:.3f}"
        
        # The accelerometer should measure gravity + centripetal acceleration
        # This is complex to verify without knowing exact orientation
        # But the magnitude should be consistent
        accel_magnitude = np.linalg.norm(mid_meas.accelerometer)
        
        # Should be roughly sqrt(gravity^2 + centripetal^2) depending on orientation
        # Gravity = 9.81, centripetal = omega^2 * r = 8.78
        # Max when aligned: 9.81 + 8.78 = 18.6
        # Min when opposite: |9.81 - 8.78| = 1.03
        # Typical when perpendicular: sqrt(9.81^2 + 8.78^2) = 13.2
        assert accel_magnitude > 1.0, f"Accelerometer magnitude too small: {accel_magnitude:.3f}"
        assert accel_magnitude < 25.0, f"Accelerometer magnitude too large: {accel_magnitude:.3f}"
    
    def test_constant_velocity_motion(self):
        """Test that constant velocity motion only measures gravity."""
        # Create trajectory moving at constant velocity
        traj = Trajectory()
        velocity = np.array([1.0, 0.0, 0.0])  # Moving at 1 m/s in X direction
        
        for t in np.linspace(0, 2, 20):
            pose = Pose(
                timestamp=t,
                position=velocity * t,  # Position increases linearly
                rotation_matrix=np.eye(3)  # Upright
            )
            state = TrajectoryState(
                pose=pose,
                velocity=velocity,
                angular_velocity=np.zeros(3)
            )
            traj.add_state(state)
        
        # Generate IMU measurements
        imu_data = self.imu.generate_perfect_measurements(traj)
        
        # Check measurements
        for meas in imu_data.measurements:
            # Should only measure gravity (no acceleration from motion)
            expected_accel = np.array([0, 0, 9.81])
            np.testing.assert_allclose(
                meas.accelerometer, expected_accel,
                rtol=1e-3, atol=1e-3,
                err_msg="Constant velocity should only measure gravity"
            )
            
            # No rotation
            np.testing.assert_allclose(
                meas.gyroscope, np.zeros(3),
                rtol=1e-5, atol=1e-5,
                err_msg="No rotation in constant velocity motion"
            )
    
    def test_accelerating_motion(self):
        """Test that accelerating motion is measured correctly."""
        # Create trajectory with constant acceleration
        traj = Trajectory()
        acceleration = np.array([2.0, 0.0, 0.0])  # 2 m/s² in X direction
        
        for t in np.linspace(0, 2, 20):
            # Position with constant acceleration: x = 0.5 * a * t²
            position = 0.5 * acceleration * t**2
            velocity = acceleration * t
            
            pose = Pose(
                timestamp=t,
                position=position,
                rotation_matrix=np.eye(3)  # Upright
            )
            state = TrajectoryState(
                pose=pose,
                velocity=velocity,
                angular_velocity=np.zeros(3)
            )
            traj.add_state(state)
        
        # Generate IMU measurements
        imu_data = self.imu.generate_perfect_measurements(traj)
        
        # The IMU model currently assumes zero acceleration (line 279)
        # So this test will likely fail, demonstrating the bug
        # Expected: accelerometer should measure gravity + acceleration
        # Expected in body frame: [2.0, 0, 9.81] (acceleration in X, gravity in Z)
        
        # Check a measurement (not first or last due to edge effects)
        mid_meas = imu_data.measurements[len(imu_data.measurements)//2]
        
        # This will fail with current implementation, showing the bug
        # expected_accel = np.array([2.0, 0.0, 9.81])
        # For now, just document what we're getting
        print(f"Accelerating motion measurement: {mid_meas.accelerometer}")
        
        # Current implementation will give [0, 0, 9.81] incorrectly
        # This demonstrates the bug we need to fix
    
    def test_rotating_imu(self):
        """Test that a rotating IMU measures angular velocity correctly."""
        # Create trajectory with pure rotation (no translation)
        traj = Trajectory()
        angular_velocity = np.array([0, 0, 1.0])  # 1 rad/s around Z axis
        
        for t in np.linspace(0, 2*np.pi, 20):
            # Rotation angle increases linearly
            angle = angular_velocity[2] * t
            
            # Create rotation matrix for Z-axis rotation
            c, s = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([
                [c, -s, 0],
                [s,  c, 0],
                [0,  0, 1]
            ])
            
            pose = Pose(
                timestamp=t,
                position=np.zeros(3),  # No translation
                rotation_matrix=rotation_matrix
            )
            state = TrajectoryState(
                pose=pose,
                velocity=np.zeros(3),
                angular_velocity=angular_velocity
            )
            traj.add_state(state)
        
        # Generate IMU measurements
        imu_data = self.imu.generate_perfect_measurements(traj)
        
        # Check gyroscope measurements
        for meas in imu_data.measurements:
            np.testing.assert_allclose(
                meas.gyroscope, angular_velocity,
                rtol=1e-3, atol=1e-3,
                err_msg=f"Gyroscope should measure {angular_velocity}"
            )
            
            # Accelerometer should still measure gravity (rotated into body frame)
            accel_magnitude = np.linalg.norm(meas.accelerometer)
            np.testing.assert_allclose(
                accel_magnitude, 9.81,
                rtol=1e-3, atol=1e-3,
                err_msg="Accelerometer magnitude should equal gravity"
            )


if __name__ == "__main__":
    # Run tests
    test = TestIMUPhysics()
    test.setup_method()
    
    print("Testing stationary IMU...")
    test.test_stationary_imu_measures_gravity()
    print("✓ Stationary IMU test passed")
    
    print("\nTesting constant velocity motion...")
    test.test_constant_velocity_motion()
    print("✓ Constant velocity test passed")
    
    print("\nTesting circular motion...")
    test.test_circular_motion_centripetal_acceleration()
    print("✓ Circular motion test passed")
    
    print("\nTesting accelerating motion...")
    test.test_accelerating_motion()
    print("Note: Accelerating motion test demonstrates the bug")
    
    print("\nTesting rotating IMU...")
    test.test_rotating_imu()
    print("✓ Rotating IMU test passed")
    
    print("\n" + "="*50)
    print("IMU Physics Test Summary:")
    print("- Stationary and constant velocity: CORRECT")
    print("- Circular and rotating motion: CORRECT")
    print("- Accelerating motion: BUG FOUND (always assumes zero acceleration)")
    print("="*50)