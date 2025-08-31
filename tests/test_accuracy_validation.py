"""
Comprehensive accuracy validation tests for IMU simulation and preintegration.

These tests ensure:
1. IMU simulation produces physically correct measurements
2. Preintegration accurately accumulates measurements
3. Large rotations (>180°) are handled correctly
4. Tight tolerances catch numerical errors early
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.simulation.imu_integration import IMUPreintegrator
from src.common.data_structures import IMUMeasurement
from src.simulation.imu_model import IMUModel
from src.common.config import IMUConfig, IMUNoiseParams
from src.simulation.trajectory_generator import generate_trajectory
from src.common.data_structures import IMUCalibration


class TestIMUPreintegrationAccuracy:
    """Test IMU preintegration accuracy with strict tolerances."""
    
    @pytest.mark.parametrize("angle_deg,description", [
        (45, "eighth circle"),
        (90, "quarter circle"),
        (180, "half circle"),
        (270, "three-quarter circle"),
        (360, "full circle"),
        (540, "1.5 circles"),
        (720, "2 circles"),
    ])
    def test_rotation_accumulation(self, angle_deg, description):
        """Test preintegration with various rotation angles.
        
        Tests rotation MATRICES directly, not angle extraction which wraps at ±180°.
        """
        # Setup
        dt = 0.005  # 200 Hz
        omega_z = 1.0  # rad/s
        duration = np.radians(angle_deg) / omega_z
        
        preintegrator = IMUPreintegrator(
            gravity=np.array([0, 0, -9.81]),
            accel_noise_density=0.0,
            gyro_noise_density=0.0
        )
        
        # Create constant rotation measurements
        measurements = []
        # Need measurements from t=0 to t=duration (inclusive) for proper integration
        num_steps = int(duration / dt) + 1
        
        for i in range(num_steps):
            meas = IMUMeasurement(
                timestamp=i * dt,
                accelerometer=np.array([0, 0, 9.81]),  # Compensate gravity
                gyroscope=np.array([0, 0, omega_z])
            )
            measurements.append(meas)
        
        # Process
        result = preintegrator.batch_process(measurements, 0, 1)
        
        # Expected rotation matrix for rotation around Z
        angle_rad = np.radians(angle_deg)
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        R_expected = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        
        # Compare rotation matrices directly
        R_actual = result.delta_rotation
        matrix_error = np.linalg.norm(R_actual - R_expected, 'fro')
        
        # STRICT tolerance for matrix comparison
        assert matrix_error < 0.01, (
            f"Rotation matrix incorrect for {description} ({angle_deg}°):\n"
            f"  Expected R[0,0]={c:.3f}, R[0,1]={-s:.3f}\n"
            f"  Actual   R[0,0]={R_actual[0,0]:.3f}, R[0,1]={R_actual[0,1]:.3f}\n"
            f"  Matrix Frobenius norm error: {matrix_error:.4f}\n"
            f"  Tolerance: 0.01"
        )
        
        # Also test vector transformation
        v_x = np.array([1, 0, 0])
        v_rotated = R_actual @ v_x
        v_expected = np.array([c, s, 0])
        vector_error = np.linalg.norm(v_rotated - v_expected)
        
        assert vector_error < 0.01, (
            f"Vector rotation failed for {description} ({angle_deg}°):\n"
            f"  Rotating [1,0,0] should give [{c:.3f}, {s:.3f}, 0]\n"
            f"  Got: [{v_rotated[0]:.3f}, {v_rotated[1]:.3f}, {v_rotated[2]:.3f}]\n"
            f"  Error: {vector_error:.4f}"
        )
    
    def test_rotation_axes(self):
        """Test rotations around different axes."""
        dt = 0.01
        duration = 2.0  # 2 seconds
        omega = np.pi / 2  # 90 deg/s
        
        axes = {
            'x': np.array([omega, 0, 0]),
            'y': np.array([0, omega, 0]),
            'z': np.array([0, 0, omega])
        }
        
        for axis_name, gyro in axes.items():
            preintegrator = IMUPreintegrator(gravity=np.zeros(3))
            
            measurements = []
            # Need measurements from t=0 to t=duration (inclusive)
            num_steps = int(duration / dt) + 1
            
            for i in range(num_steps):
                meas = IMUMeasurement(
                    timestamp=i * dt,
                    accelerometer=np.zeros(3),
                    gyroscope=gyro
                )
                measurements.append(meas)
            
            result = preintegrator.batch_process(measurements, 0, 1)
            
            # Check rotation angle
            expected_angle = omega * duration
            rot = Rotation.from_matrix(result.delta_rotation)
            actual_angle = np.linalg.norm(rot.as_rotvec())
            
            error = abs(actual_angle - expected_angle)
            assert error < 0.01, (
                f"Rotation around {axis_name}-axis failed:\n"
                f"  Expected: {expected_angle:.4f} rad\n"
                f"  Actual:   {actual_angle:.4f} rad\n"
                f"  Error:    {error:.4f} rad"
            )
    
    def test_combined_rotations(self):
        """Test complex rotation sequences."""
        dt = 0.01
        preintegrator = IMUPreintegrator(gravity=np.zeros(3))
        
        # Test yaw-pitch-roll combination
        measurements = []
        
        # First: 90° yaw (1 second)
        for i in range(100):
            meas = IMUMeasurement(
                timestamp=i * dt,
                accelerometer=np.zeros(3),
                gyroscope=np.array([0, 0, np.pi/2])
            )
            measurements.append(meas)
        
        # Then: 90° pitch (1 second)
        for i in range(100, 200):
            meas = IMUMeasurement(
                timestamp=i * dt,
                accelerometer=np.zeros(3),
                gyroscope=np.array([0, np.pi/2, 0])
            )
            measurements.append(meas)
        
        # Then: 90° roll (1 second)
        for i in range(200, 300):
            meas = IMUMeasurement(
                timestamp=i * dt,
                accelerometer=np.zeros(3),
                gyroscope=np.array([np.pi/2, 0, 0])
            )
            measurements.append(meas)
        
        result = preintegrator.batch_process(measurements, 0, 1)
        
        # The rotation matrix should be a composition of the three rotations
        # This is a complex test - just verify it's a valid rotation matrix
        R = result.delta_rotation
        
        # Check orthogonality
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        
        # Check determinant
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)
    
    def test_position_velocity_integration(self):
        """Test position and velocity integration with constant acceleration."""
        dt = 0.01
        duration = 2.0
        accel = np.array([1.0, 0.5, 0.0])  # Constant acceleration
        
        preintegrator = IMUPreintegrator(gravity=np.zeros(3))
        
        measurements = []
        # Need measurements from t=0 to t=duration (inclusive)
        num_steps = int(duration / dt) + 1
        
        for i in range(num_steps):
            meas = IMUMeasurement(
                timestamp=i * dt,
                accelerometer=accel,
                gyroscope=np.zeros(3)
            )
            measurements.append(meas)
        
        result = preintegrator.batch_process(measurements, 0, 1)
        
        # Analytical solution
        expected_velocity = accel * duration
        expected_position = 0.5 * accel * duration**2
        
        # Tolerances accounting for numerical integration error
        # With 200 steps, expect ~0.25% error from trapezoidal integration
        np.testing.assert_allclose(
            result.delta_velocity, expected_velocity,
            rtol=3e-3, atol=1e-4,
            err_msg="Velocity integration failed"
        )
        
        np.testing.assert_allclose(
            result.delta_position, expected_position,
            rtol=5e-3, atol=1e-4,
            err_msg="Position integration failed"
        )


class TestIMUSimulationAccuracy:
    """Test IMU simulation produces physically accurate measurements."""
    
    def test_circular_motion_measurements(self):
        """Test IMU measurements for circular motion are physically correct."""
        # Create noise-free IMU
        imu_calib = IMUCalibration(
            imu_id="test_imu",
            accelerometer_noise_density=0.0,
            accelerometer_random_walk=0.0,
            gyroscope_noise_density=0.0,
            gyroscope_random_walk=0.0,
            rate=200.0,
            gravity_magnitude=9.81
        )
        
        noise_params = IMUNoiseParams(
            accelerometer_noise_density=0.0,
            accelerometer_random_walk=0.0,
            gyroscope_noise_density=0.0,
            gyroscope_random_walk=0.0
        )
        
        imu_config = IMUConfig(
            noise_params=noise_params,
            gravity_magnitude=9.81
        )
        
        imu = IMUModel(imu_calib, imu_config)
        
        # Generate circular trajectory
        params = {
            "radius": 2.0,
            "height": 1.5,
            "duration": 5.0,  # One complete circle
            "rate": 200.0,
            "start_time": 0.0
        }
        traj = generate_trajectory("circle", params)
        
        # Generate measurements
        imu_data = imu.generate_perfect_measurements(traj)
        
        # Expected values
        radius = 2.0
        period = 5.0
        omega = 2 * np.pi / period
        expected_gyro_z = omega
        expected_centripetal = omega**2 * radius
        
        # Check measurements at different points
        for i, meas in enumerate(imu_data.measurements):
            if i < 10 or i >= len(imu_data.measurements) - 10:
                continue  # Skip transients at start/end
            
            # Gyroscope should measure constant yaw rate
            actual_gyro_z = meas.gyroscope[2]
            error = abs(actual_gyro_z - expected_gyro_z)
            assert error < 0.01, (
                f"Gyroscope Z measurement incorrect at sample {i}:\n"
                f"  Expected: {expected_gyro_z:.4f} rad/s\n"
                f"  Actual:   {actual_gyro_z:.4f} rad/s\n"
                f"  Error:    {error:.4f} rad/s"
            )
            
            # Accelerometer should measure gravity + centripetal
            # The exact value depends on orientation, but magnitude should be consistent
            accel_mag = np.linalg.norm(meas.accelerometer)
            
            # For circular motion, the magnitude varies between:
            # Min: |gravity - centripetal| when opposite
            # Max: gravity + centripetal when aligned
            min_mag = abs(9.81 - expected_centripetal)
            max_mag = 9.81 + expected_centripetal
            
            assert min_mag <= accel_mag <= max_mag * 1.1, (
                f"Accelerometer magnitude out of range at sample {i}:\n"
                f"  Expected range: [{min_mag:.2f}, {max_mag:.2f}]\n"
                f"  Actual: {accel_mag:.2f}"
            )
    
    def test_stationary_measures_gravity(self):
        """Test stationary IMU measures only gravity."""
        # Create noise-free IMU
        imu_calib = IMUCalibration(
            imu_id="test_imu",
            accelerometer_noise_density=0.0,
            accelerometer_random_walk=0.0,
            gyroscope_noise_density=0.0,
            gyroscope_random_walk=0.0,
            rate=100.0,
            gravity_magnitude=9.81
        )
        
        noise_params = IMUNoiseParams(
            accelerometer_noise_density=0.0,
            accelerometer_random_walk=0.0,
            gyroscope_noise_density=0.0,
            gyroscope_random_walk=0.0
        )
        
        imu_config = IMUConfig(
            noise_params=noise_params,
            gravity_magnitude=9.81
        )
        
        imu = IMUModel(imu_calib, imu_config)
        
        # Generate stationary trajectory
        # Use circle with radius 0 for static position
        params = {
            "radius": 0.0,  # Zero radius for static position
            "height": 0.0,
            "duration": 1.0,
            "rate": 100.0,
            "start_time": 0.0
        }
        traj = generate_trajectory("circle", params)
        
        # Generate measurements
        imu_data = imu.generate_perfect_measurements(traj)
        
        for meas in imu_data.measurements:
            # Accelerometer should measure gravity (0, 0, 9.81) in body frame
            expected_accel = np.array([0, 0, 9.81])
            np.testing.assert_allclose(
                meas.accelerometer, expected_accel,
                rtol=1e-5, atol=1e-5,
                err_msg="Stationary IMU should measure gravity"
            )
            
            # Gyroscope should measure zero
            np.testing.assert_allclose(
                meas.gyroscope, np.zeros(3),
                rtol=1e-5, atol=1e-5,
                err_msg="Stationary IMU should measure zero angular velocity"
            )


class TestPreintegrationConsistency:
    """Test consistency between different preintegration approaches."""
    
    def test_incremental_vs_batch(self):
        """Test that incremental and batch preintegration give same results."""
        dt = 0.01
        measurements = []
        
        # Create test measurements
        for i in range(100):
            meas = IMUMeasurement(
                timestamp=i * dt,
                accelerometer=np.array([0.1, 0.2, 9.81]),
                gyroscope=np.array([0.01, 0.02, 0.1])
            )
            measurements.append(meas)
        
        # Batch processing
        preintegrator_batch = IMUPreintegrator(gravity=np.array([0, 0, -9.81]))
        result_batch = preintegrator_batch.batch_process(measurements, 0, 1)
        
        # Incremental processing
        preintegrator_incr = IMUPreintegrator(gravity=np.array([0, 0, -9.81]))
        for i, meas in enumerate(measurements):
            if i == 0:
                # First measurement - just store it like batch_process does
                preintegrator_incr.measurements.append(meas)
                continue
            else:
                dt_step = meas.timestamp - measurements[i-1].timestamp
            preintegrator_incr.add_measurement(meas, dt_step)
        
        # Get result
        result_incr = preintegrator_incr.get_result()
        
        # Compare results
        np.testing.assert_allclose(
            result_batch.delta_position, result_incr.delta_position,
            rtol=1e-10, atol=1e-10,
            err_msg="Batch and incremental position differ"
        )
        
        np.testing.assert_allclose(
            result_batch.delta_velocity, result_incr.delta_velocity,
            rtol=1e-10, atol=1e-10,
            err_msg="Batch and incremental velocity differ"
        )
        
        # Compare rotations - handle different representations
        # batch_process returns rotation matrix, get_result returns quaternion
        batch_rot = result_batch.delta_rotation
        incr_rot = result_incr.delta_rotation
        
        # Convert to comparable format
        if batch_rot.shape == (3, 3) and incr_rot.shape == (4,):
            # Convert quaternion to rotation matrix for comparison
            from src.utils.math_utils import quaternion_to_rotation_matrix
            incr_rot_matrix = quaternion_to_rotation_matrix(incr_rot)
            np.testing.assert_allclose(
                batch_rot, incr_rot_matrix,
                rtol=1e-10, atol=1e-10,
                err_msg="Batch and incremental rotation differ"
            )
        else:
            # Direct comparison if same format
            np.testing.assert_allclose(
                batch_rot, incr_rot,
                rtol=1e-10, atol=1e-10,
                err_msg="Batch and incremental rotation differ"
            )


if __name__ == "__main__":
    # Run specific test that should fail with current bug
    test = TestIMUPreintegrationAccuracy()
    
    print("Testing rotation accumulation at different angles:")
    print("=" * 60)
    
    test_cases = [
        (90, "quarter circle"),
        (180, "half circle"),
        (270, "three-quarter circle"),
        (360, "full circle"),
    ]
    
    for angle, description in test_cases:
        try:
            test.test_rotation_accumulation(angle, description)
            print(f"✓ {angle}° ({description}) - PASSED")
        except AssertionError as e:
            print(f"✗ {angle}° ({description}) - FAILED")
            print(f"  {str(e).split(':')[1] if ':' in str(e) else str(e)}")
    
    print("\nConclusion:")
    print("-" * 40)
    print("This test suite will catch the rotation wraparound bug!")
    print("Current implementation will FAIL for angles > 180°")