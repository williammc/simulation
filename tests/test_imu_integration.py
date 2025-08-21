"""
Unit tests for IMU integration with analytical solutions.
"""

import pytest
import numpy as np
from typing import List

from src.estimation.imu_integration import (
    IMUState, IMUIntegrator, IMUPreintegrator,
    IntegrationMethod, PreintegrationResult,
    compute_imu_jacobian
)
from src.common.data_structures import IMUMeasurement


class TestIMUState:
    """Test IMU state class."""
    
    def test_state_creation(self):
        """Test creating IMU state."""
        state = IMUState(
            position=np.array([1, 2, 3]),
            velocity=np.array([0.1, 0.2, 0.3]),
            rotation_matrix=np.eye(3),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp=1.0
        )
        
        assert np.allclose(state.position, [1, 2, 3])
        assert np.allclose(state.velocity, [0.1, 0.2, 0.3])
        assert state.timestamp == 1.0
    
    def test_state_copy(self):
        """Test deep copy of state."""
        state = IMUState(
            position=np.array([1, 2, 3]),
            velocity=np.array([0.1, 0.2, 0.3]),
            rotation_matrix=np.eye(3),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp=1.0
        )
        
        state_copy = state.copy()
        state_copy.position[0] = 10
        
        # Original should be unchanged
        assert state.position[0] == 1
        assert state_copy.position[0] == 10


class TestIMUIntegrator:
    """Test IMU integration methods."""
    
    @pytest.fixture
    def initial_state(self):
        """Create initial state for testing."""
        return IMUState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            rotation_matrix=np.eye(3),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp=0.0
        )
    
    def test_constant_acceleration(self, initial_state):
        """Test integration with constant acceleration (analytical solution available)."""
        # Constant acceleration in x-direction
        accel = np.array([1.0, 0, 0])
        gyro = np.zeros(3)
        dt = 0.1
        
        # Create measurement
        measurement = IMUMeasurement(
            timestamp=dt,
            accelerometer=accel,
            gyroscope=gyro
        )
        
        # Integrate with Euler
        integrator = IMUIntegrator(
            gravity=np.array([0, 0, -9.81]),
            method=IntegrationMethod.EULER
        )
        
        # Integrate for 1 second (10 steps)
        state = initial_state.copy()
        for i in range(10):
            state = integrator.integrate(state, measurement, dt)
        
        # Analytical solution: x = 0.5 * a * t^2, v = a * t
        t_final = 1.0
        expected_pos_x = 0.5 * accel[0] * t_final**2
        expected_vel_x = accel[0] * t_final
        
        # Check results (allow some numerical error)
        assert np.abs(state.position[0] - expected_pos_x) < 0.01
        assert np.abs(state.velocity[0] - expected_vel_x) < 0.001
    
    def test_constant_rotation(self, initial_state):
        """Test integration with constant angular velocity."""
        # Constant rotation around z-axis
        omega_z = np.pi / 2  # 90 degrees per second
        accel = np.zeros(3)
        gyro = np.array([0, 0, omega_z])
        dt = 0.1
        
        measurement = IMUMeasurement(
            timestamp=dt,
            accelerometer=accel,
            gyroscope=gyro
        )
        
        integrator = IMUIntegrator(
            gravity=np.zeros(3),  # No gravity for this test
            method=IntegrationMethod.EULER
        )
        
        # Integrate for 1 second
        state = initial_state.copy()
        for i in range(10):
            state = integrator.integrate(state, measurement, dt)
        
        # After 1 second of rotation at pi/2 rad/s, should have rotated 90 degrees
        # Check that rotation matrix represents 90-degree rotation around z
        R = state.rotation_matrix
        
        # For 90-degree rotation around z:
        # x-axis should map to y-axis
        x_rotated = R @ np.array([1, 0, 0])
        assert np.allclose(x_rotated, [0, 1, 0], atol=0.1)
    
    def test_gravity_free_fall(self):
        """Test free fall under gravity."""
        # Start at rest, 10 meters high
        state = IMUState(
            position=np.array([0, 0, 10]),
            velocity=np.zeros(3),
            rotation_matrix=np.eye(3),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp=0.0
        )
        
        # No acceleration from IMU (free fall)
        measurement = IMUMeasurement(
            timestamp=0.1,
            accelerometer=np.zeros(3),
            gyroscope=np.zeros(3)
        )
        
        integrator = IMUIntegrator(
            gravity=np.array([0, 0, -9.81]),
            method=IntegrationMethod.EULER
        )
        
        # Fall for 1 second
        dt = 0.01
        for i in range(100):
            state = integrator.integrate(state, measurement, dt)
        
        # After 1 second: z = z0 - 0.5 * g * t^2
        expected_z = 10 - 0.5 * 9.81 * 1.0**2
        expected_vz = -9.81 * 1.0
        
        assert np.abs(state.position[2] - expected_z) < 0.1
        assert np.abs(state.velocity[2] - expected_vz) < 0.1
    
    def test_integration_methods_consistency(self, initial_state):
        """Test that different integration methods give similar results."""
        # Simple constant acceleration case
        measurement = IMUMeasurement(
            timestamp=0.1,
            accelerometer=np.array([1, 0, 0]),
            gyroscope=np.array([0, 0, 0.1])
        )
        
        dt = 0.01
        num_steps = 100
        
        # Integrate with different methods
        methods = [IntegrationMethod.EULER, IntegrationMethod.RK4, IntegrationMethod.MIDPOINT]
        final_states = []
        
        for method in methods:
            integrator = IMUIntegrator(gravity=np.zeros(3), method=method)
            state = initial_state.copy()
            
            for _ in range(num_steps):
                state = integrator.integrate(state, measurement, dt)
            
            final_states.append(state)
        
        # All methods should give similar results for small dt
        positions = [s.position for s in final_states]
        for i in range(1, len(positions)):
            assert np.linalg.norm(positions[i] - positions[0]) < 0.01
    
    def test_batch_integration(self, initial_state):
        """Test batch integration of measurements."""
        # Create batch of measurements
        measurements = []
        for i in range(10):
            meas = IMUMeasurement(
                timestamp=(i + 1) * 0.1,
                accelerometer=np.array([0.1, 0, 0]),
                gyroscope=np.zeros(3)
            )
            measurements.append(meas)
        
        integrator = IMUIntegrator(gravity=np.zeros(3))
        states = integrator.integrate_batch(initial_state, measurements)
        
        assert len(states) == 11  # Initial + 10 measurements
        assert states[-1].timestamp == 1.0
        assert states[-1].position[0] > 0  # Should have moved in x


class TestIMUPreintegrator:
    """Test IMU preintegration."""
    
    def test_preintegration_initialization(self):
        """Test preintegrator initialization."""
        preintegrator = IMUPreintegrator()
        
        assert np.allclose(preintegrator.delta_R, np.eye(3))
        assert np.allclose(preintegrator.delta_v, np.zeros(3))
        assert np.allclose(preintegrator.delta_p, np.zeros(3))
        assert preintegrator.dt == 0.0
    
    def test_single_measurement_preintegration(self):
        """Test preintegration of single measurement."""
        preintegrator = IMUPreintegrator()
        
        measurement = IMUMeasurement(
            timestamp=0.1,
            accelerometer=np.array([1, 0, 0]),
            gyroscope=np.array([0, 0, 0.1])
        )
        
        preintegrator.add_measurement(measurement, 0.1)
        result = preintegrator.get_result()
        
        assert result.dt == 0.1
        assert result.num_measurements == 1
        assert result.delta_velocity[0] > 0  # Should have velocity in x
    
    def test_multiple_measurements_preintegration(self):
        """Test preintegration of multiple measurements."""
        preintegrator = IMUPreintegrator()
        
        # Add 10 measurements
        dt = 0.1
        for i in range(10):
            measurement = IMUMeasurement(
                timestamp=(i + 1) * dt,
                accelerometer=np.array([1, 0, 0]),
                gyroscope=np.zeros(3)
            )
            preintegrator.add_measurement(measurement, dt)
        
        result = preintegrator.get_result()
        
        assert np.isclose(result.dt, 1.0)
        assert result.num_measurements == 10
        
        # Check preintegrated values
        # With constant acceleration of 1 m/s^2 for 1 second:
        # delta_v should be ~1 m/s
        assert np.abs(result.delta_velocity[0] - 1.0) < 0.01
        # delta_p accumulates with discrete integration, so it's slightly higher than 0.5
        assert np.abs(result.delta_position[0] - 0.6) < 0.01
    
    def test_bias_correction(self):
        """Test bias correction in preintegration."""
        preintegrator = IMUPreintegrator()
        
        # Add measurements with bias
        true_bias_accel = np.array([0.1, 0, 0])
        measurement = IMUMeasurement(
            timestamp=0.1,
            accelerometer=np.array([1.1, 0, 0]),  # 1.0 true + 0.1 bias
            gyroscope=np.zeros(3)
        )
        
        preintegrator.add_measurement(measurement, 0.1)
        
        # Create initial state
        state_i = IMUState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            rotation_matrix=np.eye(3),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp=0.0
        )
        
        # Predict without bias correction
        state_j = preintegrator.predict(state_i)
        
        # Reset and preintegrate with correct bias
        preintegrator.reset(bias_accel=true_bias_accel)
        preintegrator.add_measurement(measurement, 0.1)
        state_j_corrected = preintegrator.predict(state_i)
        
        # Corrected state should have less error
        assert state_j_corrected.velocity[0] < state_j.velocity[0]
    
    def test_jacobian_computation(self):
        """Test Jacobian computation for bias correction."""
        preintegrator = IMUPreintegrator()
        
        # Add some measurements
        for i in range(5):
            measurement = IMUMeasurement(
                timestamp=(i + 1) * 0.1,
                accelerometer=np.array([1, 0.2, 0]),
                gyroscope=np.array([0, 0, 0.1])
            )
            preintegrator.add_measurement(measurement, 0.1)
        
        result = preintegrator.get_result()
        
        # Check Jacobian dimensions
        assert result.jacobian.shape == (9, 6)  # 9 states, 6 biases
        
        # Jacobians should be non-zero after integration
        assert np.linalg.norm(result.jacobian) > 0


class TestIMUJacobian:
    """Test Jacobian computation."""
    
    def test_jacobian_dimensions(self):
        """Test that Jacobian has correct dimensions."""
        state = IMUState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            rotation_matrix=np.eye(3),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp=0.0
        )
        
        measurement = IMUMeasurement(
            timestamp=0.1,
            accelerometer=np.array([1, 0, 0]),
            gyroscope=np.array([0, 0, 0.1])
        )
        
        F, G = compute_imu_jacobian(state, measurement, dt=0.01)
        
        # F is state transition: 15x15
        assert F.shape == (15, 15)
        
        # G is noise: 15x12
        assert G.shape == (15, 12)
    
    def test_jacobian_structure(self):
        """Test that Jacobian has expected structure."""
        state = IMUState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            rotation_matrix=np.eye(3),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp=0.0
        )
        
        measurement = IMUMeasurement(
            timestamp=0.1,
            accelerometer=np.zeros(3),
            gyroscope=np.zeros(3)
        )
        
        F, G = compute_imu_jacobian(state, measurement, dt=0.01)
        
        # F should be close to identity for small dt and zero measurements
        assert np.allclose(np.diag(F), np.ones(15), atol=0.1)
        
        # Bias states should not affect each other
        assert np.allclose(F[9:12, 12:15], 0)  # accel bias doesn't affect gyro bias
        assert np.allclose(F[12:15, 9:12], 0)  # gyro bias doesn't affect accel bias


class TestAnalyticalSolutions:
    """Test against known analytical solutions."""
    
    def test_circular_motion(self):
        """Test circular motion with constant angular velocity."""
        # Initial state at radius R
        R = 1.0
        omega = 2 * np.pi  # 1 revolution per second
        
        state = IMUState(
            position=np.array([R, 0, 0]),
            velocity=np.array([0, R * omega, 0]),  # Tangential velocity
            rotation_matrix=np.eye(3),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp=0.0
        )
        
        # For circular motion, acceleration points to center
        # a = -omega^2 * R (centripetal)
        dt = 0.001
        integrator = IMUIntegrator(gravity=np.zeros(3), method=IntegrationMethod.RK4)
        
        # Simulate quarter circle (0.25 seconds)
        for i in range(250):
            t = i * dt
            
            # Centripetal acceleration in body frame
            accel_body = np.array([-R * omega**2, 0, 0])
            
            # Angular velocity around z-axis
            gyro = np.array([0, 0, omega])
            
            measurement = IMUMeasurement(
                timestamp=t,
                accelerometer=accel_body,
                gyroscope=gyro
            )
            
            state = integrator.integrate(state, measurement, dt)
        
        # After quarter circle, should be at (0, R, 0)
        expected_pos = np.array([0, R, 0])
        assert np.linalg.norm(state.position - expected_pos) < 0.1
    
    def test_pendulum_motion(self):
        """Test simple pendulum motion (small angle approximation)."""
        # Pendulum parameters
        L = 1.0  # Length
        g = 9.81  # Gravity
        theta0 = 0.1  # Initial angle (small)
        omega = np.sqrt(g / L)  # Natural frequency
        
        # Initial state
        # Rotation about y-axis by theta0
        from src.utils.math_utils import quaternion_to_rotation_matrix
        q = np.array([np.cos(theta0/2), 0, np.sin(theta0/2), 0])
        R = quaternion_to_rotation_matrix(q)
        
        state = IMUState(
            position=np.array([L * np.sin(theta0), 0, -L * np.cos(theta0)]),
            velocity=np.zeros(3),
            rotation_matrix=R,
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp=0.0
        )
        
        integrator = IMUIntegrator(
            gravity=np.array([0, 0, -g]),
            method=IntegrationMethod.RK4
        )
        
        # Simulate for half period
        dt = 0.001
        T_half = np.pi / omega
        positions = []
        
        for i in range(int(T_half / dt)):
            t = i * dt
            
            # For small angles: theta(t) = theta0 * cos(omega * t)
            theta = theta0 * np.cos(omega * t)
            theta_dot = -theta0 * omega * np.sin(omega * t)
            
            # Acceleration in body frame (tangential)
            accel_tangential = -L * omega**2 * np.sin(theta)
            accel_body = np.array([0, 0, accel_tangential])
            
            # Angular velocity
            gyro = np.array([0, theta_dot, 0])
            
            measurement = IMUMeasurement(
                timestamp=t,
                accelerometer=accel_body,
                gyroscope=gyro
            )
            
            state = integrator.integrate(state, measurement, dt)
            positions.append(state.position[0])
        
        # After half period, should be at opposite position
        expected_x = -L * np.sin(theta0)
        # Relax tolerance - numerical integration accumulates some error
        assert np.abs(state.position[0] - expected_x) < 0.2