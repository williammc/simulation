"""
Critical tests for IMU preintegration gravity handling.

These tests verify the gravity compensation bug identified in context.md.
The constant vertical velocity test is expected to FAIL with current implementation,
demonstrating the 10x error in vertical motion preintegration.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.simulation.imu_integration import IMUPreintegrator
from src.common.data_structures import IMUMeasurement


class TestPreintegrationGravity:
    """Test IMU preintegration with various gravity-related scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preintegrator = IMUPreintegrator(
            accel_noise_density=0.0,  # No noise for testing
            gyro_noise_density=0.0,
            gravity=np.array([0, 0, -9.81])
        )
        self.dt = 0.005  # 200 Hz IMU
    
    def test_stationary_upright(self):
        """Test preintegration for stationary IMU in upright position."""
        print("\n=== Test: Stationary Upright ===")
        
        # Reset with identity orientation (upright)
        self.preintegrator.reset(initial_orientation=np.eye(3))
        
        # Stationary IMU measures only gravity compensation
        # Specific force = [0, 0, 9.81] for zero acceleration
        specific_force = np.array([0, 0, 9.81])
        gyro = np.zeros(3)
        
        # Integrate for 0.5 seconds
        num_samples = int(0.5 / self.dt)
        for _ in range(num_samples):
            meas = IMUMeasurement(0, specific_force, gyro)
            self.preintegrator.add_measurement(meas, self.dt)
        
        # Expected: No motion
        print(f"Delta position: {self.preintegrator.delta_p}")
        print(f"Delta velocity: {self.preintegrator.delta_v}")
        print(f"Expected: [0, 0, 0] for both")
        
        # CURRENTLY FAILS: Gets ~1.2m in Z instead of 0
        assert np.allclose(self.preintegrator.delta_p, [0, 0, 0], atol=0.01), \
            f"Stationary should have zero position change, got {self.preintegrator.delta_p}"
        assert np.allclose(self.preintegrator.delta_v, [0, 0, 0], atol=0.01), \
            f"Stationary should have zero velocity change, got {self.preintegrator.delta_v}"
    
    def test_constant_velocity_vertical(self):
        """Test preintegration for constant vertical velocity (critical failing test)."""
        print("\n=== Test: Constant Vertical Velocity ===")
        
        # Reset with identity orientation
        self.preintegrator.reset(initial_orientation=np.eye(3))
        
        # Constant velocity upward at 0.25 m/s
        # Acceleration = 0, so specific force = -gravity = [0, 0, 9.81]
        specific_force = np.array([0, 0, 9.81])
        gyro = np.zeros(3)
        
        # Integrate for 0.5 seconds
        num_samples = int(0.5 / self.dt)
        for _ in range(num_samples):
            meas = IMUMeasurement(0, specific_force, gyro)
            self.preintegrator.add_measurement(meas, self.dt)
        
        # Expected motion (constant velocity, no acceleration)
        # With gravity-compensated preintegration, we expect zero deltas
        expected_delta_p = np.array([0, 0, 0])  # No acceleration = no position change
        expected_delta_v = np.array([0, 0, 0])  # No acceleration = no velocity change
        
        print(f"Delta position: {self.preintegrator.delta_p}")
        print(f"Expected: {expected_delta_p}")
        print(f"Error: {self.preintegrator.delta_p - expected_delta_p}")
        
        print(f"Delta velocity: {self.preintegrator.delta_v}")
        print(f"Expected: {expected_delta_v}")
        print(f"Error: {self.preintegrator.delta_v - expected_delta_v}")
        
        # This test now correctly expects zero delta_p and delta_v for constant velocity motion.
        # The actual displacement (0.125m) comes from the initial velocity term in SWBA.
        position_error = np.linalg.norm(self.preintegrator.delta_p - expected_delta_p)
        velocity_error = np.linalg.norm(self.preintegrator.delta_v - expected_delta_v)
        
        print(f"Position error magnitude: {position_error:.3f}m (should be < 0.01m)")
        print(f"Velocity error magnitude: {velocity_error:.3f}m/s (should be < 0.01m/s)")
        
        assert position_error < 0.01, \
            f"Constant velocity should give delta_p={expected_delta_p}, got {self.preintegrator.delta_p}"
        assert velocity_error < 0.01, \
            f"Constant velocity should give delta_v={expected_delta_v}, got {self.preintegrator.delta_v}"
    
    def test_constant_velocity_horizontal(self):
        """Test preintegration for constant horizontal velocity."""
        print("\n=== Test: Constant Horizontal Velocity ===")
        
        # Reset with identity orientation
        self.preintegrator.reset(initial_orientation=np.eye(3))
        
        # Moving at constant 1 m/s in X direction
        # No acceleration, so specific force = gravity compensation only
        specific_force = np.array([0, 0, 9.81])
        gyro = np.zeros(3)
        
        # Integrate for 0.5 seconds
        num_samples = int(0.5 / self.dt)
        for _ in range(num_samples):
            meas = IMUMeasurement(0, specific_force, gyro)
            self.preintegrator.add_measurement(meas, self.dt)
        
        # Expected: X motion from initial velocity, no acceleration
        # Note: Preintegration doesn't know about initial velocity,
        # so delta_p should be 0 (no acceleration)
        expected_delta_p = np.array([0, 0, 0])
        expected_delta_v = np.array([0, 0, 0])
        
        print(f"Delta position: {self.preintegrator.delta_p}")
        print(f"Delta velocity: {self.preintegrator.delta_v}")
        
        # This might work better than vertical due to no Z component
        assert np.allclose(self.preintegrator.delta_p, expected_delta_p, atol=0.1)
        assert np.allclose(self.preintegrator.delta_v, expected_delta_v, atol=0.1)
    
    def test_free_fall(self):
        """Test preintegration for free fall (gravity only)."""
        print("\n=== Test: Free Fall ===")
        
        # Reset with identity orientation
        self.preintegrator.reset(initial_orientation=np.eye(3))
        
        # Free fall: acceleration = -g, so specific force = 0
        specific_force = np.zeros(3)
        gyro = np.zeros(3)
        
        # Integrate for 0.5 seconds
        t = 0.5
        num_samples = int(t / self.dt)
        for _ in range(num_samples):
            meas = IMUMeasurement(0, specific_force, gyro)
            self.preintegrator.add_measurement(meas, self.dt)
        
        # Expected motion under gravity
        g = 9.81
        expected_delta_p = np.array([0, 0, -0.5 * g * t**2])  # -1.226m
        expected_delta_v = np.array([0, 0, -g * t])  # -4.905 m/s
        
        print(f"Delta position: {self.preintegrator.delta_p}")
        print(f"Expected: {expected_delta_p}")
        print(f"Delta velocity: {self.preintegrator.delta_v}")
        print(f"Expected: {expected_delta_v}")
        
        # This test might pass if preintegration handles zero specific force correctly
        assert np.allclose(self.preintegrator.delta_p[2], expected_delta_p[2], atol=0.1)
        assert np.allclose(self.preintegrator.delta_v[2], expected_delta_v[2], atol=0.1)
    
    def test_rotated_initial_orientation(self):
        """Test preintegration with non-identity initial orientation (like spiral)."""
        print("\n=== Test: Rotated Initial Orientation ===")
        
        # Use the same rotation as spiral trajectory
        R_init = np.array([
            [ 0.248, -0.967,  0.064],
            [ 0.936,  0.256,  0.240],
            [-0.248,  0.000,  0.969]
        ])
        
        self.preintegrator.reset(initial_orientation=R_init)
        
        # For constant velocity with this rotation, the specific force is complex
        # This demonstrates why spiral fails
        # Gravity in body frame: R^T @ [0, 0, -9.81]
        gravity_body = R_init.T @ np.array([0, 0, -9.81])
        print(f"Gravity in body frame: {gravity_body}")
        
        # For zero acceleration, specific force cancels gravity
        specific_force = -gravity_body  # This should give zero acceleration
        gyro = np.zeros(3)
        
        # Integrate for 0.5 seconds
        num_samples = int(0.5 / self.dt)
        for _ in range(num_samples):
            meas = IMUMeasurement(0, specific_force, gyro)
            self.preintegrator.add_measurement(meas, self.dt)
        
        print(f"Delta position: {self.preintegrator.delta_p}")
        print(f"Delta velocity: {self.preintegrator.delta_v}")
        
        # With proper gravity handling, these should be near zero
        # Currently FAILS badly
        assert np.linalg.norm(self.preintegrator.delta_p) < 0.1, \
            f"Zero acceleration should give small delta_p, got {self.preintegrator.delta_p}"
        assert np.linalg.norm(self.preintegrator.delta_v) < 0.1, \
            f"Zero acceleration should give small delta_v, got {self.preintegrator.delta_v}"
    
    def test_pure_rotation(self):
        """Test preintegration with pure rotation (no translation)."""
        print("\n=== Test: Pure Rotation ===")
        
        self.preintegrator.reset(initial_orientation=np.eye(3))
        
        # Pure rotation around Z axis at 0.1 rad/s
        specific_force = np.array([0, 0, 9.81])  # Gravity compensation
        gyro = np.array([0, 0, 0.1])
        
        # Integrate for 1 second
        num_samples = int(1.0 / self.dt)
        for _ in range(num_samples):
            meas = IMUMeasurement(0, specific_force, gyro)
            self.preintegrator.add_measurement(meas, self.dt)
        
        # Expected: Rotation but no translation (if gravity handled correctly)
        print(f"Delta position: {self.preintegrator.delta_p}")
        print(f"Delta velocity: {self.preintegrator.delta_v}")
        print(f"Delta rotation:\n{self.preintegrator.delta_R}")
        
        # Check rotation is approximately 0.1 radians
        expected_angle = 0.1  # rad
        c, s = np.cos(expected_angle), np.sin(expected_angle)
        expected_R = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        
        rotation_error = np.linalg.norm(self.preintegrator.delta_R - expected_R, 'fro')
        print(f"Rotation error: {rotation_error}")
        
        # Position and velocity should be small (just from rotation effects)
        assert np.linalg.norm(self.preintegrator.delta_p) < 0.5, \
            "Pure rotation should have minimal translation"
        assert np.linalg.norm(self.preintegrator.delta_v) < 0.5, \
            "Pure rotation should have minimal velocity change"


def run_all_tests():
    """Run all preintegration gravity tests and report results."""
    test = TestPreintegrationGravity()
    
    failures = []
    
    # Test 1: Stationary
    try:
        test.setup_method()
        test.test_stationary_upright()
        print("✓ Stationary test PASSED")
    except AssertionError as e:
        print(f"✗ Stationary test FAILED: {e}")
        failures.append("Stationary")
    
    # Test 2: Constant vertical velocity (EXPECTED TO FAIL)
    try:
        test.setup_method()
        test.test_constant_velocity_vertical()
        print("✓ Constant vertical velocity test PASSED")
    except AssertionError as e:
        print(f"✗ Constant vertical velocity test FAILED: {e}")
        failures.append("Constant vertical velocity")
    
    # Test 3: Constant horizontal velocity
    try:
        test.setup_method()
        test.test_constant_velocity_horizontal()
        print("✓ Constant horizontal velocity test PASSED")
    except AssertionError as e:
        print(f"✗ Constant horizontal velocity test FAILED: {e}")
        failures.append("Constant horizontal velocity")
    
    # Test 4: Free fall
    try:
        test.setup_method()
        test.test_free_fall()
        print("✓ Free fall test PASSED")
    except AssertionError as e:
        print(f"✗ Free fall test FAILED: {e}")
        failures.append("Free fall")
    
    # Test 5: Rotated initial orientation
    try:
        test.setup_method()
        test.test_rotated_initial_orientation()
        print("✓ Rotated orientation test PASSED")
    except AssertionError as e:
        print(f"✗ Rotated orientation test FAILED: {e}")
        failures.append("Rotated orientation")
    
    # Test 6: Pure rotation
    try:
        test.setup_method()
        test.test_pure_rotation()
        print("✓ Pure rotation test PASSED")
    except AssertionError as e:
        print(f"✗ Pure rotation test FAILED: {e}")
        failures.append("Pure rotation")
    
    # Summary
    print("\n" + "="*60)
    print("PREINTEGRATION GRAVITY TEST SUMMARY")
    print("="*60)
    if failures:
        print(f"FAILED TESTS ({len(failures)}):")
        for test_name in failures:
            print(f"  - {test_name}")
        print("\nThese failures demonstrate the gravity handling bug.")
        print("The preintegration integrates specific force directly,")
        print("causing ~10x error in vertical motion (1.2m instead of 0.125m).")
    else:
        print("All tests PASSED (gravity handling has been fixed!)")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()