#!/usr/bin/env python3
"""
Test our IMU preintegration against GTSAM's implementation.
This will help us identify any discrepancies in the preintegration logic.
"""

import numpy as np
import gtsam
from src.estimation.imu_integration import IMUPreintegrator
from src.common.data_structures import IMUMeasurement


def test_preintegration_against_gtsam():
    """Compare our preintegration with GTSAM's PreintegratedImuMeasurements."""
    
    # Create GTSAM preintegration params
    # Note: GTSAM uses different convention for gravity (positive up)
    gravity = np.array([0, 0, -9.81])
    
    # Create noise parameters
    accel_noise_density = 0.01
    gyro_noise_density = 0.001
    accel_random_walk = 0.001
    gyro_random_walk = 0.0001
    integration_error_cov = 1e-8
    
    # GTSAM preintegration parameters
    params = gtsam.PreintegrationParams.MakeSharedU(gravity[2])  # gravity magnitude
    params.setAccelerometerCovariance(accel_noise_density**2 * np.eye(3))
    params.setGyroscopeCovariance(gyro_noise_density**2 * np.eye(3))
    params.setIntegrationCovariance(integration_error_cov * np.eye(3))
    
    # Zero biases for this test
    bias = gtsam.imuBias.ConstantBias()
    
    # Create GTSAM preintegrated measurements
    gtsam_pim = gtsam.PreintegratedImuMeasurements(params, bias)
    
    # Create our preintegrator
    our_preintegrator = IMUPreintegrator(
        accel_noise_density=accel_noise_density,
        gyro_noise_density=gyro_noise_density,
        accel_random_walk=accel_random_walk,
        gyro_random_walk=gyro_random_walk,
        gravity=gravity
    )
    
    # Reset with identity orientation
    our_preintegrator.reset(initial_orientation=np.eye(3))
    
    # Test Case 1: Simple constant measurements for circular motion
    print("Test Case 1: Circular Motion at Constant Height")
    print("=" * 60)
    
    # For circular motion at constant height:
    # - Centripetal acceleration in body Y
    # - No vertical acceleration (constant height)
    # - Constant angular velocity around Z
    
    dt = 0.01  # 100 Hz
    num_measurements = 100  # 1 second of data
    
    # Specific force for circular motion (gravity already removed)
    # This is what the IMU actually measures
    accel = np.array([0.0, 3.125, 0.0])  # Centripetal acceleration
    gyro = np.array([0.0, 0.0, 1.256])   # Rotation around Z
    
    # Add measurements to both systems
    for i in range(num_measurements):
        # GTSAM expects measurements in specific force (gravity removed)
        gtsam_pim.integrateMeasurement(accel, gyro, dt)
        
        # Our system also expects specific force
        meas = IMUMeasurement(
            timestamp=i * dt,
            accelerometer=accel,
            gyroscope=gyro
        )
        our_preintegrator.add_measurement(meas, dt)
    
    # Get results from GTSAM
    gtsam_delta_p = gtsam_pim.deltaPij()
    gtsam_delta_v = gtsam_pim.deltaVij()
    gtsam_delta_R = gtsam_pim.deltaRij().matrix()
    
    # Get results from our implementation
    our_result = our_preintegrator.get_result()
    our_delta_p = our_result.delta_position
    our_delta_v = our_result.delta_velocity
    from src.utils.math_utils import quaternion_to_rotation_matrix
    our_delta_R = quaternion_to_rotation_matrix(our_result.delta_rotation)
    
    # Compare results
    print(f"Time integrated: {num_measurements * dt:.2f}s")
    print("\nDelta Position:")
    print(f"  GTSAM:     {gtsam_delta_p}")
    print(f"  Ours:      {our_delta_p}")
    print(f"  Difference: {np.linalg.norm(gtsam_delta_p - our_delta_p):.6f}m")
    
    print("\nDelta Velocity:")
    print(f"  GTSAM:     {gtsam_delta_v}")
    print(f"  Ours:      {our_delta_v}")
    print(f"  Difference: {np.linalg.norm(gtsam_delta_v - our_delta_v):.6f}m/s")
    
    print("\nDelta Rotation:")
    rotation_error = np.linalg.norm(gtsam_delta_R - our_delta_R, 'fro')
    print(f"  Frobenius norm difference: {rotation_error:.6f}")
    
    # Test Case 2: Zero measurements (should give zero deltas)
    print("\n\nTest Case 2: Zero Measurements (Static)")
    print("=" * 60)
    
    # Reset both systems
    gtsam_pim.resetIntegration()
    our_preintegrator.reset(initial_orientation=np.eye(3))
    
    # Add zero measurements
    for i in range(50):
        gtsam_pim.integrateMeasurement(np.zeros(3), np.zeros(3), dt)
        
        meas = IMUMeasurement(
            timestamp=i * dt,
            accelerometer=np.zeros(3),
            gyroscope=np.zeros(3)
        )
        our_preintegrator.add_measurement(meas, dt)
    
    print(f"Time integrated: {50 * dt:.2f}s")
    print("\nDelta Position:")
    print(f"  GTSAM:     {gtsam_pim.deltaPij()}")
    print(f"  Ours:      {our_preintegrator.get_result().delta_position}")
    
    print("\nDelta Velocity:")
    print(f"  GTSAM:     {gtsam_pim.deltaVij()}")
    print(f"  Ours:      {our_preintegrator.get_result().delta_velocity}")
    
    # Test Case 3: Pure rotation
    print("\n\nTest Case 3: Pure Rotation (No Translation)")
    print("=" * 60)
    
    # Reset both systems
    gtsam_pim.resetIntegration()
    our_preintegrator.reset(initial_orientation=np.eye(3))
    
    # Pure rotation around Z axis
    for i in range(100):
        gtsam_pim.integrateMeasurement(np.zeros(3), np.array([0, 0, np.pi/2]), dt)
        
        meas = IMUMeasurement(
            timestamp=i * dt,
            accelerometer=np.zeros(3),
            gyroscope=np.array([0, 0, np.pi/2])
        )
        our_preintegrator.add_measurement(meas, dt)
    
    print(f"Expected rotation: {100 * dt * np.pi/2:.2f} rad = {np.degrees(100 * dt * np.pi/2):.1f}°")
    
    gtsam_angle = np.arccos((np.trace(gtsam_pim.deltaRij().matrix()) - 1) / 2)
    our_result = our_preintegrator.get_result()
    our_R = quaternion_to_rotation_matrix(our_result.delta_rotation)
    our_angle = np.arccos((np.trace(our_R) - 1) / 2)
    
    print(f"\nActual rotation:")
    print(f"  GTSAM: {np.degrees(gtsam_angle):.1f}°")
    print(f"  Ours:  {np.degrees(our_angle):.1f}°")
    
    print(f"\nDelta Position (should be ~0):")
    print(f"  GTSAM: |p| = {np.linalg.norm(gtsam_pim.deltaPij()):.6f}m")
    print(f"  Ours:  |p| = {np.linalg.norm(our_result.delta_position):.6f}m")


if __name__ == "__main__":
    test_preintegration_against_gtsam()