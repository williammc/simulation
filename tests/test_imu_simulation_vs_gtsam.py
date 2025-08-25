#!/usr/bin/env python3
"""
Test to compare our IMU simulation model with GTSAM's expectations.
This helps identify if we're generating IMU measurements correctly.
"""

import numpy as np
import gtsam
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from src.simulation.imu_model import IMUModel, IMUNoiseConfig
from src.simulation.trajectory_generator import generate_trajectory
from src.common.data_structures import IMUCalibration


def test_imu_model_comparison():
    """Compare our IMU model output with GTSAM's expected measurements."""
    
    print("=" * 80)
    print("IMU MODEL COMPARISON: Our Simulation vs GTSAM Expectations")
    print("=" * 80)
    
    # Test 1: Static case (hovering at constant height)
    print("\nTest 1: Static Hovering")
    print("-" * 40)
    
    # What should an IMU measure when hovering?
    # - Accelerometer: Upward force to counteract gravity = [0, 0, g]
    # - Gyroscope: Zero rotation = [0, 0, 0]
    
    print("Expected IMU measurements for hovering:")
    print("  Accelerometer: [0, 0, 9.81] m/s² (upward specific force)")
    print("  Gyroscope: [0, 0, 0] rad/s")
    
    # Generate a static trajectory
    params = {
        "start_position": [0, 0, 1],
        "end_position": [0, 0, 1],  # Same position - static
        "duration": 1.0,
        "rate": 100.0,
        "start_time": 0.0
    }
    static_traj = generate_trajectory("line", params)
    
    # Create IMU model
    imu_calib = IMUCalibration(
        imu_id="test_imu",
        accelerometer_noise_density=0.0,  # No noise for testing
        accelerometer_random_walk=0.0,
        gyroscope_noise_density=0.0,
        gyroscope_random_walk=0.0,
        rate=200.0
    )
    
    noise_config = IMUNoiseConfig(
        accel_noise_density=0.0,
        gyro_noise_density=0.0,
        gravity_magnitude=9.81,
        seed=42
    )
    
    imu_model = IMUModel(calibration=imu_calib, noise_config=noise_config)
    
    # Generate IMU measurements
    imu_data = imu_model.generate_perfect_measurements(static_traj)
    
    # Check first few measurements
    print("\nOur IMU model output for static hovering:")
    for i in range(min(3, len(imu_data.measurements))):
        m = imu_data.measurements[i]
        print(f"  t={m.timestamp:.3f}: accel={m.accelerometer}, gyro={m.gyroscope}")
    
    # Average measurements
    accels = np.array([m.accelerometer for m in imu_data.measurements])
    gyros = np.array([m.gyroscope for m in imu_data.measurements])
    
    mean_accel = np.mean(accels, axis=0)
    mean_gyro = np.mean(gyros, axis=0)
    
    print(f"\nAverage measurements:")
    print(f"  Accelerometer: {mean_accel}")
    print(f"  Gyroscope: {mean_gyro}")
    
    # Test what GTSAM does with these measurements
    print("\nGTSAM preintegration with our measurements:")
    params_gtsam = gtsam.PreintegrationParams.MakeSharedU(9.81)
    bias = gtsam.imuBias.ConstantBias()
    pim = gtsam.PreintegratedImuMeasurements(params_gtsam, bias)
    
    dt = 0.005  # 200 Hz
    for i in range(100):  # 0.5 seconds
        if i < len(imu_data.measurements):
            m = imu_data.measurements[i]
            pim.integrateMeasurement(m.accelerometer, m.gyroscope, dt)
    
    print(f"  Delta position after 0.5s: {pim.deltaPij()}")
    print(f"  Delta velocity after 0.5s: {pim.deltaVij()}")
    print(f"  Expected for hovering: both should be ~[0, 0, 0]")
    
    # Test 2: Circular motion at constant height
    print("\n\nTest 2: Circular Motion at Constant Height")
    print("-" * 40)
    
    # Generate circular trajectory
    circle_params = {
        "radius": 2.0,
        "height": 1.5,
        "duration": 5.0,  # 5 second period
        "rate": 200.0,
        "start_time": 0.0
    }
    circle_traj = generate_trajectory("circle", circle_params)
    
    # Expected for circular motion
    radius = 2.0
    period = 5.0
    omega = 2 * np.pi / period
    v_tangential = omega * radius
    a_centripetal = omega**2 * radius
    
    print(f"Circle parameters:")
    print(f"  Radius: {radius} m")
    print(f"  Period: {period} s")
    print(f"  Angular velocity: {omega:.3f} rad/s")
    print(f"  Tangential velocity: {v_tangential:.3f} m/s")
    print(f"  Centripetal acceleration: {a_centripetal:.3f} m/s²")
    
    # Generate IMU measurements
    circle_imu = imu_model.generate_perfect_measurements(circle_traj)
    
    print(f"\nGenerated {len(circle_imu.measurements)} IMU measurements")
    
    # Sample measurements at different points in the circle
    sample_times = [0.0, period/4, period/2, 3*period/4]  # 0°, 90°, 180°, 270°
    
    print("\nIMU measurements at different points in the circle:")
    for target_t in sample_times:
        # Find closest measurement
        closest_idx = np.argmin([abs(m.timestamp - target_t) for m in circle_imu.measurements])
        m = circle_imu.measurements[closest_idx]
        angle = (m.timestamp / period) * 360
        
        print(f"\n  At t={m.timestamp:.3f}s (≈{angle:.0f}°):")
        print(f"    Accel: [{m.accelerometer[0]:.3f}, {m.accelerometer[1]:.3f}, {m.accelerometer[2]:.3f}]")
        print(f"    Gyro:  [{m.gyroscope[0]:.3f}, {m.gyroscope[1]:.3f}, {m.gyroscope[2]:.3f}]")
        
        # Expected accelerometer in body frame
        # X: tangent (forward), Y: pointing to center, Z: up
        # Centripetal acceleration is in Y (toward center)
        # Z should be ~9.81 to counteract gravity
        print(f"    Expected accel: [~0, ~{a_centripetal:.3f}, ~9.81]")
        print(f"    Expected gyro:  [0, 0, ~{omega:.3f}]")
    
    # Test GTSAM preintegration with circular motion
    print("\n\nGTSAM preintegration with circular motion:")
    pim_circle = gtsam.PreintegratedImuMeasurements(params_gtsam, bias)
    
    # Integrate for one full circle
    num_meas = int(period / dt)
    for i in range(min(num_meas, len(circle_imu.measurements))):
        m = circle_imu.measurements[i]
        pim_circle.integrateMeasurement(m.accelerometer, m.gyroscope, dt)
    
    dp = pim_circle.deltaPij()
    dv = pim_circle.deltaVij()
    dr = pim_circle.deltaRij()
    
    print(f"  After one full rotation ({period}s):")
    print(f"    Delta position: {dp}")
    print(f"    Delta velocity: {dv}")
    print(f"    |Delta position|: {np.linalg.norm(dp):.3f} m")
    print(f"    Expected: Should return close to origin (small error)")
    
    # Extract rotation angle
    angle_rad = np.arccos((np.trace(dr.matrix()) - 1) / 2)
    angle_deg = np.degrees(angle_rad)
    print(f"    Total rotation: {angle_deg:.1f}°")
    print(f"    Expected: ~360° for one full circle")
    
    # Test 3: Compare specific force understanding
    print("\n\nTest 3: Specific Force Model Comparison")
    print("-" * 40)
    
    print("\nUnderstanding of specific force (what IMU measures):")
    print("  IMU accelerometer measures: f = a - g (in body frame)")
    print("  Where a is actual acceleration and g is gravity")
    
    # Create a simple upward acceleration case
    print("\n3a. Object accelerating upward at 2 m/s²:")
    print("  Actual acceleration: [0, 0, 2] m/s² (upward)")
    print("  Gravity: [0, 0, -9.81] m/s² (downward)")
    print("  Specific force: [0, 0, 2 - (-9.81)] = [0, 0, 11.81] m/s²")
    
    # Create trajectory with upward acceleration
    # This is tricky - we need a trajectory with constant upward acceleration
    # For simplicity, we'll manually check the model's output
    
    print("\n3b. Object in free fall:")
    print("  Actual acceleration: [0, 0, -9.81] m/s² (gravity)")
    print("  Gravity: [0, 0, -9.81] m/s²")
    print("  Specific force: [0, 0, -9.81 - (-9.81)] = [0, 0, 0] m/s²")
    print("  (IMU reads zero in free fall!)")
    
    # Test 4: Check if our model matches GTSAM's conventions
    print("\n\nTest 4: Convention Check")
    print("-" * 40)
    
    print("Checking if our IMU model uses same conventions as GTSAM:")
    print("1. Gravity direction: GTSAM uses negative Z for gravity")
    print(f"   Our model gravity: {imu_model.gravity_world}")
    
    print("\n2. Specific force formula: f = a - g")
    print("   Our model should output specific force, not acceleration")
    
    print("\n3. Body frame convention:")
    print("   Both should use same body frame orientation")
    
    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Check hovering case
    hovering_correct = abs(mean_accel[2] - 9.81) < 0.1
    print(f"\n1. Hovering test: {'✓ PASS' if hovering_correct else '✗ FAIL'}")
    print(f"   Expected Z accel: 9.81, Got: {mean_accel[2]:.3f}")
    
    # Check circular motion
    rotation_correct = abs(angle_deg - 360) < 10
    print(f"\n2. Circular motion: {'✓ PASS' if rotation_correct else '✗ FAIL'}")
    print(f"   Expected rotation: 360°, Got: {angle_deg:.1f}°")
    
    # Check Z drift
    z_drift = dp[2]
    expected_z_drift = 9.81 * period**2 / 2  # From integrating constant upward force
    z_drift_ratio = z_drift / expected_z_drift if expected_z_drift > 0 else 0
    
    print(f"\n3. Z drift analysis:")
    print(f"   Measured Z drift: {z_drift:.3f} m")
    print(f"   Expected from constant 9.81 m/s² specific force: {expected_z_drift:.3f} m")
    print(f"   Ratio: {z_drift_ratio:.3f}")
    
    if abs(z_drift_ratio - 1.0) < 0.1:
        print("   ✓ Z drift matches expected value - IMU model is correct")
        print("   The drift is from integrating specific force, not an error!")
    else:
        print("   ✗ Z drift doesn't match - possible IMU model issue")
    
    return hovering_correct and rotation_correct


def plot_imu_measurements(imu_data, title="IMU Measurements"):
    """Plot IMU measurements over time."""
    
    times = [m.timestamp for m in imu_data.measurements]
    accels = np.array([m.accelerometer for m in imu_data.measurements])
    gyros = np.array([m.gyroscope for m in imu_data.measurements])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Accelerometer
    ax1.plot(times, accels[:, 0], 'r-', label='X', alpha=0.7)
    ax1.plot(times, accels[:, 1], 'g-', label='Y', alpha=0.7)
    ax1.plot(times, accels[:, 2], 'b-', label='Z', alpha=0.7)
    ax1.axhline(y=9.81, color='b', linestyle='--', alpha=0.3, label='g=9.81')
    ax1.set_ylabel('Specific Force (m/s²)')
    ax1.set_title(f'{title} - Accelerometer')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gyroscope
    ax2.plot(times, gyros[:, 0], 'r-', label='X', alpha=0.7)
    ax2.plot(times, gyros[:, 1], 'g-', label='Y', alpha=0.7)
    ax2.plot(times, gyros[:, 2], 'b-', label='Z', alpha=0.7)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.set_title(f'{title} - Gyroscope')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    test_imu_model_comparison()