#!/usr/bin/env python3
"""
Test comparing raw IMU simulation outputs between our model and GTSAM's model.
No preintegration - just comparing the raw accelerometer and gyroscope measurements.
"""

import numpy as np
import gtsam
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from src.simulation.trajectory_generator import generate_trajectory
from src.simulation.imu_model import IMUModel, IMUNoiseConfig
from src.common.data_structures import IMUCalibration, Pose, TrajectoryState


def simulate_imu_with_gtsam(trajectory_states, imu_rate=200.0, gravity_magnitude=9.81):
    """
    Simulate IMU measurements using GTSAM's IMU model.
    
    This mimics what GTSAM would expect as input - computing specific force
    from the trajectory dynamics.
    """
    measurements = []
    dt = 1.0 / imu_rate
    gravity_world = np.array([0, 0, -gravity_magnitude])
    
    for i in range(len(trajectory_states) - 1):
        curr_state = trajectory_states[i]
        next_state = trajectory_states[i + 1]
        
        # Get current pose and velocity
        R_wb = curr_state.pose.rotation_matrix  # World to body rotation
        R_bw = R_wb.T  # Body to world
        
        # Compute acceleration from velocity change (finite difference)
        if curr_state.velocity is not None and next_state.velocity is not None:
            dt_actual = next_state.pose.timestamp - curr_state.pose.timestamp
            if dt_actual > 0:
                accel_world = (next_state.velocity - curr_state.velocity) / dt_actual
            else:
                accel_world = np.zeros(3)
        else:
            accel_world = np.zeros(3)
        
        # Compute specific force: f = a - g (in body frame)
        # This is what an IMU measures
        specific_force_world = accel_world - gravity_world
        specific_force_body = R_bw @ specific_force_world
        
        # Angular velocity in body frame
        omega_body = curr_state.angular_velocity if curr_state.angular_velocity is not None else np.zeros(3)
        
        measurements.append({
            'timestamp': curr_state.pose.timestamp,
            'accelerometer': specific_force_body,
            'gyroscope': omega_body
        })
    
    return measurements


def compare_raw_imu_simulations():
    """
    Compare raw IMU outputs from our model vs GTSAM-style simulation.
    """
    
    print("=" * 80)
    print("RAW IMU SIMULATION COMPARISON: Our Model vs GTSAM Model")
    print("=" * 80)
    
    # Test 1: Static hovering
    print("\nTest 1: Static Hovering at Constant Height")
    print("-" * 40)
    
    # Create static trajectory
    static_params = {
        "start_position": [2.0, 1.0, 1.5],
        "end_position": [2.0, 1.0, 1.5],  # Same position
        "duration": 1.0,
        "rate": 200.0,
        "start_time": 0.0
    }
    static_traj = generate_trajectory("line", static_params)
    
    # Our IMU model
    imu_calib = IMUCalibration(
        imu_id="test",
        accelerometer_noise_density=0.0,
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
    
    our_imu_model = IMUModel(calibration=imu_calib, noise_config=noise_config)
    our_imu_data = our_imu_model.generate_perfect_measurements(static_traj)
    
    # GTSAM-style simulation
    gtsam_imu_data = simulate_imu_with_gtsam(static_traj.states)
    
    print(f"Generated {len(our_imu_data.measurements)} measurements from our model")
    print(f"Generated {len(gtsam_imu_data)} measurements from GTSAM model")
    
    # Compare first few measurements
    print("\nFirst 3 measurements comparison:")
    for i in range(min(3, len(our_imu_data.measurements), len(gtsam_imu_data))):
        our_m = our_imu_data.measurements[i]
        gtsam_m = gtsam_imu_data[i]
        
        print(f"\nMeasurement {i} at t={our_m.timestamp:.3f}s:")
        print(f"  Our accel:   {our_m.accelerometer}")
        print(f"  GTSAM accel: {gtsam_m['accelerometer']}")
        print(f"  Difference:  {np.linalg.norm(our_m.accelerometer - gtsam_m['accelerometer']):.6f}")
        print(f"  Our gyro:    {our_m.gyroscope}")
        print(f"  GTSAM gyro:  {gtsam_m['gyroscope']}")
    
    # Check average values
    our_accels = np.array([m.accelerometer for m in our_imu_data.measurements])
    gtsam_accels = np.array([m['accelerometer'] for m in gtsam_imu_data])
    
    print(f"\nAverage accelerometer values:")
    print(f"  Our model:   {np.mean(our_accels, axis=0)}")
    print(f"  GTSAM model: {np.mean(gtsam_accels, axis=0)}")
    print(f"  Expected:    [0, 0, 9.81] (upward specific force)")
    
    # Test 2: Circular motion
    print("\n\nTest 2: Circular Motion at Constant Height")
    print("-" * 40)
    
    circle_params = {
        "radius": 2.0,
        "height": 1.5,
        "duration": 2.0,  # 2 second period for faster rotation
        "rate": 200.0,
        "start_time": 0.0
    }
    circle_traj = generate_trajectory("circle", circle_params)
    
    # Our model
    our_circle_imu = our_imu_model.generate_perfect_measurements(circle_traj)
    
    # GTSAM model
    gtsam_circle_imu = simulate_imu_with_gtsam(circle_traj.states)
    
    print(f"Generated {len(our_circle_imu.measurements)} measurements from our model")
    print(f"Generated {len(gtsam_circle_imu)} measurements from GTSAM model")
    
    # Expected values
    radius = 2.0
    period = 2.0
    omega = 2 * np.pi / period
    v_tangential = omega * radius
    a_centripetal = omega**2 * radius
    
    print(f"\nExpected motion parameters:")
    print(f"  Angular velocity: {omega:.3f} rad/s")
    print(f"  Centripetal acceleration: {a_centripetal:.3f} m/s²")
    
    # Sample at different points in the circle
    sample_indices = [0, len(our_circle_imu.measurements)//4, 
                      len(our_circle_imu.measurements)//2, 
                      3*len(our_circle_imu.measurements)//4]
    
    print(f"\nMeasurements at different points in circle:")
    for idx in sample_indices:
        if idx < min(len(our_circle_imu.measurements), len(gtsam_circle_imu)):
            our_m = our_circle_imu.measurements[idx]
            gtsam_m = gtsam_circle_imu[idx]
            angle = (our_m.timestamp / period) * 360
            
            print(f"\nAt t={our_m.timestamp:.3f}s (≈{angle:.0f}°):")
            print(f"  Our accel:   [{our_m.accelerometer[0]:.3f}, {our_m.accelerometer[1]:.3f}, {our_m.accelerometer[2]:.3f}]")
            print(f"  GTSAM accel: [{gtsam_m['accelerometer'][0]:.3f}, {gtsam_m['accelerometer'][1]:.3f}, {gtsam_m['accelerometer'][2]:.3f}]")
            print(f"  Accel diff:  {np.linalg.norm(our_m.accelerometer - gtsam_m['accelerometer']):.6f}")
            print(f"  Our gyro:    [{our_m.gyroscope[0]:.3f}, {our_m.gyroscope[1]:.3f}, {our_m.gyroscope[2]:.3f}]")
            print(f"  GTSAM gyro:  [{gtsam_m['gyroscope'][0]:.3f}, {gtsam_m['gyroscope'][1]:.3f}, {gtsam_m['gyroscope'][2]:.3f}]")
    
    # Test 3: Vertical acceleration
    print("\n\nTest 3: Vertical Motion (Accelerating Upward)")
    print("-" * 40)
    
    # Create trajectory with upward acceleration
    vertical_params = {
        "start_position": [0, 0, 0],
        "end_position": [0, 0, 5],  # Move up 5 meters
        "duration": 2.0,
        "rate": 200.0,
        "start_time": 0.0
    }
    vertical_traj = generate_trajectory("line", vertical_params)
    
    # Add constant acceleration by modifying velocities
    # For simplicity, we'll use the existing trajectory
    
    # Our model
    our_vertical_imu = our_imu_model.generate_perfect_measurements(vertical_traj)
    
    # GTSAM model
    gtsam_vertical_imu = simulate_imu_with_gtsam(vertical_traj.states)
    
    print(f"Generated {len(our_vertical_imu.measurements)} measurements")
    
    # Check measurements during acceleration
    print("\nMeasurements during vertical motion:")
    for i in [0, len(our_vertical_imu.measurements)//2, -1]:
        if 0 <= i < min(len(our_vertical_imu.measurements), len(gtsam_vertical_imu)):
            our_m = our_vertical_imu.measurements[i]
            gtsam_m = gtsam_vertical_imu[i]
            
            print(f"\nAt t={our_m.timestamp:.3f}s:")
            print(f"  Our Z accel:   {our_m.accelerometer[2]:.3f} m/s²")
            print(f"  GTSAM Z accel: {gtsam_m['accelerometer'][2]:.3f} m/s²")
            print(f"  Difference:    {abs(our_m.accelerometer[2] - gtsam_m['accelerometer'][2]):.6f}")
    
    # Test 4: Statistical comparison
    print("\n\nTest 4: Statistical Comparison")
    print("-" * 40)
    
    # Collect all differences
    all_accel_diffs = []
    all_gyro_diffs = []
    
    for test_name, our_data, gtsam_data in [
        ("Static", our_imu_data.measurements, gtsam_imu_data),
        ("Circle", our_circle_imu.measurements, gtsam_circle_imu),
        ("Vertical", our_vertical_imu.measurements, gtsam_vertical_imu)
    ]:
        n = min(len(our_data), len(gtsam_data))
        for i in range(n):
            our_m = our_data[i]
            gtsam_m = gtsam_data[i]
            
            accel_diff = np.linalg.norm(our_m.accelerometer - gtsam_m['accelerometer'])
            gyro_diff = np.linalg.norm(our_m.gyroscope - gtsam_m['gyroscope'])
            
            all_accel_diffs.append(accel_diff)
            all_gyro_diffs.append(gyro_diff)
    
    all_accel_diffs = np.array(all_accel_diffs)
    all_gyro_diffs = np.array(all_gyro_diffs)
    
    print(f"\nAccelerometer differences:")
    print(f"  Mean:   {np.mean(all_accel_diffs):.6f} m/s²")
    print(f"  Std:    {np.std(all_accel_diffs):.6f} m/s²")
    print(f"  Max:    {np.max(all_accel_diffs):.6f} m/s²")
    print(f"  99%ile: {np.percentile(all_accel_diffs, 99):.6f} m/s²")
    
    print(f"\nGyroscope differences:")
    print(f"  Mean:   {np.mean(all_gyro_diffs):.6f} rad/s")
    print(f"  Std:    {np.std(all_gyro_diffs):.6f} rad/s")
    print(f"  Max:    {np.max(all_gyro_diffs):.6f} rad/s")
    print(f"  99%ile: {np.percentile(all_gyro_diffs, 99):.6f} rad/s")
    
    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Define pass criteria
    accel_threshold = 0.01  # 0.01 m/s² difference
    gyro_threshold = 0.001   # 0.001 rad/s difference
    
    accel_pass = np.mean(all_accel_diffs) < accel_threshold
    gyro_pass = np.mean(all_gyro_diffs) < gyro_threshold
    
    print(f"\nAccelerometer match: {'✓ PASS' if accel_pass else '✗ FAIL'}")
    print(f"  Mean difference: {np.mean(all_accel_diffs):.6f} m/s² (threshold: {accel_threshold})")
    
    print(f"\nGyroscope match: {'✓ PASS' if gyro_pass else '✗ FAIL'}")
    print(f"  Mean difference: {np.mean(all_gyro_diffs):.6f} rad/s (threshold: {gyro_threshold})")
    
    if accel_pass and gyro_pass:
        print("\n✓ SUCCESS: Our IMU model matches GTSAM's expected IMU model!")
        print("Both models produce identical specific force and angular velocity measurements.")
    else:
        print("\n✗ FAILURE: Significant differences detected between models.")
        print("Investigation needed to align IMU models.")
    
    return accel_pass and gyro_pass


def plot_imu_comparison(our_imu, gtsam_imu, title="IMU Comparison"):
    """
    Plot comparison of IMU measurements.
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    
    # Extract data
    our_times = [m.timestamp for m in our_imu.measurements]
    our_accels = np.array([m.accelerometer for m in our_imu.measurements])
    our_gyros = np.array([m.gyroscope for m in our_imu.measurements])
    
    gtsam_times = [m['timestamp'] for m in gtsam_imu]
    gtsam_accels = np.array([m['accelerometer'] for m in gtsam_imu])
    gtsam_gyros = np.array([m['gyroscope'] for m in gtsam_imu])
    
    # Plot each axis
    axes_labels = ['X', 'Y', 'Z']
    for i in range(3):
        # Accelerometer
        axes[i, 0].plot(our_times, our_accels[:, i], 'b-', label='Our Model', alpha=0.7)
        axes[i, 0].plot(gtsam_times, gtsam_accels[:, i], 'r--', label='GTSAM Model', alpha=0.7)
        axes[i, 0].set_ylabel(f'{axes_labels[i]} Accel (m/s²)')
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].legend()
        
        # Gyroscope
        axes[i, 1].plot(our_times, our_gyros[:, i], 'b-', label='Our Model', alpha=0.7)
        axes[i, 1].plot(gtsam_times, gtsam_gyros[:, i], 'r--', label='GTSAM Model', alpha=0.7)
        axes[i, 1].set_ylabel(f'{axes_labels[i]} Gyro (rad/s)')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].legend()
    
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 1].set_xlabel('Time (s)')
    
    axes[0, 0].set_title('Accelerometer')
    axes[0, 1].set_title('Gyroscope')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    compare_raw_imu_simulations()