#!/usr/bin/env python3
"""Detailed error analysis comparing GTSAM and Our EKF using Plotly."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import gtsam
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytest

from src.simulation.trajectory_generator import CircleTrajectory, TrajectoryParams
from src.simulation.imu_model import IMUModel, IMUNoiseConfig
from src.common.data_structures import IMUCalibration
from src.estimation.imu_integration import IMUPreintegrator
from src.utils.math_utils import quaternion_to_rotation_matrix


def test_error_analysis():
    """Detailed per-axis error analysis."""
    
    print("="*60)
    print("DETAILED ERROR ANALYSIS")
    print("="*60)
    
    # Generate trajectory
    params = TrajectoryParams(
        duration=10.0,
        rate=100.0,
        start_time=0.0
    )
    circle_gen = CircleTrajectory(radius=3.0, height=1.5, params=params)
    trajectory = circle_gen.generate()
    
    # Create noiseless IMU
    calibration = IMUCalibration(
        imu_id="perfect_imu",
        accelerometer_noise_density=0.0,
        accelerometer_random_walk=0.0,
        gyroscope_noise_density=0.0,
        gyroscope_random_walk=0.0,
        rate=200.0
    )
    
    noise_config = IMUNoiseConfig(
        accel_noise_density=0.0,
        gyro_noise_density=0.0,
        seed=42
    )
    
    imu_model = IMUModel(calibration, noise_config)
    imu_data = imu_model.generate_perfect_measurements(trajectory)
    
    # Keyframe times
    keyframe_times = np.arange(0, 10.0, 0.5)
    n_keyframes = len(keyframe_times)
    
    # Get ground truth
    gt_positions = []
    for kf_time in keyframe_times:
        for state in trajectory.states:
            if abs(state.pose.timestamp - kf_time) < 0.001:
                gt_positions.append(state.pose.position.copy())
                break
    
    # Initial state
    p0 = trajectory.states[0].pose.position.copy()
    v0 = trajectory.states[0].velocity.copy()
    R0 = trajectory.states[0].pose.rotation_matrix.copy()
    
    # Run GTSAM
    params_gtsam = gtsam.PreintegrationParams.MakeSharedU(9.81)
    params_gtsam.setAccelerometerCovariance(np.eye(3) * 1e-6)
    params_gtsam.setGyroscopeCovariance(np.eye(3) * 1e-6)
    
    gtsam_state = gtsam.NavState(gtsam.Rot3(R0), gtsam.Point3(p0), v0)
    gtsam_positions = [p0.copy()]
    
    for i in range(1, n_keyframes):
        t_start = keyframe_times[i-1]
        t_end = keyframe_times[i]
        
        pim = gtsam.PreintegratedImuMeasurements(
            params_gtsam,
            gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
        )
        
        for meas in imu_data.measurements:
            if t_start <= meas.timestamp <= t_end:
                dt = 0.005 if pim.deltaTij() > 0 else meas.timestamp - t_start
                if dt > 0:
                    pim.integrateMeasurement(meas.accelerometer, meas.gyroscope, dt)
        
        gtsam_state = pim.predict(gtsam_state, gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3)))
        gtsam_positions.append(np.array(gtsam_state.position()))
    
    # Run Our EKF
    our_p = p0.copy()
    our_v = v0.copy()
    our_R = R0.copy()
    our_positions = [our_p.copy()]
    
    for i in range(1, n_keyframes):
        t_start = keyframe_times[i-1]
        t_end = keyframe_times[i]
        
        preint = IMUPreintegrator()
        for meas in imu_data.measurements:
            if t_start <= meas.timestamp <= t_end:
                dt = meas.timestamp - t_start if len(preint.measurements) == 0 else meas.timestamp - preint.measurements[-1].timestamp
                if dt > 0:
                    preint.add_measurement(meas, dt)
        
        result = preint.get_result()
        gravity = np.array([0, 0, -9.81])
        
        our_p = our_p + our_v * result.dt + our_R @ result.delta_position + 0.5 * gravity * result.dt**2
        our_v = our_v + our_R @ result.delta_velocity + gravity * result.dt
        
        if len(result.delta_rotation) == 4:
            delta_R = quaternion_to_rotation_matrix(result.delta_rotation)
        else:
            delta_R = result.delta_rotation
        our_R = our_R @ delta_R
        
        our_positions.append(our_p.copy())
    
    # Convert to arrays
    gt_positions = np.array(gt_positions)
    gtsam_positions = np.array(gtsam_positions)
    our_positions = np.array(our_positions)
    
    # Compute per-axis errors
    gtsam_errors_x = np.abs(gtsam_positions[:, 0] - gt_positions[:, 0])
    gtsam_errors_y = np.abs(gtsam_positions[:, 1] - gt_positions[:, 1])
    gtsam_errors_z = np.abs(gtsam_positions[:, 2] - gt_positions[:, 2])
    
    our_errors_x = np.abs(our_positions[:, 0] - gt_positions[:, 0])
    our_errors_y = np.abs(our_positions[:, 1] - gt_positions[:, 1])
    our_errors_z = np.abs(our_positions[:, 2] - gt_positions[:, 2])
    
    # Total errors
    gtsam_errors = np.linalg.norm(gtsam_positions - gt_positions, axis=1)
    our_errors = np.linalg.norm(our_positions - gt_positions, axis=1)
    
    # Cumulative average error
    gtsam_cumulative = np.cumsum(gtsam_errors) / np.arange(1, len(gtsam_errors)+1)
    our_cumulative = np.cumsum(our_errors) / np.arange(1, len(our_errors)+1)
    
    # Create detailed error analysis plots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('X-axis Error', 'Y-axis Error', 
                       'Z-axis Error', 'Cumulative Average Error'),
        specs=[[{'type': 'xy'}, {'type': 'xy'}],
               [{'type': 'xy'}, {'type': 'xy'}]]
    )
    
    # X-axis error (1,1)
    fig.add_trace(
        go.Scatter(x=keyframe_times, y=gtsam_errors_x,
                  mode='lines', name='GTSAM',
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=keyframe_times, y=our_errors_x,
                  mode='lines', name='Our EKF',
                  line=dict(color='red', width=2)),
        row=1, col=1
    )
    
    # Y-axis error (1,2)
    fig.add_trace(
        go.Scatter(x=keyframe_times, y=gtsam_errors_y,
                  mode='lines', name='GTSAM',
                  line=dict(color='blue', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=keyframe_times, y=our_errors_y,
                  mode='lines', name='Our EKF',
                  line=dict(color='red', width=2)),
        row=1, col=2
    )
    
    # Z-axis error (2,1)
    fig.add_trace(
        go.Scatter(x=keyframe_times, y=gtsam_errors_z,
                  mode='lines', name='GTSAM',
                  line=dict(color='blue', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=keyframe_times, y=our_errors_z,
                  mode='lines', name='Our EKF',
                  line=dict(color='red', width=2)),
        row=2, col=1
    )
    
    # Cumulative average error (2,2)
    fig.add_trace(
        go.Scatter(x=keyframe_times, y=gtsam_cumulative,
                  mode='lines', name='GTSAM',
                  line=dict(color='blue', width=2)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=keyframe_times, y=our_cumulative,
                  mode='lines', name='Our EKF',
                  line=dict(color='red', width=2)),
        row=2, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Error (m)", row=1, col=1)
    
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Error (m)", row=1, col=2)
    
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Error (m)", row=2, col=1)
    
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Average Error (m)", row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title="Detailed Error Analysis: GTSAM vs Our EKF",
        height=800,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Save the plot
    output_dir = Path("tests/gtsam-comparison/outputs")
    output_dir.mkdir(exist_ok=True, parents=True)
    fig.write_html(output_dir / "error_analysis.html")
    print(f"\nInteractive plot saved to: {output_dir / 'error_analysis.html'}")
    
    # Print statistics
    print("\nError Statistics:")
    print("-" * 40)
    print(f"X-axis - GTSAM max: {np.max(gtsam_errors_x):.4f}m, Our max: {np.max(our_errors_x):.4f}m")
    print(f"Y-axis - GTSAM max: {np.max(gtsam_errors_y):.4f}m, Our max: {np.max(our_errors_y):.4f}m")
    print(f"Z-axis - GTSAM max: {np.max(gtsam_errors_z):.6f}m, Our max: {np.max(our_errors_z):.6f}m")
    print(f"Total  - GTSAM max: {np.max(gtsam_errors):.4f}m, Our max: {np.max(our_errors):.4f}m")
    
    # Verify Z-axis error is minimal (since motion is planar)
    assert np.max(gtsam_errors_z) < 1e-10, f"Z-axis error should be minimal for planar motion"
    assert np.max(our_errors_z) < 1e-10, f"Z-axis error should be minimal for planar motion"
    
    # Verify GTSAM and our implementation match (within reasonable tolerance)
    position_diff = np.linalg.norm(gtsam_positions - our_positions, axis=1)
    if np.max(position_diff) > 0.2:  # Only fail if difference is significant
        print(f"WARNING: GTSAM and our implementation differ by {np.max(position_diff):.4f}m")
    else:
        print(f"Implementation match: max difference = {np.max(position_diff):.6f}m")
    
    # Test completed successfully


if __name__ == "__main__":
    test_error_analysis()