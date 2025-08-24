#!/usr/bin/env python3
"""Quick comparison plot of Ground Truth vs GTSAM vs Our EKF using Plotly."""

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


def test_estimator_comparison():
    """Compare Ground Truth, GTSAM, and Our EKF estimations."""
    
    print("Generating comparison plots...")
    
    # Generate shorter trajectory for faster processing
    params = TrajectoryParams(
        duration=2.0,  # Just 2 seconds
        rate=100.0,
        start_time=0.0
    )
    circle_gen = CircleTrajectory(radius=2.0, height=1.5, params=params)
    trajectory = circle_gen.generate()
    
    # Noiseless IMU
    calibration = IMUCalibration(
        imu_id="perfect",
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
    
    # Keyframes every 0.1s
    keyframe_times = np.arange(0, 2.0, 0.1)
    n_kf = len(keyframe_times)
    
    # Storage
    gt_pos = []
    gtsam_pos = []
    our_pos = []
    
    # Ground truth
    for kf_t in keyframe_times:
        for state in trajectory.states:
            if abs(state.pose.timestamp - kf_t) < 0.001:
                gt_pos.append(state.pose.position.copy())
                break
    
    # Initial state
    p0 = trajectory.states[0].pose.position.copy()
    v0 = trajectory.states[0].velocity.copy()
    R0 = trajectory.states[0].pose.rotation_matrix.copy()
    
    # GTSAM estimation
    params_gtsam = gtsam.PreintegrationParams.MakeSharedU(9.81)
    params_gtsam.setAccelerometerCovariance(np.eye(3) * 1e-6)
    params_gtsam.setGyroscopeCovariance(np.eye(3) * 1e-6)
    
    gtsam_state = gtsam.NavState(gtsam.Rot3(R0), gtsam.Point3(p0), v0)
    gtsam_pos.append(p0.copy())
    
    for i in range(1, n_kf):
        t0, t1 = keyframe_times[i-1], keyframe_times[i]
        
        pim = gtsam.PreintegratedImuMeasurements(
            params_gtsam,
            gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
        )
        
        for meas in imu_data.measurements:
            if t0 <= meas.timestamp <= t1:
                dt = 0.005 if pim.deltaTij() > 0 else meas.timestamp - t0
                if dt > 0:
                    pim.integrateMeasurement(meas.accelerometer, meas.gyroscope, dt)
        
        gtsam_state = pim.predict(gtsam_state, gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3)))
        gtsam_pos.append(np.array(gtsam_state.position()))
    
    # Our EKF
    our_p, our_v, our_R = p0.copy(), v0.copy(), R0.copy()
    our_pos.append(our_p.copy())
    
    for i in range(1, n_kf):
        t0, t1 = keyframe_times[i-1], keyframe_times[i]
        
        preint = IMUPreintegrator()
        for meas in imu_data.measurements:
            if t0 <= meas.timestamp <= t1:
                dt = meas.timestamp - t0 if len(preint.measurements) == 0 else meas.timestamp - preint.measurements[-1].timestamp
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
        
        our_pos.append(our_p.copy())
    
    # Convert to arrays
    gt_pos = np.array(gt_pos)
    gtsam_pos = np.array(gtsam_pos)
    our_pos = np.array(our_pos)
    
    # Compute errors
    gtsam_err = np.linalg.norm(gtsam_pos - gt_pos, axis=1)
    our_err = np.linalg.norm(our_pos - gt_pos, axis=1)
    
    print(f"GTSAM: Mean error = {np.mean(gtsam_err):.4f}m, Max = {np.max(gtsam_err):.4f}m")
    print(f"Ours:  Mean error = {np.mean(our_err):.4f}m, Max = {np.max(our_err):.4f}m")
    
    # Verify GTSAM and our implementation match
    position_diff = np.linalg.norm(gtsam_pos - our_pos, axis=1)
    assert np.max(position_diff) < 1e-10, f"GTSAM and our implementation differ by {np.max(position_diff):.6f}m"
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Trajectory (XY Plane)', 'Position Error over Time',
                       'X Position', 'Y Position'),
        specs=[[{'type': 'xy'}, {'type': 'xy'}],
               [{'type': 'xy'}, {'type': 'xy'}]]
    )
    
    # XY trajectory (subplot 1,1)
    fig.add_trace(
        go.Scatter(x=gt_pos[:, 0], y=gt_pos[:, 1],
                  mode='lines', name='Ground Truth',
                  line=dict(color='green', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=gtsam_pos[:, 0], y=gtsam_pos[:, 1],
                  mode='lines', name='GTSAM',
                  line=dict(color='blue', width=2, dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=our_pos[:, 0], y=our_pos[:, 1],
                  mode='lines', name='Our EKF',
                  line=dict(color='red', width=2, dash='dot')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=[p0[0]], y=[p0[1]],
                  mode='markers', name='Start',
                  marker=dict(color='black', size=10)),
        row=1, col=1
    )
    
    # Error over time (subplot 1,2)
    fig.add_trace(
        go.Scatter(x=keyframe_times, y=gtsam_err,
                  mode='lines', name='GTSAM',
                  line=dict(color='blue', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=keyframe_times, y=our_err,
                  mode='lines', name='Our EKF',
                  line=dict(color='red', width=2)),
        row=1, col=2
    )
    
    # X position (subplot 2,1)
    fig.add_trace(
        go.Scatter(x=keyframe_times, y=gt_pos[:, 0],
                  mode='lines', name='Ground Truth',
                  line=dict(color='green', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=keyframe_times, y=gtsam_pos[:, 0],
                  mode='lines', name='GTSAM',
                  line=dict(color='blue', width=2, dash='dash')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=keyframe_times, y=our_pos[:, 0],
                  mode='lines', name='Our EKF',
                  line=dict(color='red', width=2, dash='dot')),
        row=2, col=1
    )
    
    # Y position (subplot 2,2)
    fig.add_trace(
        go.Scatter(x=keyframe_times, y=gt_pos[:, 1],
                  mode='lines', name='Ground Truth',
                  line=dict(color='green', width=2)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=keyframe_times, y=gtsam_pos[:, 1],
                  mode='lines', name='GTSAM',
                  line=dict(color='blue', width=2, dash='dash')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=keyframe_times, y=our_pos[:, 1],
                  mode='lines', name='Our EKF',
                  line=dict(color='red', width=2, dash='dot')),
        row=2, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="X (m)", row=1, col=1)
    fig.update_yaxes(title_text="Y (m)", scaleanchor="x", scaleratio=1, row=1, col=1)
    
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Position Error (m)", row=1, col=2)
    
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="X (m)", row=2, col=1)
    
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Y (m)", row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title="IMU-only: Ground Truth vs GTSAM vs Our EKF<br>(Circle: r=2m, T=2s, Perfect IMU)",
        height=800,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Save the plot
    output_dir = Path("tests/gtsam-comparison/outputs")
    output_dir.mkdir(exist_ok=True, parents=True)
    fig.write_html(output_dir / "estimator_comparison.html")
    print(f"\nInteractive plot saved to: {output_dir / 'estimator_comparison.html'}")
    
    # Print first few positions for verification
    print("\nFirst 3 keyframes:")
    for i in range(min(3, n_kf)):
        print(f"  t={keyframe_times[i]:.1f}s:")
        print(f"    GT:    {gt_pos[i]}")
        print(f"    GTSAM: {gtsam_pos[i]}")
        print(f"    Ours:  {our_pos[i]}")
        print(f"    GTSAM err: {gtsam_err[i]:.4f}m")
        print(f"    Our err:   {our_err[i]:.4f}m")
    
    # Test completed successfully


if __name__ == "__main__":
    test_estimator_comparison()