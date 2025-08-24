#!/usr/bin/env python3
"""Detailed comparison of Ground Truth, GTSAM, and Our EKF with 3D visualization using Plotly."""

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


def test_detailed_comparison():
    """Detailed comparison with longer trajectory and 3D visualization."""
    
    print("="*60)
    print("DETAILED COMPARISON: Ground Truth vs GTSAM vs Our EKF")
    print("="*60)
    
    # Generate perfect circular trajectory (slower for better accuracy)
    params = TrajectoryParams(
        duration=10.0,  # 10 seconds for full circle (slower rotation)
        rate=100.0,
        start_time=0.0
    )
    circle_gen = CircleTrajectory(radius=3.0, height=1.5, params=params)
    trajectory = circle_gen.generate()
    
    print(f"Generated trajectory: radius=3m, period=10s")
    print(f"  Angular velocity: {2*np.pi/10:.3f} rad/s")
    print(f"  Centripetal acceleration: {(2*np.pi/10)**2 * 3:.3f} m/s²")
    
    # Create noiseless IMU
    calibration = IMUCalibration(
        imu_id="perfect_imu",
        accelerometer_noise_density=0.0,
        accelerometer_random_walk=0.0,
        gyroscope_noise_density=0.0,
        gyroscope_random_walk=0.0,
        rate=200.0  # 200 Hz
    )
    
    noise_config = IMUNoiseConfig(
        accel_noise_density=0.0,
        accel_random_walk=0.0,
        gyro_noise_density=0.0,
        gyro_random_walk=0.0,
        accel_bias_initial=np.zeros(3),
        gyro_bias_initial=np.zeros(3),
        gravity_magnitude=9.81,
        seed=42
    )
    
    imu_model = IMUModel(calibration, noise_config)
    imu_data = imu_model.generate_perfect_measurements(trajectory)
    
    print(f"Generated {len(imu_data.measurements)} IMU measurements")
    
    # Keyframe times (every 0.5 seconds)
    keyframe_times = np.arange(0, 10.0, 0.5)
    n_keyframes = len(keyframe_times)
    
    # Storage for trajectories
    gt_positions = []
    gtsam_positions = []
    our_positions = []
    
    # Get ground truth at keyframes
    for kf_time in keyframe_times:
        for state in trajectory.states:
            if abs(state.pose.timestamp - kf_time) < 0.001:
                gt_positions.append(state.pose.position.copy())
                break
    
    # Initial state
    initial_state = trajectory.states[0]
    p0 = initial_state.pose.position.copy()
    v0 = initial_state.velocity.copy()
    R0 = initial_state.pose.rotation_matrix.copy()
    
    print(f"\nProcessing {n_keyframes} keyframes...")
    
    # GTSAM Estimation
    print("\nGTSAM Estimation:")
    gravity = 9.81
    params_gtsam = gtsam.PreintegrationParams.MakeSharedU(gravity)
    params_gtsam.setAccelerometerCovariance(np.eye(3) * 1e-6)
    params_gtsam.setGyroscopeCovariance(np.eye(3) * 1e-6)
    params_gtsam.setIntegrationCovariance(np.eye(3) * 1e-9)
    
    gtsam_state = gtsam.NavState(
        gtsam.Rot3(R0),
        gtsam.Point3(p0),
        v0
    )
    gtsam_positions.append(p0.copy())
    
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
    
    # Our EKF Estimation
    print("Our EKF Estimation:")
    our_p = p0.copy()
    our_v = v0.copy()
    our_R = R0.copy()
    our_positions.append(our_p.copy())
    
    for i in range(1, n_keyframes):
        t_start = keyframe_times[i-1]
        t_end = keyframe_times[i]
        
        our_preint = IMUPreintegrator()
        
        for meas in imu_data.measurements:
            if t_start <= meas.timestamp <= t_end:
                if len(our_preint.measurements) == 0:
                    dt = meas.timestamp - t_start
                else:
                    dt = meas.timestamp - our_preint.measurements[-1].timestamp
                if dt > 0:
                    our_preint.add_measurement(meas, dt)
        
        result = our_preint.get_result()
        
        gravity_vec = np.array([0, 0, -9.81])
        dt = result.dt
        
        our_p = our_p + our_v * dt + our_R @ result.delta_position + 0.5 * gravity_vec * dt**2
        our_v = our_v + our_R @ result.delta_velocity + gravity_vec * dt
        
        if len(result.delta_rotation) == 4:
            delta_R = quaternion_to_rotation_matrix(result.delta_rotation)
        else:
            delta_R = result.delta_rotation
        our_R = our_R @ delta_R
        
        our_positions.append(our_p.copy())
    
    # Convert to numpy arrays
    gt_positions = np.array(gt_positions)
    gtsam_positions = np.array(gtsam_positions)
    our_positions = np.array(our_positions)
    
    # Compute Errors
    gtsam_errors = np.linalg.norm(gtsam_positions - gt_positions, axis=1)
    our_errors = np.linalg.norm(our_positions - gt_positions, axis=1)
    
    print(f"\nGTSAM Errors:")
    print(f"  Mean: {np.mean(gtsam_errors):.4f} m")
    print(f"  Max:  {np.max(gtsam_errors):.4f} m")
    print(f"  Final: {gtsam_errors[-1]:.4f} m")
    
    print(f"\nOur EKF Errors:")
    print(f"  Mean: {np.mean(our_errors):.4f} m")
    print(f"  Max:  {np.max(our_errors):.4f} m")
    print(f"  Final: {our_errors[-1]:.4f} m")
    
    # Verify GTSAM and our implementation match
    position_diff = np.linalg.norm(gtsam_positions - our_positions, axis=1)
    assert np.max(position_diff) < 1e-9, f"GTSAM and our implementation differ by {np.max(position_diff):.6f}m"
    
    # Create comprehensive plots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('3D Trajectories', 'XY Plane (Top View)', 'Position Error over Time',
                       'X Position over Time', 'Y Position over Time', 'Error Difference (Our - GTSAM)'),
        specs=[[{'type': 'scatter3d'}, {'type': 'xy'}, {'type': 'xy'}],
               [{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}]],
        column_widths=[0.35, 0.325, 0.325],
        row_heights=[0.5, 0.5]
    )
    
    # 3D Trajectory Plot (1,1)
    fig.add_trace(
        go.Scatter3d(x=gt_positions[:, 0], y=gt_positions[:, 1], z=gt_positions[:, 2],
                    mode='lines', name='Ground Truth',
                    line=dict(color='green', width=3)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter3d(x=gtsam_positions[:, 0], y=gtsam_positions[:, 1], z=gtsam_positions[:, 2],
                    mode='lines', name='GTSAM',
                    line=dict(color='blue', width=3, dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter3d(x=our_positions[:, 0], y=our_positions[:, 1], z=our_positions[:, 2],
                    mode='lines', name='Our EKF',
                    line=dict(color='red', width=3, dash='dot')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter3d(x=[gt_positions[0, 0]], y=[gt_positions[0, 1]], z=[gt_positions[0, 2]],
                    mode='markers', name='Start',
                    marker=dict(color='black', size=8)),
        row=1, col=1
    )
    
    # XY Plane View (1,2)
    fig.add_trace(
        go.Scatter(x=gt_positions[:, 0], y=gt_positions[:, 1],
                  mode='lines', name='Ground Truth',
                  line=dict(color='green', width=2), showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=gtsam_positions[:, 0], y=gtsam_positions[:, 1],
                  mode='lines', name='GTSAM',
                  line=dict(color='blue', width=2, dash='dash'), showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=our_positions[:, 0], y=our_positions[:, 1],
                  mode='lines', name='Our EKF',
                  line=dict(color='red', width=2, dash='dot'), showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=[gt_positions[0, 0]], y=[gt_positions[0, 1]],
                  mode='markers', marker=dict(color='black', size=10),
                  showlegend=False),
        row=1, col=2
    )
    
    # Error over Time (1,3)
    fig.add_trace(
        go.Scatter(x=keyframe_times, y=gtsam_errors,
                  mode='lines', name='GTSAM Error',
                  line=dict(color='blue', width=2)),
        row=1, col=3
    )
    fig.add_trace(
        go.Scatter(x=keyframe_times, y=our_errors,
                  mode='lines', name='Our EKF Error',
                  line=dict(color='red', width=2)),
        row=1, col=3
    )
    
    # X Position over Time (2,1)
    fig.add_trace(
        go.Scatter(x=keyframe_times, y=gt_positions[:, 0],
                  mode='lines', name='Ground Truth',
                  line=dict(color='green', width=2), showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=keyframe_times, y=gtsam_positions[:, 0],
                  mode='lines', name='GTSAM',
                  line=dict(color='blue', width=2, dash='dash'), showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=keyframe_times, y=our_positions[:, 0],
                  mode='lines', name='Our EKF',
                  line=dict(color='red', width=2, dash='dot'), showlegend=False),
        row=2, col=1
    )
    
    # Y Position over Time (2,2)
    fig.add_trace(
        go.Scatter(x=keyframe_times, y=gt_positions[:, 1],
                  mode='lines', name='Ground Truth',
                  line=dict(color='green', width=2), showlegend=False),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=keyframe_times, y=gtsam_positions[:, 1],
                  mode='lines', name='GTSAM',
                  line=dict(color='blue', width=2, dash='dash'), showlegend=False),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=keyframe_times, y=our_positions[:, 1],
                  mode='lines', name='Our EKF',
                  line=dict(color='red', width=2, dash='dot'), showlegend=False),
        row=2, col=2
    )
    
    # Error Difference (2,3)
    error_diff = our_errors - gtsam_errors
    fig.add_trace(
        go.Scatter(x=keyframe_times, y=error_diff,
                  mode='lines', name='Error Difference',
                  line=dict(color='black', width=2)),
        row=2, col=3
    )
    fig.add_trace(
        go.Scatter(x=keyframe_times, y=np.zeros_like(keyframe_times),
                  mode='lines', line=dict(color='gray', width=1, dash='dash'),
                  showlegend=False),
        row=2, col=3
    )
    
    # Add filled areas for error difference
    fig.add_trace(
        go.Scatter(x=np.concatenate([keyframe_times[error_diff > 0], keyframe_times[error_diff > 0][::-1]]),
                  y=np.concatenate([error_diff[error_diff > 0], np.zeros(np.sum(error_diff > 0))]),
                  fill='toself', fillcolor='rgba(255,0,0,0.2)',
                  line=dict(color='rgba(255,255,255,0)'),
                  name='Our worse', showlegend=True),
        row=2, col=3
    )
    fig.add_trace(
        go.Scatter(x=np.concatenate([keyframe_times[error_diff <= 0], keyframe_times[error_diff <= 0][::-1]]),
                  y=np.concatenate([error_diff[error_diff <= 0], np.zeros(np.sum(error_diff <= 0))]),
                  fill='toself', fillcolor='rgba(0,255,0,0.2)',
                  line=dict(color='rgba(255,255,255,0)'),
                  name='Our better', showlegend=True),
        row=2, col=3
    )
    
    # Update axes
    fig.update_scenes(dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    ), row=1, col=1)
    
    fig.update_xaxes(title_text="X (m)", row=1, col=2)
    fig.update_yaxes(title_text="Y (m)", scaleanchor="x", scaleratio=1, row=1, col=2)
    
    fig.update_xaxes(title_text="Time (s)", row=1, col=3)
    fig.update_yaxes(title_text="Position Error (m)", row=1, col=3)
    
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="X Position (m)", row=2, col=1)
    
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Y Position (m)", row=2, col=2)
    
    fig.update_xaxes(title_text="Time (s)", row=2, col=3)
    fig.update_yaxes(title_text="Error Difference (m)", row=2, col=3)
    
    # Update layout
    fig.update_layout(
        title="IMU-only Estimation Comparison<br>Circle: r=3m, T=10s, Perfect IMU",
        height=900,
        showlegend=True,
        hovermode='closest'
    )
    
    # Save the plot
    output_dir = Path("tests/gtsam-comparison/outputs")
    output_dir.mkdir(exist_ok=True, parents=True)
    fig.write_html(output_dir / "detailed_comparison.html")
    print(f"\nInteractive plot saved to: {output_dir / 'detailed_comparison.html'}")
    
    # Test completed successfully


if __name__ == "__main__":
    test_detailed_comparison()