#!/usr/bin/env python3
"""Compare estimation errors at different rotation speeds using Plotly."""

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


def test_speed_comparison():
    """Test how rotation speed affects IMU integration error."""
    
    # Test different periods (speeds)
    periods = [10.0, 5.0, 2.0, 1.0]  # seconds for full rotation
    colors = ['green', 'blue', 'orange', 'red']
    labels = ['T=10s (slow)', 'T=5s', 'T=2s', 'T=1s (fast)']
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Trajectories at Different Speeds', 
                       'Error over Normalized Time',
                       'Error vs Angular Position'),
        specs=[[{'type': 'xy'}, {'type': 'xy'}, {'type': 'xy'}]]
    )
    
    # Storage for summary data
    summary_data = []
    
    for period, color, label in zip(periods, colors, labels):
        print(f"\nProcessing {label}...")
        
        # Generate trajectory
        params = TrajectoryParams(
            duration=period,  # One full rotation
            rate=100.0,
            start_time=0.0
        )
        circle_gen = CircleTrajectory(radius=2.0, height=1.5, params=params)
        trajectory = circle_gen.generate()
        
        # Generate IMU data with perfect (noiseless) IMU
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
        
        # Keyframes at 10% intervals (but not including the very end)
        n_keyframes = 10
        keyframe_times = np.linspace(0, period * 0.9, n_keyframes)
        
        # Get ground truth
        gt_positions = []
        for kf_time in keyframe_times:
            for state in trajectory.states:
                if abs(state.pose.timestamp - kf_time) < 0.005:
                    gt_positions.append(state.pose.position.copy())
                    break
        
        if len(gt_positions) != n_keyframes:
            print(f"  Warning: Ground truth size mismatch. Expected {n_keyframes}, got {len(gt_positions)}")
            continue
        
        gt_positions = np.array(gt_positions)
        
        # Initial state
        p0 = trajectory.states[0].pose.position.copy()
        v0 = trajectory.states[0].velocity.copy()
        R0 = trajectory.states[0].pose.rotation_matrix.copy()
        
        # GTSAM estimation
        params_gtsam = gtsam.PreintegrationParams.MakeSharedU(9.81)
        params_gtsam.setAccelerometerCovariance(np.eye(3) * 1e-6)
        params_gtsam.setGyroscopeCovariance(np.eye(3) * 1e-6)
        
        gtsam_state = gtsam.NavState(gtsam.Rot3(R0), gtsam.Point3(p0), v0)
        gtsam_positions = [p0.copy()]
        
        # Our EKF
        our_p, our_v, our_R = p0.copy(), v0.copy(), R0.copy()
        our_positions = [our_p.copy()]
        
        for i in range(1, n_keyframes):
            t_start = keyframe_times[i-1]
            t_end = keyframe_times[i]
            
            # GTSAM preintegration
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
            
            # Our preintegration
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
        
        gtsam_positions = np.array(gtsam_positions)
        our_positions = np.array(our_positions)
        
        # Compute errors
        gtsam_errors = np.linalg.norm(gtsam_positions - gt_positions, axis=1)
        our_errors = np.linalg.norm(our_positions - gt_positions, axis=1)
        
        # Verify GTSAM and our implementation match
        position_diff = np.linalg.norm(gtsam_positions - our_positions, axis=1)
        assert np.max(position_diff) < 1e-10, f"GTSAM and our implementation differ by {np.max(position_diff):.6f}m"
        
        # Compute statistics
        omega = 2 * np.pi / period
        centripetal = omega**2 * 2.0  # omega^2 * r
        
        summary_data.append({
            'period': period,
            'omega': omega,
            'centripetal': centripetal,
            'gtsam_final_error': gtsam_errors[-1],
            'our_final_error': our_errors[-1]
        })
        
        print(f"  Angular velocity: {omega:.3f} rad/s")
        print(f"  Centripetal accel: {centripetal:.1f} m/s²")
        print(f"  GTSAM final error: {gtsam_errors[-1]:.4f} m")
        print(f"  Our final error: {our_errors[-1]:.4f} m")
        
        # Plot trajectory (subplot 1)
        fig.add_trace(
            go.Scatter(x=gt_positions[:, 0], y=gt_positions[:, 1],
                      mode='lines', name=f'{label} (GT)',
                      line=dict(color=color, width=2),
                      showlegend=True),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=gtsam_positions[:, 0], y=gtsam_positions[:, 1],
                      mode='lines', name=f'{label} (GTSAM)',
                      line=dict(color=color, width=1, dash='dash'),
                      showlegend=False),
            row=1, col=1
        )
        
        # Plot error over normalized time (subplot 2)
        normalized_time = keyframe_times / period
        fig.add_trace(
            go.Scatter(x=normalized_time, y=gtsam_errors,
                      mode='lines', name=label,
                      line=dict(color=color, width=2)),
            row=1, col=2
        )
        
        # Plot error vs angular position (subplot 3)
        angular_positions = normalized_time * 2 * np.pi
        fig.add_trace(
            go.Scatter(x=angular_positions, y=gtsam_errors,
                      mode='lines', name=label,
                      line=dict(color=color, width=2),
                      showlegend=False),
            row=1, col=3
        )
    
    # Add start point marker
    fig.add_trace(
        go.Scatter(x=[2], y=[0], mode='markers',
                  marker=dict(color='black', size=10),
                  name='Start', showlegend=False),
        row=1, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text="X (m)", row=1, col=1)
    fig.update_yaxes(title_text="Y (m)", scaleanchor="x", scaleratio=1, row=1, col=1)
    
    fig.update_xaxes(title_text="Normalized Time (fraction of rotation)", row=1, col=2)
    fig.update_yaxes(title_text="Position Error (m)", row=1, col=2)
    
    fig.update_xaxes(title_text="Angular Position (radians)", row=1, col=3)
    fig.update_yaxes(title_text="Position Error (m)", row=1, col=3)
    
    fig.update_layout(
        title="Effect of Rotation Speed on IMU Integration Error<br>(Perfect IMU, r=2m)",
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Save the plot
    output_dir = Path("tests/gtsam-comparison/outputs")
    output_dir.mkdir(exist_ok=True, parents=True)
    fig.write_html(output_dir / "speed_comparison.html")
    print(f"\nInteractive plot saved to: {output_dir / 'speed_comparison.html'}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Final Error after One Full Rotation")
    print("="*60)
    print(f"{'Period':<10} {'Angular Vel':<15} {'Centripetal':<15} {'Final Error':<12}")
    print(f"{'(s)':<10} {'(rad/s)':<15} {'(m/s²)':<15} {'(m)':<12}")
    print("-"*60)
    for data in summary_data:
        print(f"{data['period']:<10.1f} {data['omega']:<15.3f} {data['centripetal']:<15.1f} {data['gtsam_final_error']:<12.4f}")
    
    # Verify error increases with speed
    errors = [d['gtsam_final_error'] for d in summary_data]
    assert all(errors[i] < errors[i+1] for i in range(len(errors)-1)), "Error should increase with rotation speed"
    
    # Test completed successfully


if __name__ == "__main__":
    test_speed_comparison()