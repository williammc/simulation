#!/usr/bin/env python3
"""
Utility to inspect and extract ground truth from simulation data.
"""

import json
import numpy as np
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from typing import Optional

app = typer.Typer()
console = Console()


@app.command()
def inspect(
    input_file: Path = typer.Argument(..., help="Simulation JSON file"),
    show_trajectory: bool = typer.Option(True, "--trajectory/--no-trajectory", help="Show trajectory info"),
    show_landmarks: bool = typer.Option(True, "--landmarks/--no-landmarks", help="Show landmarks info"),
    show_noise: bool = typer.Option(True, "--noise/--no-noise", help="Show noise comparison"),
    export_gt: Optional[Path] = typer.Option(None, "--export-gt", help="Export ground truth to separate file")
):
    """Inspect ground truth data in simulation output."""
    
    # Load simulation data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    console.print(f"\n[bold cyan]Simulation Data: {input_file.name}[/bold cyan]\n")
    
    # Check if data has ground truth
    has_gt = 'groundtruth' in data
    has_measurements = 'measurements' in data
    
    if not has_gt:
        console.print("[red]No ground truth found in this file![/red]")
        console.print("This might be real data or SLAM output, not simulation data.")
        return
    
    # Display metadata
    if 'metadata' in data:
        meta = data['metadata']
        console.print("[bold]Metadata:[/bold]")
        console.print(f"  Source: {meta.get('source', 'unknown')}")
        console.print(f"  Version: {meta.get('version', 'unknown')}")
        console.print(f"  Timestamp: {meta.get('timestamp', 'unknown')}")
        if 'trajectory_type' in meta:
            console.print(f"  Trajectory Type: {meta['trajectory_type']}")
        console.print()
    
    # Ground Truth Trajectory
    if show_trajectory and 'trajectory' in data['groundtruth']:
        gt_traj = data['groundtruth']['trajectory']
        console.print("[bold green]Ground Truth Trajectory:[/bold green]")
        console.print(f"  Number of poses: {len(gt_traj)}")
        
        if gt_traj:
            # Time range
            t_start = gt_traj[0]['timestamp']
            t_end = gt_traj[-1]['timestamp']
            console.print(f"  Time range: {t_start:.3f}s - {t_end:.3f}s")
            console.print(f"  Duration: {t_end - t_start:.3f}s")
            
            # Trajectory length
            length = 0.0
            for i in range(1, len(gt_traj)):
                pos1 = np.array(gt_traj[i-1]['position'])
                pos2 = np.array(gt_traj[i]['position'])
                length += np.linalg.norm(pos2 - pos1)
            console.print(f"  Total length: {length:.2f} meters")
            
            # Position bounds
            positions = np.array([p['position'] for p in gt_traj])
            console.print(f"  X range: [{positions[:,0].min():.2f}, {positions[:,0].max():.2f}]")
            console.print(f"  Y range: [{positions[:,1].min():.2f}, {positions[:,1].max():.2f}]")
            console.print(f"  Z range: [{positions[:,2].min():.2f}, {positions[:,2].max():.2f}]")
            
            # Sample first few poses
            console.print("\n  [italic]First 3 poses (ground truth):[/italic]")
            for i in range(min(3, len(gt_traj))):
                p = gt_traj[i]
                console.print(f"    t={p['timestamp']:.3f}: "
                            f"pos=[{p['position'][0]:.3f}, {p['position'][1]:.3f}, {p['position'][2]:.3f}]")
        console.print()
    
    # Ground Truth Landmarks
    if show_landmarks and 'landmarks' in data['groundtruth']:
        landmarks = data['groundtruth']['landmarks']
        console.print("[bold green]Ground Truth Landmarks:[/bold green]")
        console.print(f"  Number of landmarks: {len(landmarks)}")
        
        if landmarks:
            # Position bounds
            positions = np.array([lm['position'] for lm in landmarks])
            console.print(f"  X range: [{positions[:,0].min():.2f}, {positions[:,0].max():.2f}]")
            console.print(f"  Y range: [{positions[:,1].min():.2f}, {positions[:,1].max():.2f}]")
            console.print(f"  Z range: [{positions[:,2].min():.2f}, {positions[:,2].max():.2f}]")
            
            # Sample landmarks
            console.print("\n  [italic]First 5 landmarks (ground truth):[/italic]")
            for i in range(min(5, len(landmarks))):
                lm = landmarks[i]
                console.print(f"    LM {lm['id']}: [{lm['position'][0]:.3f}, {lm['position'][1]:.3f}, {lm['position'][2]:.3f}]")
        console.print()
    
    # Noise Analysis
    if show_noise and has_measurements:
        console.print("[bold yellow]Noise Analysis:[/bold yellow]")
        
        # Check IMU noise
        if 'imu' in data['measurements'] and data['measurements']['imu']:
            imu_meas = data['measurements']['imu']
            console.print(f"  IMU measurements: {len(imu_meas)}")
            
            # Check if we have corresponding ground truth IMU
            if show_trajectory and gt_traj and len(gt_traj) > 1:
                # Estimate noise by comparing with ground truth velocities
                console.print("  [italic]IMU noise characteristics:[/italic]")
                
                # Sample some measurements
                accels = np.array([m['accelerometer'] for m in imu_meas[:100]])
                gyros = np.array([m['gyroscope'] for m in imu_meas[:100]])
                
                accel_std = np.std(accels, axis=0)
                gyro_std = np.std(gyros, axis=0)
                
                console.print(f"    Accel std: [{accel_std[0]:.4f}, {accel_std[1]:.4f}, {accel_std[2]:.4f}] m/s²")
                console.print(f"    Gyro std: [{gyro_std[0]:.4f}, {gyro_std[1]:.4f}, {gyro_std[2]:.4f}] rad/s")
        
        # Check camera noise
        if 'camera_frames' in data['measurements'] and data['measurements']['camera_frames']:
            cam_frames = data['measurements']['camera_frames']
            console.print(f"\n  Camera frames: {len(cam_frames)}")
            
            total_obs = sum(len(f['observations']) for f in cam_frames)
            console.print(f"  Total observations: {total_obs}")
            
            if cam_frames and cam_frames[0]['observations']:
                console.print("  [italic]Note: Camera measurements include pixel noise[/italic]")
        console.print()
    
    # Export ground truth if requested
    if export_gt:
        gt_only = {
            'metadata': data.get('metadata', {}),
            'groundtruth': data.get('groundtruth', {}),
            'calibration': data.get('calibration', {})
        }
        gt_only['metadata']['note'] = 'Ground truth only - extracted from simulation'
        
        with open(export_gt, 'w') as f:
            json.dump(gt_only, f, indent=2)
        
        console.print(f"[green]✓ Ground truth exported to: {export_gt}[/green]")
    
    # Summary table
    table = Table(title="Data Summary")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")
    
    table.add_row(
        "Ground Truth", 
        "✓" if has_gt else "✗",
        f"Trajectory + Landmarks" if has_gt else "Not available"
    )
    table.add_row(
        "Measurements",
        "✓" if has_measurements else "✗", 
        "IMU + Camera (with noise)" if has_measurements else "Not available"
    )
    table.add_row(
        "Calibration",
        "✓" if 'calibration' in data else "✗",
        "Camera + IMU params" if 'calibration' in data else "Not available"
    )
    
    console.print("\n", table)
    
    # Instructions
    console.print("\n[bold]How to use this data:[/bold]")
    console.print("1. [green]Ground truth[/green] is in 'groundtruth.trajectory' and 'groundtruth.landmarks'")
    console.print("2. [yellow]Noisy measurements[/yellow] are in 'measurements.imu' and 'measurements.camera_frames'")
    console.print("3. Run SLAM on the noisy measurements, then compare with ground truth")
    console.print("4. Use './run.sh plot <file> --trajectory' to visualize the ground truth")
    console.print("5. Use './run.sh evaluate' to compute ATE/RPE metrics against ground truth")


@app.command()
def extract_gt(
    input_file: Path = typer.Argument(..., help="Simulation JSON file"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for ground truth")
):
    """Extract only ground truth from simulation data."""
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    if 'groundtruth' not in data:
        console.print("[red]No ground truth found in file![/red]")
        return
    
    # Create output filename if not provided
    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}_groundtruth.json"
    
    # Extract ground truth
    gt_data = {
        'metadata': {
            'source': 'ground_truth_extraction',
            'original_file': str(input_file),
            **data.get('metadata', {})
        },
        'trajectory': data['groundtruth'].get('trajectory', []),
        'landmarks': data['groundtruth'].get('landmarks', [])
    }
    
    with open(output_file, 'w') as f:
        json.dump(gt_data, f, indent=2)
    
    console.print(f"[green]✓ Ground truth extracted to: {output_file}[/green]")


@app.command()
def compare_noise(
    input_file: Path = typer.Argument(..., help="Simulation JSON file with noise")
):
    """Compare noisy measurements with ground truth to see noise levels."""
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    if 'groundtruth' not in data or 'measurements' not in data:
        console.print("[red]File must contain both ground truth and measurements![/red]")
        return
    
    console.print("\n[bold cyan]Noise Comparison Analysis[/bold cyan]\n")
    
    # Compare IMU data if available
    if 'trajectory' in data['groundtruth'] and 'imu' in data['measurements']:
        gt_traj = data['groundtruth']['trajectory']
        imu_meas = data['measurements']['imu']
        
        console.print("[bold]IMU Noise Analysis:[/bold]")
        
        # Compute ground truth accelerations (numerical differentiation)
        if len(gt_traj) > 2:
            # Sample at similar timestamps
            for i in range(min(5, len(imu_meas))):
                imu = imu_meas[i]
                t = imu['timestamp']
                
                # Find closest ground truth poses
                closest_idx = min(range(len(gt_traj)), 
                                key=lambda j: abs(gt_traj[j]['timestamp'] - t))
                
                if closest_idx > 0 and closest_idx < len(gt_traj) - 1:
                    # Estimate true acceleration from ground truth
                    dt = gt_traj[closest_idx + 1]['timestamp'] - gt_traj[closest_idx - 1]['timestamp']
                    if 'velocity' in gt_traj[closest_idx]:
                        v1 = np.array(gt_traj[closest_idx - 1].get('velocity', [0,0,0]))
                        v2 = np.array(gt_traj[closest_idx + 1].get('velocity', [0,0,0]))
                        true_accel = (v2 - v1) / dt if dt > 0 else np.zeros(3)
                        true_accel[2] += 9.81  # Add gravity
                        
                        meas_accel = np.array(imu['accelerometer'])
                        noise = meas_accel - true_accel
                        
                        console.print(f"  t={t:.3f}:")
                        console.print(f"    Measured: [{meas_accel[0]:.3f}, {meas_accel[1]:.3f}, {meas_accel[2]:.3f}]")
                        console.print(f"    True:     [{true_accel[0]:.3f}, {true_accel[1]:.3f}, {true_accel[2]:.3f}]")
                        console.print(f"    Noise:    [{noise[0]:.3f}, {noise[1]:.3f}, {noise[2]:.3f}]")
    
    # Compare camera observations
    if 'camera_frames' in data['measurements'] and 'landmarks' in data['groundtruth']:
        console.print("\n[bold]Camera Noise Analysis:[/bold]")
        console.print("  Camera observations include pixel noise")
        console.print("  Use reprojection error to quantify noise level")
        
        # Get first frame with observations
        for frame in data['measurements']['camera_frames'][:1]:
            if frame['observations']:
                n_obs = len(frame['observations'])
                console.print(f"  Frame at t={frame['timestamp']:.3f}: {n_obs} observations")
                console.print("  [italic]Pixel measurements are perturbed from true projections[/italic]")


if __name__ == "__main__":
    app()