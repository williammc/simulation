"""
Simulation command implementation.
Generates synthetic SLAM data with configurable trajectories and sensors.
"""

import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.simulation.trajectory_generator import generate_trajectory
from src.simulation.landmark_generator import generate_landmarks, LandmarkGeneratorConfig
from src.simulation.camera_model import PinholeCamera, generate_camera_observations
from src.simulation.imu_model import IMUModel, IMUNoiseConfig
from src.common.data_structures import (
    CameraCalibration, CameraIntrinsics, CameraExtrinsics, CameraModel,
    IMUCalibration, CameraData, PreintegratedIMUData
)
from src.common.json_io import save_simulation_data
from src.estimation.imu_integration import IMUPreintegrator
from src.utils.preintegration_utils import (
    preintegrate_between_keyframes,
    attach_preintegrated_to_frames,
    PreintegrationCache
)
from src.simulation.keyframe_selector import mark_keyframes_in_camera_data
from src.common.config import KeyframeSelectionConfig
import numpy as np

console = Console()


def run_simulation(
    trajectory: str,
    config: Optional[Path],
    duration: float,
    output: Optional[Path],
    seed: Optional[int],
    noise_config: Optional[Path] = None,
    add_noise: bool = False,
    enable_preintegration: bool = False,
    keyframe_config: Optional[KeyframeSelectionConfig] = None,
) -> int:
    """
    Run simulation to generate synthetic SLAM data.
    
    Args:
        trajectory: Trajectory type (circle, figure8, spiral, line)
        config: Path to simulation config YAML file
        duration: Simulation duration in seconds
        output: Output directory for simulation data
        seed: Random seed for reproducibility
        noise_config: Path to noise configuration YAML file
        add_noise: Enable noise in measurements
    
    Returns:
        Exit code (0 for success)
    """
    console.print(f"[bold green]Running simulation[/bold green]")
    console.print(f"  Trajectory: [cyan]{trajectory}[/cyan]")
    console.print(f"  Duration: [cyan]{duration}s[/cyan]")
    
    if config:
        console.print(f"  Config: [cyan]{config}[/cyan]")
    if seed is not None:
        console.print(f"  Seed: [cyan]{seed}[/cyan]")
    if add_noise:
        console.print(f"  Noise: [cyan]Enabled[/cyan]")
        if noise_config:
            console.print(f"  Noise config: [cyan]{noise_config}[/cyan]")
    
    output_dir = output or Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Load noise configuration
    noise_params = {}
    if noise_config and noise_config.exists():
        with open(noise_config, 'r') as f:
            noise_params = yaml.safe_load(f)
    elif add_noise:
        # Use default noise config file if it exists
        default_noise_config = Path("config/noise_config.yaml")
        if default_noise_config.exists():
            with open(default_noise_config, 'r') as f:
                noise_params = yaml.safe_load(f)
    
    # Load config or use defaults
    if config and config.exists():
        with open(config, 'r') as f:
            params = yaml.safe_load(f)
            # Override preintegration setting from config if present
            if 'preintegration' in params and 'enabled' in params['preintegration']:
                enable_preintegration = params['preintegration']['enabled']
    else:
        # Default parameters for each trajectory type
        if trajectory == "circle":
            params = {
                "radius": 2.0,
                "height": 1.5,
            }
        elif trajectory == "figure8":
            params = {
                "scale_x": 3.0,
                "scale_y": 2.0,
                "height": 1.5,
            }
        elif trajectory == "spiral":
            params = {
                "initial_radius": 0.5,
                "final_radius": 3.0,
                "initial_height": 0.5,
                "final_height": 3.0,
            }
        elif trajectory == "line":
            params = {
                "start_position": [0, 0, 1],
                "end_position": [10, 0, 1],
            }
        else:
            console.print(f"[red]Error: Unknown trajectory type: {trajectory}[/red]")
            return 1
    
    # Add common parameters
    params["duration"] = duration
    params["rate"] = 100.0  # Hz
    params["start_time"] = 0.0
    
    # Simulate with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating trajectory...", total=None)
        
        try:
            # Generate trajectory
            traj = generate_trajectory(trajectory, params)
            
            progress.update(task, description="Generating landmarks...")
            
            # Generate landmarks adaptively around trajectory
            landmark_params = noise_params.get("landmarks", {})
            landmark_config = LandmarkGeneratorConfig(
                num_landmarks=landmark_params.get("num_landmarks", 500),
                distribution=landmark_params.get("distribution", "uniform"),
                min_separation=landmark_params.get("min_separation", 0.1),
                seed=seed
            )
            
            # Use adaptive generation to place landmarks near trajectory
            use_adaptive = landmark_params.get("adaptive", True)
            landmarks = generate_landmarks(
                config=landmark_config,
                trajectory=traj,
                adaptive=use_adaptive
            )
            
            progress.update(task, description="Generating sensor measurements...")
            
            # Create camera calibration
            camera_intrinsics = CameraIntrinsics(
                model=CameraModel.PINHOLE,
                width=640,
                height=480,
                fx=500.0,
                fy=500.0,
                cx=320.0,
                cy=240.0,
                distortion=np.zeros(5)
            )
            
            camera_extrinsics = CameraExtrinsics(
                B_T_C=np.eye(4)  # Camera at body center for simplicity
            )
            
            camera_calib = CameraCalibration(
                camera_id="cam0",
                intrinsics=camera_intrinsics,
                extrinsics=camera_extrinsics
            )
            
            # Create IMU calibration
            imu_calib = IMUCalibration(
                imu_id="imu0",
                accelerometer_noise_density=0.01,
                accelerometer_random_walk=0.001,
                gyroscope_noise_density=0.001,
                gyroscope_random_walk=0.0001,
                rate=200.0
            )
            
            # Generate camera observations
            from src.simulation.camera_model import CameraNoiseConfig
            
            camera_noise_params = noise_params.get("camera", {})
            camera_noise_config = CameraNoiseConfig(
                pixel_noise_std=camera_noise_params.get("pixel_noise_std", 1.0),
                add_noise=add_noise and camera_noise_params.get("add_noise", True),
                outlier_probability=camera_noise_params.get("outlier_probability", 0.01),
                outlier_std=camera_noise_params.get("outlier_std", 10.0),
                seed=seed
            )
            
            camera = PinholeCamera(camera_calib, noise_config=camera_noise_config)
            camera_data = CameraData(camera_id="cam0", rate=30.0)
            
            # Sample camera observations at 30 Hz
            camera_dt = 1.0 / 30.0
            time_range = traj.get_time_range()
            camera_times = np.arange(time_range[0], time_range[1], camera_dt)
            
            for t in camera_times:
                # Get pose at this time
                pose = traj.get_pose_at_time(t)
                if pose is not None:
                    frame = generate_camera_observations(
                        camera, landmarks, pose, t, "cam0"
                    )
                    if len(frame.observations) > 0:  # Only add if there are observations
                        camera_data.add_frame(frame)
            
            # Generate IMU measurements
            imu_noise_params = noise_params.get("imu", {})
            accel_params = imu_noise_params.get("accelerometer", {})
            gyro_params = imu_noise_params.get("gyroscope", {})
            
            imu_noise_config = IMUNoiseConfig(
                accel_noise_density=accel_params.get("noise_density", 0.01),
                accel_random_walk=accel_params.get("random_walk", 0.001),
                accel_bias_stability=accel_params.get("bias_stability", 0.0001),
                gyro_noise_density=gyro_params.get("noise_density", 0.001),
                gyro_random_walk=gyro_params.get("random_walk", 0.0001),
                gyro_bias_stability=gyro_params.get("bias_stability", 0.0001),
                gravity_magnitude=imu_noise_params.get("gravity_magnitude", 9.81),
                seed=seed
            )
            
            imu_model = IMUModel(
                calibration=imu_calib,
                noise_config=imu_noise_config
            )
            
            # Generate with or without noise
            imu_add_noise = add_noise and imu_noise_params.get("add_noise", True)
            if imu_add_noise:
                imu_data = imu_model.generate_noisy_measurements(traj)
            else:
                imu_data = imu_model.generate_perfect_measurements(traj)
            
            # Mark keyframes using selection strategy
            progress.update(task, description="Selecting keyframes...")
            
            # Use provided config or create default
            if keyframe_config is None:
                keyframe_config = KeyframeSelectionConfig()
            
            # Get poses for each camera frame
            frame_poses = []
            for frame in camera_data.frames:
                pose = traj.get_pose_at_time(frame.timestamp)
                if pose is not None:
                    frame_poses.append(pose)
                else:
                    # Use interpolation if exact pose not available
                    frame_poses.append(traj.get_pose_at_time(frame.timestamp))
            
            # Mark keyframes based on selection strategy
            mark_keyframes_in_camera_data(
                camera_data.frames,
                frame_poses,
                keyframe_config
            )
            
            # Count keyframes
            keyframe_count = sum(1 for f in camera_data.frames if f.is_keyframe)
            console.print(f"  [green]✓[/green] Selected {keyframe_count} keyframes using {keyframe_config.strategy.value} strategy")
            
            # Generate preintegrated IMU data if enabled
            preintegrated_imu_data = None
            if enable_preintegration:
                progress.update(task, description="Preintegrating IMU measurements...")
                
                # Get keyframe schedule from marked frames
                keyframe_schedule = [
                    (f.keyframe_id, f.timestamp) 
                    for f in camera_data.frames 
                    if f.is_keyframe
                ]
                
                # Preintegrate IMU between keyframes
                if len(keyframe_schedule) >= 2:
                    keyframe_ids = [kf[0] for kf in keyframe_schedule]
                    keyframe_times = [kf[1] for kf in keyframe_schedule]
                    
                    # Create preintegrator with IMU calibration parameters
                    preintegrator = IMUPreintegrator(
                        accel_noise_density=imu_calib.accelerometer_noise_density,
                        gyro_noise_density=imu_calib.gyroscope_noise_density,
                        accel_random_walk=imu_calib.accelerometer_random_walk,
                        gyro_random_walk=imu_calib.gyroscope_random_walk,
                        gravity=np.array([0, 0, -imu_calib.gravity_magnitude])
                    )
                    
                    # Create cache for efficiency
                    cache = PreintegrationCache()
                    
                    # Preintegrate between keyframes
                    preintegrated_imu_data = preintegrate_between_keyframes(
                        imu_data.measurements,
                        keyframe_ids,
                        keyframe_times,
                        preintegrator,
                        cache
                    )
                    
                    # Attach preintegrated data to camera frames
                    attach_preintegrated_to_frames(
                        camera_data.frames,
                        preintegrated_imu_data
                    )
                    
                    console.print(f"  [green]✓[/green] Preintegrated IMU between {len(keyframe_schedule)} keyframes")
            
            progress.update(task, description="Saving data...")
            
            # Generate output file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"simulation_{trajectory}_{timestamp}.json"
            
            # Save to JSON
            metadata = {
                "trajectory_type": trajectory,
                "duration": duration,
                "coordinate_system": "ENU",
                "seed": seed,
                "preintegration_enabled": enable_preintegration,
                "keyframe_selection_strategy": keyframe_config.strategy.value if keyframe_config else None,
                "num_keyframes": keyframe_count if 'keyframe_count' in locals() else 0
            }
            
            save_simulation_data(
                filepath=output_file,
                trajectory=traj,
                landmarks=landmarks,
                imu_data=imu_data,
                camera_data=camera_data,
                camera_calibrations=[camera_calib],
                imu_calibrations=[imu_calib],
                metadata=metadata,
                preintegrated_imu_data=preintegrated_imu_data
            )
            
        except Exception as e:
            console.print(f"[red]Error during simulation: {e}[/red]")
            return 1
    
    console.print(f"[green]✓[/green] Simulation complete: [cyan]{output_file}[/cyan]")
    
    # Show summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Output size: {output_file.stat().st_size:,} bytes")
    console.print(f"  Trajectory points: {len(traj.states)}")
    console.print(f"  Landmarks: {len(landmarks.landmarks)}")
    console.print(f"  IMU measurements: {len(imu_data.measurements)}")
    console.print(f"  Camera frames: {len(camera_data.frames)}")
    
    # Count total observations
    total_obs = sum(len(frame.observations) for frame in camera_data.frames)
    console.print(f"  Camera observations: {total_obs}")
    
    time_range = traj.get_time_range()
    console.print(f"  Time range: {time_range[0]:.2f}s - {time_range[1]:.2f}s")
    
    return 0