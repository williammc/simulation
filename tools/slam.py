"""
SLAM estimator command implementation.
Runs EKF, SWBA, or SRIF estimators on simulation data.
"""

import json
import yaml
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import tracemalloc

from rich.console import Console
from rich.progress import Progress, BarColumn, TaskProgressColumn, TimeRemainingColumn

console = Console()


def run_slam(
    estimator: str,
    input_data: Path,
    config: Optional[Path] = None,
    output: Optional[Path] = None,
) -> Optional[Path]:
    """
    Run SLAM estimator on simulation data.
    
    Args:
        estimator: Estimator type (ekf, swba, srif)
        input_data: Path to simulation data JSON file
        config: Path to estimator config YAML file
        output: Output directory for SLAM results
    
    Returns:
        Path to output file if successful, None if error
    """
    # Import estimators
    from src.estimation.ekf_slam import EKFSlam
    from src.estimation.swba_slam import SlidingWindowBA
    from src.estimation.srif_slam import SRIFSlam
    from src.common.json_io import load_simulation_data
    from src.common.config import EKFConfig, SWBAConfig, SRIFConfig
    
    # Validate input
    if not input_data.exists():
        console.print(f"[red]✗ Error: Input file not found: {input_data}[/red]")
        return None
    
    # Validate estimator type
    estimator_lower = estimator.lower()
    if estimator_lower not in ['ekf', 'swba', 'srif']:
        console.print(f"[red]✗ Error: Unknown estimator: {estimator}[/red]")
        console.print("  Available estimators: ekf, swba, srif")
        return None
    
    console.print(f"\n[bold green]Running {estimator.upper()} Estimator[/bold green]")
    console.print(f"  Input: [cyan]{input_data}[/cyan]")
    
    # Load configuration
    estimator_config = None
    if config and config.exists():
        console.print(f"  Config: [cyan]{config}[/cyan]")
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
    else:
        # Use default config
        config = Path(f"config/{estimator_lower}_default.yaml")
        if config.exists():
            console.print(f"  Config: [cyan]{config}[/cyan] (default)")
            with open(config, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            console.print("  Config: Using built-in defaults")
            config_data = {}
    
    # Load simulation data
    console.print("\n[yellow]Loading simulation data...[/yellow]")
    try:
        sim_data = load_simulation_data(str(input_data))
        
        # Extract components
        if isinstance(sim_data, dict):
            trajectory_gt = sim_data.get('trajectory')
            landmarks = sim_data.get('landmarks')
            camera_data = sim_data.get('camera_data')
            imu_data = sim_data.get('imu_data')
            preintegrated_imu = sim_data.get('preintegrated_imu', [])
            camera_calibrations = sim_data.get('camera_calibrations', [])
            imu_calibrations = sim_data.get('imu_calibrations', [])
            metadata = sim_data.get('metadata', {})
        else:
            # Handle object-based format
            trajectory_gt = getattr(sim_data, 'ground_truth_trajectory', None)
            landmarks = getattr(sim_data, 'landmarks', None)
            camera_data = getattr(sim_data, 'camera_measurements', None)
            imu_data = getattr(sim_data, 'imu_measurements', None)
            preintegrated_imu = getattr(sim_data, 'preintegrated_imu', [])
            camera_calibrations = getattr(sim_data, 'camera_calibrations', [])
            imu_calibrations = getattr(sim_data, 'imu_calibrations', [])
            metadata = getattr(sim_data, 'metadata', {})
        
        # Display data info
        if metadata:
            console.print(f"  Trajectory: [cyan]{metadata.get('trajectory_type', 'unknown')}[/cyan]")
            console.print(f"  Duration: [cyan]{metadata.get('duration', 0)}s[/cyan]")
            console.print(f"  Keyframes: [cyan]{metadata.get('num_keyframes', 'N/A')}[/cyan]")
        
    except Exception as e:
        console.print(f"[red]✗ Error loading simulation data: {e}[/red]")
        traceback.print_exc()
        return None
    
    # Get camera calibration (use first camera)
    camera_calib = None
    if camera_calibrations:
        if isinstance(camera_calibrations, list):
            camera_calib = camera_calibrations[0] if len(camera_calibrations) > 0 else None
        elif isinstance(camera_calibrations, dict):
            camera_calib = list(camera_calibrations.values())[0]
        else:
            camera_calib = camera_calibrations
    
    if not camera_calib:
        console.print("[yellow]Warning: No camera calibration found, using defaults[/yellow]")
    
    # Get IMU calibration (use first IMU)
    imu_calib = None
    if imu_calibrations:
        if isinstance(imu_calibrations, list):
            imu_calib = imu_calibrations[0] if len(imu_calibrations) > 0 else None
        else:
            imu_calib = imu_calibrations
    
    # Create estimator
    console.print(f"\n[yellow]Initializing {estimator.upper()} estimator...[/yellow]")
    
    try:
        if estimator_lower == 'ekf':
            # Create EKF config
            ekf_config = EKFConfig(**config_data.get('ekf', {}))
            estimator_instance = EKFSlam(ekf_config, camera_calib, imu_calib)
            
        elif estimator_lower == 'swba':
            # Create SWBA config
            swba_config = SWBAConfig(**config_data.get('swba', {}))
            estimator_instance = SlidingWindowBA(swba_config, camera_calib, imu_calib)
            
        elif estimator_lower == 'srif':
            # Create SRIF config
            srif_config = SRIFConfig(**config_data.get('srif', {}))
            estimator_instance = SRIFSlam(srif_config, camera_calib, imu_calib)
        
    except Exception as e:
        console.print(f"[red]✗ Error creating estimator: {e}[/red]")
        return None
    
    # Initialize estimator with first pose
    if trajectory_gt and len(trajectory_gt.states) > 0:
        initial_pose = trajectory_gt.states[0].pose
        estimator_instance.initialize(initial_pose)
        console.print(f"  Initialized at t={initial_pose.timestamp:.3f}")
    else:
        console.print("[red]✗ Error: No ground truth trajectory found[/red]")
        return None
    
    # Run estimation
    console.print(f"\n[yellow]Running {estimator.upper()} estimation...[/yellow]")
    
    # Track performance
    tracemalloc.start()
    start_time = time.perf_counter()
    
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        
        # Process based on available data
        if preintegrated_imu:
            # Use preintegrated IMU data
            task = progress.add_task(
                f"Processing {len(preintegrated_imu)} preintegrated IMU factors...", 
                total=len(preintegrated_imu)
            )
            
            # Get keyframes if available
            keyframes = []
            if camera_data and hasattr(camera_data, 'frames'):
                keyframes = [f for f in camera_data.frames if f.is_keyframe]
            elif isinstance(camera_data, list):
                keyframes = [f for f in camera_data if getattr(f, 'is_keyframe', False)]
            
            # Process each preintegrated IMU factor
            for i, preint_data in enumerate(preintegrated_imu):
                # Predict with preintegrated IMU
                estimator_instance.predict(preint_data)
                
                # Update with keyframe if available
                if i < len(keyframes):
                    estimator_instance.update(keyframes[i], landmarks)
                
                # Run optimization for SWBA
                if estimator_lower == 'swba' and (i + 1) % 5 == 0:
                    estimator_instance.optimize()
                
                progress.update(task, advance=1)
        
        else:
            # Fallback: process raw IMU if available (shouldn't happen with simplified estimators)
            console.print("[yellow]Warning: No preintegrated IMU found[/yellow]")
            
            # Process camera frames if available
            if camera_data:
                frames = camera_data.frames if hasattr(camera_data, 'frames') else camera_data
                task = progress.add_task(
                    f"Processing {len(frames)} camera frames...", 
                    total=len(frames)
                )
                
                for frame in frames:
                    estimator_instance.update(frame, landmarks)
                    progress.update(task, advance=1)
    
    # Get results
    runtime = time.perf_counter() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    console.print(f"\n[green]✓ Estimation complete![/green]")
    console.print(f"  Runtime: {runtime:.2f} seconds")
    console.print(f"  Peak memory: {peak_mem / 1024 / 1024:.1f} MB")
    
    # Get estimated trajectory
    try:
        result = estimator_instance.get_result()
        estimated_trajectory = result.trajectory
        estimated_landmarks = result.landmarks
    except Exception as e:
        console.print(f"[red]✗ Error getting results: {e}[/red]")
        return None
    
    
    # Save results
    output_dir = output or Path("output/slam")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{estimator_lower}_{timestamp}.json"
    
    # Create output data
    output_data = {
        "metadata": {
            "estimator": estimator_lower,
            "input_file": str(input_data),
            "timestamp": timestamp,
            "runtime_seconds": runtime,
            "peak_memory_mb": peak_mem / 1024 / 1024
        },
        "estimated_trajectory": estimated_trajectory.to_dict() if hasattr(estimated_trajectory, 'to_dict') else {},
        "estimated_landmarks": estimated_landmarks.to_dict() if hasattr(estimated_landmarks, 'to_dict') else {}
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    console.print(f"\n[green]✓ Results saved to:[/green] {output_file}")
    
    # Display summary table
    from rich.table import Table
    
    table = Table(title=f"{estimator.upper()} Results Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Runtime", f"{runtime:.2f} s")
    table.add_row("Peak Memory", f"{peak_mem / 1024 / 1024:.1f} MB")
    
    # Trajectory info
    if estimated_trajectory and hasattr(estimated_trajectory, 'states'):
        table.add_row("Trajectory States", str(len(estimated_trajectory.states)))
    if estimated_landmarks:
        landmark_count = len(estimated_landmarks.landmarks) if hasattr(estimated_landmarks, 'landmarks') else 0
        table.add_row("Landmarks", str(landmark_count))
    
    console.print("\n")
    console.print(table)
    
    return output_file