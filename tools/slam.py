"""
SLAM estimator command implementation.
Runs EKF, SWBA, or SRIF estimators on simulation data.
"""

import yaml
import time
from pathlib import Path
from typing import Optional
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
    
    console.print(f"\n[bold]Running {estimator.upper()} Estimator[/bold]")
    console.print(f"  Input: {input_data}")
    
    # Load configuration
    if config and config.exists():
        console.print(f"  Config: {config}")
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
    else:
        # Use default config
        config = Path(f"config/{estimator_lower}_default.yaml")
        if config.exists():
            console.print(f"  Config: {config} (default)")
            with open(config, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            console.print("  Config: Using built-in defaults")
            config_data = {}
    
    # Load simulation data
    try:
        sim_data = load_simulation_data(str(input_data))
        
        # Extract components
        if isinstance(sim_data, dict):
            trajectory_gt = sim_data.get('trajectory')
            landmarks = sim_data.get('landmarks')
            camera_data = sim_data.get('camera_data')
            preintegrated_imu = sim_data.get('preintegrated_imu', [])
            camera_calibrations = sim_data.get('camera_calibrations', [])
            imu_calibrations = sim_data.get('imu_calibrations', [])
        else:
            # Handle object-based format
            trajectory_gt = getattr(sim_data, 'ground_truth_trajectory', None)
            landmarks = getattr(sim_data, 'landmarks', None)
            camera_data = getattr(sim_data, 'camera_measurements', None)
            preintegrated_imu = getattr(sim_data, 'preintegrated_imu', [])
            camera_calibrations = getattr(sim_data, 'camera_calibrations', [])
            imu_calibrations = getattr(sim_data, 'imu_calibrations', [])
        
        
    except Exception as e:
        console.print(f"[red]✗ Error loading simulation data: {e}[/red]")
        import traceback
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
    else:
        console.print("[red]✗ Error: No ground truth trajectory found[/red]")
        return None
    
    # Run estimation
    
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
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    console.print(f"\n[green]✓ Estimation complete[/green] ({runtime:.2f}s, {peak_mem / 1024 / 1024:.1f} MB)")
    
    # Get estimated result
    try:
        result = estimator_instance.get_result()
    except Exception as e:
        console.print(f"[red]✗ Error getting results: {e}[/red]")
        return None
    
    # Update result with runtime
    result.runtime_ms = runtime * 1000
    
    # Save results using EstimatorResultStorage
    from src.estimation.result_io import EstimatorResultStorage
    
    output_dir = output or Path("output/slam")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the estimator config (this is the actual config used by the estimator)
    estimator_config = estimator_instance.config
    
    # Add simulation metadata
    simulation_metadata = {
        "input_file": str(input_data),
        "trajectory_type": sim_data.get("metadata", {}).get("trajectory_type", "unknown") if isinstance(sim_data, dict) else "unknown",
        "duration": sim_data.get("metadata", {}).get("duration", 0) if isinstance(sim_data, dict) else 0,
        "peak_memory_mb": peak_mem / 1024 / 1024
    }
    
    # Save using EstimatorResultStorage
    try:
        output_file = EstimatorResultStorage.save_result(
            result=result,
            config=estimator_config,
            output_path=output_dir,
            simulation_metadata=simulation_metadata
        )
        console.print(f"\n[green]✓ Results saved to:[/green] {output_file}")
        return output_file
    except Exception as e:
        console.print(f"[red]✗ Error saving results: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None