"""
SLAM estimator command implementation.
Runs EKF, SWBA, or SRIF estimators on simulation data.
"""

import yaml
import time
from pathlib import Path
from typing import Optional, Dict, Any
import tracemalloc
import numpy as np

from rich.console import Console
from rich.progress import Progress, BarColumn, TaskProgressColumn, TimeRemainingColumn
from src.utils.config_loader import ConfigLoader
from src.common.config import EstimatorType

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
    # Legacy estimators (deprecated - will show warnings)
    from src.estimation.legacy.ekf_slam import EKFSlam
    from src.estimation.legacy.swba_slam import SlidingWindowBA
    from src.estimation.legacy.srif_slam import SRIFSlam
    # Import estimators
    from src.estimation.preprocessing import VisualMeasurementPreprocessor
    from src.estimation.base_estimator import EstimatorConfig
    from src.common.json_io import load_simulation_data
    from src.common.config import EKFConfig, SWBAConfig, SRIFConfig
    
    # Validate input
    if not input_data.exists():
        console.print(f"[red]✗ Error: Input file not found: {input_data}[/red]")
        return None
    
    # Validate estimator type
    estimator_lower = estimator.lower()
    valid_estimators = ['ekf', 'swba', 'srif', 'raw-imu-ekf', 'clean-swba', 'cpp-swba']
    if estimator_lower not in valid_estimators:
        console.print(f"[red]✗ Error: Unknown estimator: {estimator}[/red]")
        console.print(f"  Available estimators: {', '.join(valid_estimators)}")
        return None
    
    # Warn about legacy estimators
    if estimator_lower in ['ekf', 'swba', 'srif']:
        console.print(f"[yellow]⚠ Warning: '{estimator}' is a legacy estimator.[/yellow]")
        console.print(f"[yellow]  Consider using 'clean-swba' for better performance.[/yellow]")
    elif estimator_lower == 'clean-swba':
        console.print(f"[green]Using clean SWBA estimator (simplified flow)[/green]")
    elif estimator_lower == 'cpp-swba':
        console.print(f"[green]Using C++ SWBA estimator (camera-model-independent)[/green]")
    
    console.print(f"\n[bold]Running {estimator.upper()} Estimator[/bold]")
    console.print(f"  Input: {input_data}")
    
    # Initialize ConfigLoader
    loader = ConfigLoader(base_path=Path.cwd())
    
    # Load configuration
    if config and config.exists():
        console.print(f"  Config: {config}")
        config_data = loader.load(config)
    else:
        # Use default config
        config = Path(f"config/estimators/{estimator_lower}.yaml")
        if config.exists():
            console.print(f"  Config: {config} (default)")
            config_data = loader.load(config)
        else:
            console.print("  Config: Using built-in defaults")
            config_data = {}
    
    # Load simulation data
    try:
        sim_data = load_simulation_data(str(input_data))
        
        # Extract components
        if isinstance(sim_data, dict):
            trajectory_gt = sim_data.get('trajectory')
            landmarks_data = sim_data.get('landmarks')
            camera_data = sim_data.get('camera_data')
            preintegrated_imu = sim_data.get('preintegrated_imu', [])
            # Try both possible keys for raw IMU
            raw_imu_data = sim_data.get('imu_data') or sim_data.get('imu_measurements')
            raw_imu = []
            if raw_imu_data:
                # If it's an IMUData object, get the measurements list
                if hasattr(raw_imu_data, 'measurements'):
                    raw_imu = raw_imu_data.measurements
                elif isinstance(raw_imu_data, list):
                    raw_imu = raw_imu_data
            camera_calibrations = sim_data.get('camera_calibrations', [])
            imu_calibrations = sim_data.get('imu_calibrations', [])
        else:
            # Handle object-based format
            trajectory_gt = getattr(sim_data, 'ground_truth_trajectory', None)
            landmarks_data = getattr(sim_data, 'landmarks', None)
            camera_data = getattr(sim_data, 'camera_measurements', None)
            preintegrated_imu = getattr(sim_data, 'preintegrated_imu', [])
            # Try both possible attributes for raw IMU
            raw_imu_data = getattr(sim_data, 'imu_data', None) or getattr(sim_data, 'imu_measurements', [])
            raw_imu = []
            if raw_imu_data:
                if hasattr(raw_imu_data, 'measurements'):
                    raw_imu = raw_imu_data.measurements
                elif isinstance(raw_imu_data, list):
                    raw_imu = raw_imu_data
            camera_calibrations = getattr(sim_data, 'camera_calibrations', [])
            imu_calibrations = getattr(sim_data, 'imu_calibrations', [])
        
        # Convert landmarks to Map object
        from src.common.data_structures import Map, Landmark
        landmarks = Map()
        if landmarks_data:
            if isinstance(landmarks_data, dict):
                for lid, lm_data in landmarks_data.items():
                    if isinstance(lm_data, dict):
                        landmarks.add_landmark(Landmark(
                            id=int(lid) if isinstance(lid, str) else lid,
                            position=np.array(lm_data.get('position', [0, 0, 0])),
                            covariance=np.array(lm_data.get('covariance', np.eye(3)))
                        ))
                    elif hasattr(lm_data, 'position'):
                        landmarks.add_landmark(Landmark(
                            id=int(lid) if isinstance(lid, str) else lid,
                            position=np.array(lm_data.position),
                            covariance=np.array(lm_data.covariance) if hasattr(lm_data, 'covariance') else np.eye(3)
                        ))
            elif hasattr(landmarks_data, 'landmarks'):
                # Already a Map object
                landmarks = landmarks_data
        
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
        
        elif estimator_lower == 'raw-imu-ekf':
            # Create EKF config for raw IMU processing
            ekf_config = EKFConfig(**config_data.get('ekf', {}))
            # Override to use raw IMU processing
            ekf_config.use_preintegrated_imu = False
            estimator_instance = EKFSlam(ekf_config, camera_calib, imu_calib, use_preintegrated_imu=False)
            console.print("[cyan]Using raw IMU processing (no preintegration)[/cyan]")
        
        elif estimator_lower == 'clean-swba':
            # Create clean SWBA estimator with simplified flow
            from src.estimation.new.swba_estimator_clean import CleanSWBAEstimator, CleanSWBAConfig
            clean_swba_config = CleanSWBAConfig(
                window_size=config_data.get('clean_swba', {}).get('window_size', 10),
                min_pnp_points=4,
                pnp_reprojection_threshold=3.0,
                max_iterations=10,
                pixel_noise=1.0,
                ideal_noise=0.01
            )
            estimator_instance = CleanSWBAEstimator(clean_swba_config)
            
        elif estimator_lower == 'cpp-swba':
            # C++ SWBA is run as external process - no Python instance needed
            estimator_instance = None
            console.print("[green]Will run C++ SWBA estimator as external process[/green]")
        
    except Exception as e:
        console.print(f"[red]✗ Error creating estimator: {e}[/red]")
        return None
    
    # Initialize estimator with first pose and velocity (skip for cpp-swba)
    if estimator_lower != 'cpp-swba':
        if trajectory_gt and len(trajectory_gt.states) > 0:
            initial_pose = trajectory_gt.states[0].pose
            initial_velocity = trajectory_gt.states[0].velocity if hasattr(trajectory_gt.states[0], 'velocity') else None
            
            # Check if estimator supports initial velocity 
            if estimator_lower in ['ekf', 'swba', 'srif'] and initial_velocity is not None:
                estimator_instance.initialize(initial_pose, initial_velocity=initial_velocity)
                console.print(f"[cyan]Initialized with velocity: [{initial_velocity[0]:.2f}, {initial_velocity[1]:.2f}, {initial_velocity[2]:.2f}] m/s[/cyan]")
            else:
                estimator_instance.initialize(initial_pose)
                if initial_velocity is not None and np.linalg.norm(initial_velocity) > 0.1:
                    console.print(f"[yellow]Warning: Initial velocity [{initial_velocity[0]:.2f}, {initial_velocity[1]:.2f}, {initial_velocity[2]:.2f}] m/s not used[/yellow]")
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
        
        # Process based on available data and estimator type
        if estimator_lower == 'cpp-swba':
            # Run C++ SWBA estimator as external process
            import subprocess
            import os
            
            # Find the C++ executable
            cpp_exe_paths = [
                Path("cpp_estimation/build/examples/run_swba_estimator"),
                Path("../cpp_estimation/build/examples/run_swba_estimator"),
                Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "cpp_estimation/build/examples/run_swba_estimator"
            ]
            
            cpp_exe = None
            for path in cpp_exe_paths:
                if path.exists():
                    cpp_exe = path
                    break
            
            if not cpp_exe:
                console.print("[red]✗ Error: C++ SWBA executable not found. Please build cpp_estimation first.[/red]")
                console.print("  Run: cd cpp_estimation/build && cmake .. && make run_swba_estimator")
                return None
            
            # Prepare output directory
            output_dir = output or Path("output/slam")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Build command
            cmd = [
                str(cpp_exe),
                "--input", str(input_data),
                "--output", str(output_dir)
            ]
            
            # Add config if provided
            if config and config.exists():
                cmd.extend(["--config", str(config)])
            
            # Add verbose flag if needed
            if config_data.get('verbose', False):
                cmd.append("--verbose")
            
            console.print(f"[cyan]Running C++ executable: {cpp_exe.name}[/cyan]")
            
            try:
                # Run the C++ estimator
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                # Show output
                if result.stdout:
                    for line in result.stdout.split('\n'):
                        if line.strip():
                            console.print(f"  {line}")
                
                # Check for output file
                output_file = output_dir / "cpp_swba_result.json"
                if output_file.exists():
                    console.print(f"[green]✓ C++ SWBA completed successfully[/green]")
                    return output_file
                else:
                    console.print("[red]✗ Error: Output file not created[/red]")
                    return None
                    
            except subprocess.CalledProcessError as e:
                console.print(f"[red]✗ C++ SWBA failed with exit code {e.returncode}[/red]")
                if e.stderr:
                    console.print(f"[red]Error output:[/red]")
                    for line in e.stderr.split('\n'):
                        if line.strip():
                            console.print(f"  [red]{line}[/red]")
                return None
            
        elif estimator_lower == 'clean-swba':
            # Special processing for camera-model-independent SWBA
            # Create preprocessor with projection service
            from src.estimation.projection_adapters import PinholeProjectionAdapter
            
            # Create projection adapter using camera calibration
            if camera_calib:
                projection_adapter = PinholeProjectionAdapter(camera_calib)
            else:
                console.print("[yellow]Warning: No camera calibration, using default pinhole model[/yellow]")
                # Create default calibration
                from src.common.data_structures import CameraCalibration
                default_calib = CameraCalibration(
                    camera_id="cam0",
                    image_width=640,
                    image_height=480,
                    K=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]),
                    D=np.zeros(5),
                    model="pinhole"
                )
                projection_adapter = PinholeProjectionAdapter(default_calib)
            
            # Create preprocessor
            preprocessor = VisualMeasurementPreprocessor(
                projection_service=projection_adapter,
                pixel_noise_std=1.0,
                robust_kernel='huber',
                huber_delta=1.0
            )
            
            # Process preintegrated IMU and camera frames
            if preintegrated_imu:
                task = progress.add_task(
                    f"Processing with camera-model-independent pipeline...", 
                    total=len(preintegrated_imu)
                )
                
                # Get keyframes
                keyframes = []
                if camera_data and hasattr(camera_data, 'frames'):
                    keyframes = [f for f in camera_data.frames if f.is_keyframe]
                elif isinstance(camera_data, list):
                    keyframes = [f for f in camera_data if getattr(f, 'is_keyframe', False)]
                
                kf_idx = 0
                for i, preint_data in enumerate(preintegrated_imu):
                    # Convert simulation PreintegratedIMUData to our PreprocessedIMUData interface
                    from src.estimation.interfaces import PreprocessedIMUData
                    converted_imu = PreprocessedIMUData(
                        from_keyframe_id=preint_data.from_keyframe_id,
                        to_keyframe_id=preint_data.to_keyframe_id,
                        delta_position=preint_data.delta_position,
                        delta_velocity=preint_data.delta_velocity,
                        delta_rotation=preint_data.delta_rotation,
                        covariance=preint_data.covariance,
                        delta_t=preint_data.dt,  # Convert dt to delta_t
                        num_measurements=preint_data.num_measurements
                    )
                    
                    # Predict with converted IMU data
                    estimator_instance.predict(converted_imu, converted_imu.delta_t)
                    
                    # Process visual frame if we have a matching keyframe
                    if kf_idx < len(keyframes) and keyframes[kf_idx].timestamp <= preint_data.dt * (i + 1):
                        raw_frame = keyframes[kf_idx]
                        
                        # Get current state for preprocessing
                        from src.common.data_structures import TrajectoryState, Pose
                        current_pose = estimator_instance.current_pose
                        current_state = TrajectoryState(
                            pose=current_pose,
                            velocity=estimator_instance.current_velocity,
                            angular_velocity=None
                        )
                        
                        # Preprocess the frame
                        processed_frame = preprocessor.process_frame(
                            raw_frame,
                            current_state,
                            landmarks,
                            compute_jacobians=True,
                            chi2_threshold=5.991
                        )
                        
                        # Update with processed frame
                        estimator_instance.update(processed_frame, landmarks)
                        kf_idx += 1
                    
                    # Run optimization periodically
                    if (i + 1) % 5 == 0:
                        estimator_instance.optimize()
                    
                    progress.update(task, advance=1)
            else:
                console.print("[yellow]Warning: No preintegrated IMU for new-swba[/yellow]")
        
        elif estimator_lower == 'raw-imu-ekf' and raw_imu:
            # Use raw IMU measurements for raw-imu-ekf
            task = progress.add_task(
                f"Processing {len(raw_imu)} raw IMU measurements...", 
                total=len(raw_imu)
            )
            
            # Get keyframes if available
            keyframes = []
            if camera_data and hasattr(camera_data, 'frames'):
                keyframes = [f for f in camera_data.frames if f.is_keyframe]
            elif isinstance(camera_data, list):
                keyframes = [f for f in camera_data if getattr(f, 'is_keyframe', False)]
            
            # Group raw IMU measurements between keyframes
            imu_idx = 0
            for kf_idx, keyframe in enumerate(keyframes):
                # Collect IMU measurements up to this keyframe
                imu_batch = []
                while imu_idx < len(raw_imu) and raw_imu[imu_idx].timestamp <= keyframe.timestamp:
                    imu_batch.append(raw_imu[imu_idx])
                    progress.update(task, advance=1)
                    imu_idx += 1
                
                # Predict with raw IMU batch
                if imu_batch:
                    estimator_instance.predict(imu_batch)
                
                # Update with keyframe
                estimator_instance.update(keyframe, landmarks)
                
                # Run optimization if needed
                if (kf_idx + 1) % 5 == 0:
                    estimator_instance.optimize()
            
        elif preintegrated_imu:
            # Use preintegrated IMU data for other estimators
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
                
                # Generate visual observations for this timestep if we have landmarks
                # Only do visual updates every 10th keyframe (twice per trajectory)
                # Visual observation generation for EKF and SWBA
                if i % 10 == 0 and landmarks and hasattr(landmarks, 'landmarks') and estimator_lower in ['ekf', 'swba']:
                    # Create a mock camera frame with observations
                    from src.common.data_structures import CameraFrame, CameraObservation, ImagePoint
                    
                    # Get current state estimate
                    # EKF has a 'state' attribute, others use get_state() or get_current_state()
                    current_state = None
                    if hasattr(estimator_instance, 'state') and estimator_instance.state is not None:
                        # EKF case
                        current_state = estimator_instance.state
                    elif hasattr(estimator_instance, 'get_state'):
                        # SWBA/SRIF case
                        try:
                            state_obj = estimator_instance.get_state()
                            # Create a simple object with position and rotation_matrix attributes
                            class SimpleState:
                                def __init__(self, pos, rot):
                                    self.position = pos
                                    self.rotation_matrix = rot
                            current_state = SimpleState(
                                state_obj.robot_pose.position,
                                state_obj.robot_pose.rotation_matrix
                            )
                        except:
                            pass
                    elif hasattr(estimator_instance, 'get_current_state'):
                        # Base estimator case
                        try:
                            state_obj = estimator_instance.get_current_state()
                            class SimpleState:
                                def __init__(self, pos, rot):
                                    self.position = pos
                                    self.rotation_matrix = rot
                            current_state = SimpleState(
                                state_obj.robot_pose.position,
                                state_obj.robot_pose.rotation_matrix
                            )
                        except:
                            pass
                    
                    if current_state is not None:
                        # Project landmarks and create observations
                        observations = []
                        # Iterate over landmark dictionary values
                        landmark_dict = landmarks.landmarks if isinstance(landmarks.landmarks, dict) else {}
                        
                        # Limit to a subset of landmarks (max 50) for stability
                        max_landmarks = 50
                        landmark_items = list(landmark_dict.items())[:max_landmarks]
                        
                        for landmark_id, landmark in landmark_items:
                            # Try to project this landmark
                            predicted_pixel, _ = estimator_instance._predict_measurement(
                                landmark.position,
                                current_state.position,
                                current_state.rotation_matrix
                            )
                            
                            if predicted_pixel is not None:
                                # Add more realistic measurement noise (2 pixel std)
                                pixel_noise = np.random.randn(2) * 2.0  # 2 pixel std is more realistic
                                noisy_pixel = predicted_pixel + pixel_noise
                                
                                # Check if pixel is within reasonable bounds
                                if 0 <= noisy_pixel[0] < 640 and 0 <= noisy_pixel[1] < 480:
                                    # Create observation
                                    obs = CameraObservation(
                                        pixel=ImagePoint(u=noisy_pixel[0], v=noisy_pixel[1]),
                                        landmark_id=landmark.id,
                                        descriptor=landmark.descriptor if hasattr(landmark, 'descriptor') else None
                                    )
                                    observations.append(obs)
                        
                        # Create camera frame with observations
                        if observations:
                            # Get current timestamp from estimator state
                            current_timestamp = 0.0
                            if hasattr(estimator_instance, 'current_state') and estimator_instance.current_state:
                                current_timestamp = estimator_instance.current_state.timestamp
                            elif hasattr(estimator_instance, 'state') and estimator_instance.state:
                                current_timestamp = getattr(estimator_instance.state, 'timestamp', 0.0)
                            
                            camera_frame = CameraFrame(
                                timestamp=current_timestamp,
                                camera_id="cam0",
                                observations=observations,
                                is_keyframe=True
                            )
                            # Update with visual observations
                            estimator_instance.update(camera_frame, landmarks)
                
                # Run optimization for SWBA
                if estimator_lower == 'swba' and (i + 1) % 5 == 0:
                    estimator_instance.optimize()
                
                progress.update(task, advance=1)
        
        else:
            # Fallback: warn if no appropriate IMU data
            if estimator_lower == 'raw-imu-ekf':
                console.print("[yellow]Warning: No raw IMU measurements found[/yellow]")
            else:
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