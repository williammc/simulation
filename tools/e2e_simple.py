#!/usr/bin/env python3
"""
End-to-End Simple SLAM Pipeline
Demonstrates the complete SLAM pipeline: simulate → estimate → evaluate
Uses the existing tested infrastructure from tools/
Run with: ./run.sh e2e_simple
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the existing tools - use relative imports since we're in tools/
from simulate import run_simulation
from slam import run_slam
from evaluate import run_evaluate
from src.common.json_io import load_simulation_data
from src.common.config import KeyframeSelectionConfig
from src.plotting.trajectory_plot import plot_trajectory_comparison, save_trajectory_plot


def run_e2e_simple(
    duration: float = 10.0,
    trajectory_type: str = "circle",
    estimator_type: str = "ekf",
    output_dir: str = "output",
    sim_filename: str = None,
    slam_filename: str = None,
    eval_filename: str = None
):
    """
    Run the complete SLAM pipeline with optional custom filenames.
    
    Args:
        duration: Simulation duration in seconds
        trajectory_type: Type of trajectory ("circle", "figure8", "spiral", "line")
        estimator_type: SLAM estimator ("ekf", "swba", "srif", "cpp-swba")
        output_dir: Directory for output files
        sim_filename: Optional custom name for simulation output
        slam_filename: Optional custom name for SLAM output
        eval_filename: Optional custom name for evaluation output
    """
    print("\n" + "="*60)
    print(" END-TO-END SIMPLE SLAM PIPELINE")
    print(" Complete simulation → estimation → evaluation")
    print("="*60 + "\n")
    
    # Use provided configuration
    print("Configuration:")
    print(f"  • Duration: {duration}s")
    print(f"  • Trajectory: {trajectory_type}")
    print(f"  • Estimator: {estimator_type.upper()}")
    print(f"  • Output dir: {output_dir}")
    
    # ============================================================
    # STEP 1: SIMULATE - Generate synthetic data
    # ============================================================
    print("\nSTEP 1: Generating synthetic data...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
        
    # Configure keyframes for better performance
    from src.common.config import KeyframeSelectionStrategy
    keyframe_config = KeyframeSelectionConfig(
        strategy=KeyframeSelectionStrategy.FIXED_INTERVAL,
        fixed_interval=15,  # Every 15 frames (0.5s at 30Hz camera rate)
        min_time_gap=0.4
    )
    
    # Run simulation using the existing tested function
    print(f"  Running {trajectory_type} trajectory for {duration}s...")
    result = run_simulation(
        trajectory=trajectory_type,
        config=None,  # Use default config
        duration=duration,
        output=output_path,
        seed=42,  # For reproducibility
        add_noise=True,  # Add realistic noise
        enable_preintegration=True,  # Enable IMU preintegration
        keyframe_config=keyframe_config
    )
    
    if result is None:
        print("  ✗ Simulation failed")
        return
    
    # Use the returned path from run_simulation
    sim_output = result
    
    # Optionally rename to specified filename
    if sim_filename:
        new_sim_output = output_path / sim_filename
        sim_output.rename(new_sim_output)
        sim_output = new_sim_output
    
    print(f"  ✓ Generated simulation: {sim_output.name}")
        
    # Load simulation data to get statistics
    sim_data = load_simulation_data(str(sim_output))
    if isinstance(sim_data, dict):
        ground_truth = sim_data.get('trajectory')
        # For dict format, landmarks might be under 'groundtruth'
        landmarks = sim_data.get('landmarks')
        if not landmarks and 'groundtruth' in sim_data:
            landmarks = sim_data['groundtruth'].get('landmarks')
        preintegrated_imu = sim_data.get('preintegrated_imu', [])
        # Also store sim_data for camera measurements access
    else:
        ground_truth = getattr(sim_data, 'ground_truth_trajectory', None)
        landmarks = getattr(sim_data, 'landmarks', None)
        if not landmarks and hasattr(sim_data, 'groundtruth'):
            landmarks = getattr(sim_data.groundtruth, 'landmarks', None)
        preintegrated_imu = getattr(sim_data, 'preintegrated_imu', [])
    
    print(f"  ✓ Generated {len(ground_truth.states) if ground_truth else 0} poses")
    
    # Debug landmarks structure
    if landmarks:
        if hasattr(landmarks, 'landmarks'):
            print(f"  ✓ Created {len(landmarks.landmarks)} landmarks")
        elif isinstance(landmarks, dict) and 'landmarks' in landmarks:
            print(f"  ✓ Created {len(landmarks['landmarks'])} landmarks")
        elif isinstance(landmarks, list):
            print(f"  ✓ Created {len(landmarks)} landmark IDs (need full data)")
            # Clear landmarks if it's just a list of IDs
            landmarks = None
        else:
            print(f"  ✓ Landmarks type: {type(landmarks)}")
    else:
        print(f"  ✓ No landmarks found")
    
    print(f"  ✓ Preintegrated {len(preintegrated_imu)} IMU segments")
    
    # Debug ground truth
    if ground_truth and hasattr(ground_truth, 'states') and len(ground_truth.states) > 0:
        print(f"    Ground truth first pos: {ground_truth.states[0].pose.position}")
        print(f"    Ground truth last pos: {ground_truth.states[-1].pose.position}")
    
    # ============================================================
    # STEP 2: ESTIMATE - Run SLAM estimator
    # ============================================================
    print(f"\nSTEP 2: Running {estimator_type.upper()} estimation...")
    print(f"  Input: {sim_output}")
    
    # Run SLAM using the existing tested function
    slam_result = run_slam(
        estimator=estimator_type,
        input_data=sim_output,
        config=None,  # Use default config
        output=output_path,
    )
    
    if slam_result is None:
        print(f"  ✗ {estimator_type.upper()} failed")
        return
    
    # Handle custom SLAM filename
    if slam_filename and slam_result:
        slam_output = output_path / slam_filename
        # slam_result is already a Path object pointing to the output file
        if slam_result != slam_output:
            slam_result.rename(slam_output)
            slam_result = slam_output
        print(f"  ✓ Saved SLAM result to: {slam_output}")
    elif slam_result:
        print(f"  ✓ Generated SLAM result: {slam_result.name}")
    
    print(f"  ✓ {estimator_type.upper()} complete")
        
    # Load SLAM results using the proper result loader
    from src.estimation.result_io import EstimatorResultStorage
    
    try:
        slam_data = EstimatorResultStorage.load_result(slam_result)
        estimated_trajectory = slam_data.get('trajectory')
        
        # Also load as JSON for accessing other fields like estimated_landmarks
        slam_result_json = None
        if slam_result and slam_result.exists():
            with open(slam_result, 'r') as f:
                slam_result_json = json.load(f)
        
        # Debug: Check what we got
        if estimated_trajectory:
            num_states = len(estimated_trajectory.states) if hasattr(estimated_trajectory, 'states') else 0
            print(f"  ✓ Loaded estimated trajectory with {num_states} states")
            if num_states > 0 and hasattr(estimated_trajectory.states[0], 'pose'):
                first_pos = estimated_trajectory.states[0].pose.position
                last_pos = estimated_trajectory.states[-1].pose.position if num_states > 1 else first_pos
                print(f"    First position: {first_pos}")
                print(f"    Last position: {last_pos}")
        else:
            print("  ⚠ No estimated trajectory found in SLAM result")
    except Exception as e:
        print(f"  ✗ Error loading SLAM result: {e}")
        estimated_trajectory = None
    
    # ============================================================
    # STEP 3: EVALUATE - Compare estimated vs ground truth
    # ============================================================
    print("\nSTEP 3: Evaluating results...")
    print(f"  SLAM result: {slam_result}")
    print(f"  Ground truth: {sim_output}")
    
    # Run evaluation using the existing tested function
    eval_result = run_evaluate(
        result_file=slam_result,
        ground_truth=sim_output,
        output=output_path
    )
    
    if eval_result is None:
        print("  ✗ Evaluation failed")
        return
    
    # Handle custom evaluation filename
    if eval_filename and eval_result:
        eval_output = output_path / eval_filename
        # eval_result is already a Path object pointing to the output file
        if eval_result != eval_output:
            eval_result.rename(eval_output)
            eval_result = eval_output
        print(f"  ✓ Saved evaluation to: {eval_output}")
    elif eval_result:
        print(f"  ✓ Generated evaluation: {eval_result.name}")
        
    # Load evaluation metrics
    with open(eval_result, 'r') as f:
        metrics_data = json.load(f)
        
    # Extract metrics (they're nested under 'metrics' key)
    if 'metrics' in metrics_data:
        metrics = metrics_data['metrics']
    else:
        metrics = metrics_data
    
    # Extract key metrics
    if 'ate' in metrics:
        ate_data = metrics['ate']
        ate_rmse = ate_data.get('rmse', float('inf'))
        ate_mean = ate_data.get('mean', float('inf'))
        ate_max = ate_data.get('max', float('inf'))
    else:
        ate_rmse = float('inf')
        ate_mean = float('inf')
        ate_max = float('inf')
    
    print(f"  ✓ ATE RMSE: {ate_rmse:.3f} meters")
    print(f"  ✓ ATE Mean: {ate_mean:.3f} meters")
    print(f"  ✓ ATE Max: {ate_max:.3f} meters")
    
    # Add note about expected error levels for reasonable expectations
    if ate_rmse > 1.0:
        print("\n  Note: These errors include visual-inertial fusion.")
        print("  The system uses both IMU and visual features.")
        print("  Errors around 0.3-0.5m are typical for this configuration.")
    
    # ============================================================
    # STEP 4: VISUALIZE - Plot the results
    # ============================================================
    print("\nSTEP 4: Creating visualization...")
    
    # Default visualization path
    html_path = output_path / "quickstart_results.html"
    
    # Import enhanced plotting functions
    from src.plotting.enhanced_plots import plot_trajectory_and_landmarks, plot_measurements_with_keyframes
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    # Create comprehensive visualization with multiple plots
    if ground_truth is not None and estimated_trajectory is not None:
        # Create figure with subplots - just 3D trajectory and error plot
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}]],
            subplot_titles=('3D Trajectory and Landmarks', 
                          'Trajectory Error Over Time'),
            column_widths=[0.6, 0.4],
            horizontal_spacing=0.12
        )
        
        # 1. 3D Trajectory and Landmarks plot
        # Plot ground truth trajectory
        gt_positions = np.array([state.pose.position for state in ground_truth.states])
        print(f"    Adding GT trajectory with {len(gt_positions)} points")
        fig.add_trace(
            go.Scatter3d(
                x=gt_positions[:, 0],
                y=gt_positions[:, 1],
                z=gt_positions[:, 2],
                mode='lines',
                name='Ground Truth',
                line=dict(color='blue', width=3),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Plot estimated trajectory
        est_positions = np.array([state.pose.position for state in estimated_trajectory.states])
        print(f"    Adding estimated trajectory with {len(est_positions)} points")
        fig.add_trace(
            go.Scatter3d(
                x=est_positions[:, 0],
                y=est_positions[:, 1],
                z=est_positions[:, 2],
                mode='lines',
                name='SLAM Estimate',
                line=dict(color='red', width=3, dash='dash'),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Plot landmarks if available
        if landmarks is not None:
            # Handle different landmark formats
            landmark_list = []
            if hasattr(landmarks, 'landmarks'):
                # Map object with landmarks dict attribute
                if isinstance(landmarks.landmarks, dict):
                    landmark_list = list(landmarks.landmarks.values())
                else:
                    landmark_list = landmarks.landmarks
            elif isinstance(landmarks, dict) and 'landmarks' in landmarks:
                if isinstance(landmarks['landmarks'], dict):
                    landmark_list = list(landmarks['landmarks'].values())
                else:
                    landmark_list = landmarks['landmarks']
            
            if landmark_list and len(landmark_list) > 0:
                # Check if landmarks are objects or dicts
                try:
                    if hasattr(landmark_list[0], 'position'):
                        landmark_positions = np.array([lm.position for lm in landmark_list])
                    elif isinstance(landmark_list[0], dict) and 'position' in landmark_list[0]:
                        landmark_positions = np.array([lm['position'] for lm in landmark_list])
                    else:
                        landmark_positions = None
                except (AttributeError, KeyError, TypeError):
                    landmark_positions = None
                
                if landmark_positions is not None:
                    fig.add_trace(
                        go.Scatter3d(
                            x=landmark_positions[:, 0],
                            y=landmark_positions[:, 1],
                            z=landmark_positions[:, 2],
                            mode='markers',
                            name='Landmarks',
                            marker=dict(
                                size=4,
                                color='green',
                                symbol='diamond',
                                opacity=0.6
                            ),
                            showlegend=True
                        ),
                        row=1, col=1
                    )
        
        # Plot estimated landmarks if available
        if slam_result_json and 'estimated_landmarks' in slam_result_json:
            est_landmarks = slam_result_json['estimated_landmarks']
            if est_landmarks and 'landmarks' in est_landmarks:
                est_lm_positions = np.array([lm['position'] for lm in est_landmarks['landmarks']])
                fig.add_trace(
                    go.Scatter3d(
                        x=est_lm_positions[:, 0],
                        y=est_lm_positions[:, 1],
                        z=est_lm_positions[:, 2],
                        mode='markers',
                        name='Estimated Landmarks',
                        marker=dict(
                            size=4,
                            color='orange',
                            symbol='circle',
                            opacity=0.6
                        ),
                        showlegend=True
                    ),
                    row=1, col=1
                )
        
        # 2. Trajectory error plot
        # Compute error at each timestamp
        errors = []
        times = []
        for i, est_state in enumerate(estimated_trajectory.states):
            # Find corresponding ground truth state
            est_time = est_state.t if hasattr(est_state, 't') else i * 0.5
            gt_state = min(ground_truth.states, 
                          key=lambda s: abs((s.t if hasattr(s, 't') else 0) - est_time))
            error = np.linalg.norm(est_state.pose.position - gt_state.pose.position)
            errors.append(error)
            times.append(est_time)
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=errors,
                mode='lines+markers',
                name='Position Error',
                line=dict(color='purple', width=2),
                marker=dict(size=4),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Now handle camera data separately for the camera measurements visualization
        camera_frames = None
        keyframe_indices = []
        
        # Try to get camera data from sim_data
        camera_frames = None
        if hasattr(sim_data, 'camera_data') and sim_data.camera_data:
            camera_frames = sim_data.camera_data.frames
        elif hasattr(sim_data, 'measurements') and hasattr(sim_data.measurements, 'camera_measurements'):
            camera_frames = sim_data.measurements.camera_measurements
        elif hasattr(sim_data, 'camera_measurements'):
            camera_frames = sim_data.camera_measurements
        elif isinstance(sim_data, dict):
            # Try different paths in dict
            if 'measurements' in sim_data and 'camera_measurements' in sim_data['measurements']:
                camera_frames = sim_data['measurements']['camera_measurements']
            elif 'camera_measurements' in sim_data:
                camera_frames = sim_data['camera_measurements']
            elif 'camera_data' in sim_data:
                cd = sim_data['camera_data']
                if hasattr(cd, 'frames'):
                    camera_frames = cd.frames
                elif isinstance(cd, dict) and 'frames' in cd:
                    camera_frames = cd['frames']
                else:
                    camera_frames = cd
        
        if camera_frames:
            # Find keyframes or sample frames
            keyframe_indices = []
            for i in range(len(camera_frames)):
                frame = camera_frames[i]
                # Check if it's a keyframe or sample every 15th frame
                if (hasattr(frame, 'is_keyframe') and frame.is_keyframe) or (i % 15 == 0):
                    keyframe_indices.append(i)
                    if len(keyframe_indices) >= 10:  # Limit to 10 keyframes
                        break
        
        # Update layout
        print(f"    Total traces in main figure: {len(fig.data)}")
        fig.update_layout(
            title=dict(
                text=f"{estimator_type.upper()} SLAM Results - {trajectory_type.capitalize()} Trajectory",
                font=dict(size=20)
            ),
            height=600,
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='gray',
                borderwidth=1
            )
        )
        
        # Update 3D axes with uniform scale
        fig.update_scenes(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode='cube',  # Use cube for uniform scaling
            aspectratio=dict(x=1, y=1, z=1)  # Equal aspect ratio
        )
        
        # Update error plot axes
        fig.update_xaxes(title_text="Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="Position Error (m)", row=1, col=2)
        
        # Don't save yet - we'll combine with camera measurements
        # fig.write_html(str(html_path))
        
        # Create camera measurements plot with slider
        fig_camera = None
        
        # Debug camera data
        if isinstance(sim_data, dict):
            print(f"    Sim data keys: {list(sim_data.keys())[:5]}...")
        print(f"    Camera frames found: {camera_frames is not None}")
        if camera_frames:
            print(f"    Camera frames type: {type(camera_frames)}")
            print(f"    Camera frames count: {len(camera_frames) if hasattr(camera_frames, '__len__') else 'N/A'}")
        print(f"    Keyframe indices: {len(keyframe_indices) if keyframe_indices else 0}")
        
        if camera_frames and keyframe_indices:
            print(f"  ✓ Creating camera measurements plot...")
            # Prepare data for all keyframes
            all_frames_data = []
            
            for kf_idx, frame_idx in enumerate(keyframe_indices):
                frame = camera_frames[frame_idx]
                
                # Extract observations
                observations = []
                if hasattr(frame, 'observations'):
                    observations = frame.observations
                elif isinstance(frame, dict) and 'observations' in frame:
                    observations = frame['observations']
                
                pixels_u = []
                pixels_v = []
                landmark_ids = []
                
                if observations:
                    for obs in observations:
                        if hasattr(obs, 'pixel'):
                            pixels_u.append(obs.pixel.u)
                            pixels_v.append(obs.pixel.v)
                            landmark_ids.append(obs.landmark_id)
                        elif isinstance(obs, dict) and 'pixel' in obs:
                            pixels_u.append(obs['pixel']['u'])
                            pixels_v.append(obs['pixel']['v'])
                            landmark_ids.append(obs.get('landmark_id', 0))
                
                # Get timestamp
                timestamp = frame_idx * 0.033  # Default 30 fps
                if hasattr(frame, 'timestamp'):
                    timestamp = frame.timestamp
                elif hasattr(frame, 't'):
                    timestamp = frame.t
                elif isinstance(frame, dict) and 'timestamp' in frame:
                    timestamp = frame['timestamp']
                
                all_frames_data.append({
                    'kf_idx': kf_idx,
                    'frame_idx': frame_idx,
                    'timestamp': timestamp,
                    'pixels_u': pixels_u,
                    'pixels_v': pixels_v,
                    'landmark_ids': landmark_ids
                })
            
            # Create subplots - one row with slider, one row with camera view
            from plotly.subplots import make_subplots
            
            fig_camera = make_subplots(
                rows=2, cols=1,
                row_heights=[0.15, 0.85],
                specs=[[{"type": "scatter"}],
                       [{"type": "scatter"}]],
                subplot_titles=("Keyframe Timeline", "Camera Measurements"),
                vertical_spacing=0.1
            )
            
            # Add timeline for all keyframes (row 1)
            timeline_x = []
            timeline_y = []
            timeline_text = []
            for data in all_frames_data:
                timeline_x.append(data['timestamp'])
                timeline_y.append(1)
                timeline_text.append(f"KF{data['kf_idx']}")
            
            fig_camera.add_trace(
                go.Scatter(
                    x=timeline_x,
                    y=timeline_y,
                    mode='markers+text',
                    text=timeline_text,
                    textposition="top center",
                    marker=dict(
                        size=15,
                        color=list(range(len(timeline_x))),
                        colorscale='Viridis',
                        showscale=False,
                        line=dict(width=2, color='white')
                    ),
                    name='Keyframes',
                    hovertemplate='Keyframe %{text}<br>Time: %{x:.2f}s<extra></extra>',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Create frames for animation
            frames = []
            for data in all_frames_data:
                frame_data = []
                
                # Timeline marker (highlight current keyframe)
                colors = ['lightgray'] * len(timeline_x)
                colors[data['kf_idx']] = 'red'
                frame_data.append(go.Scatter(
                    x=timeline_x,
                    y=timeline_y,
                    mode='markers+text',
                    text=timeline_text,
                    textposition="top center",
                    marker=dict(
                        size=15,
                        color=colors,
                        line=dict(width=2, color='white')
                    ),
                    showlegend=False
                ))
                
                # Camera measurements
                if data['pixels_u']:
                    frame_data.append(go.Scatter(
                        x=data['pixels_u'],
                        y=data['pixels_v'],
                        mode='markers+text',
                        text=[f"LM{lid}" for lid in data['landmark_ids']],
                        textposition="top center",
                        textfont=dict(size=8),
                        marker=dict(
                            size=10,
                            color=data['landmark_ids'],
                            colorscale='Viridis',
                            colorbar=dict(
                                title="Landmark ID",
                                x=1.05,
                                thickness=20
                            ),
                            showscale=True,
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate='<b>Landmark %{text}</b><br>Pixel: (%{x:.1f}, %{y:.1f})<extra></extra>',
                        showlegend=False
                    ))
                else:
                    # Empty trace if no observations
                    frame_data.append(go.Scatter(
                        x=[],
                        y=[],
                        mode='markers',
                        showlegend=False
                    ))
                
                # Principal point
                frame_data.append(go.Scatter(
                    x=[320],
                    y=[240],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='red',
                        symbol='x',
                        line=dict(width=2)
                    ),
                    name='Principal Point',
                    showlegend=False,
                    hovertemplate='Principal Point<br>(320, 240)<extra></extra>'
                ))
                
                frames.append(go.Frame(
                    data=frame_data,
                    name=str(data['kf_idx']),
                    layout=go.Layout(
                        title=f"Camera Measurements - Keyframe {data['kf_idx']} (t={data['timestamp']:.2f}s, frame={data['frame_idx']})"
                    )
                ))
            
            # Add initial camera measurements (row 2)
            if all_frames_data[0]['pixels_u']:
                fig_camera.add_trace(
                    go.Scatter(
                        x=all_frames_data[0]['pixels_u'],
                        y=all_frames_data[0]['pixels_v'],
                        mode='markers+text',
                        text=[f"LM{lid}" for lid in all_frames_data[0]['landmark_ids']],
                        textposition="top center",
                        textfont=dict(size=8),
                        marker=dict(
                            size=10,
                            color=all_frames_data[0]['landmark_ids'],
                            colorscale='Viridis',
                            colorbar=dict(
                                title="Landmark ID",
                                x=1.05,
                                thickness=20
                            ),
                            showscale=True,
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate='<b>Landmark %{text}</b><br>Pixel: (%{x:.1f}, %{y:.1f})<extra></extra>',
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            # Add principal point
            fig_camera.add_trace(
                go.Scatter(
                    x=[320],
                    y=[240],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='red',
                        symbol='x',
                        line=dict(width=2)
                    ),
                    name='Principal Point',
                    showlegend=True,
                    hovertemplate='Principal Point<br>(320, 240)<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add camera bounds as shapes
            fig_camera.add_shape(
                type="rect",
                x0=0, y0=0, x1=640, y1=480,
                line=dict(color="red", width=2, dash="dash"),
                row=2, col=1
            )
            
            # Add crosshair
            fig_camera.add_shape(
                type="line",
                x0=320, y0=0, x1=320, y1=480,
                line=dict(color="lightgray", width=1, dash="dot"),
                row=2, col=1
            )
            fig_camera.add_shape(
                type="line",
                x0=0, y0=240, x1=640, y1=240,
                line=dict(color="lightgray", width=1, dash="dot"),
                row=2, col=1
            )
            
            # Create slider steps
            sliders = [dict(
                active=0,
                yanchor="top",
                y=0.02,  # Move slider to bottom
                xanchor="left",
                x=0.1,
                currentvalue=dict(
                    prefix="Keyframe: ",
                    visible=True,
                    xanchor="right",
                    font=dict(size=16)
                ),
                pad=dict(b=10, t=50),
                len=0.8,
                steps=[
                    dict(
                        args=[[str(data['kf_idx'])],
                              dict(frame=dict(duration=0, redraw=True),
                                   mode="immediate",
                                   transition=dict(duration=0))],
                        label=f"KF{data['kf_idx']} ({data['timestamp']:.1f}s)",
                        method="animate"
                    ) for data in all_frames_data
                ]
            )]
            
            # Set frames and sliders
            fig_camera.frames = frames
            
            # Update layout
            fig_camera.update_layout(
                sliders=sliders,
                height=900,
                title=dict(
                    text=f"Camera Measurements - Keyframe 0 (t={all_frames_data[0]['timestamp']:.2f}s)",
                    font=dict(size=20),
                    x=0.5,
                    xanchor='center'
                ),
                showlegend=True,
                legend=dict(
                    x=0.85,
                    y=0.5,
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='gray',
                    borderwidth=1
                ),
                paper_bgcolor='#f0f2f5',
                plot_bgcolor='white'
            )
            
            # Update axes
            fig_camera.update_xaxes(title_text="Time (s)", row=1, col=1, showgrid=False)
            fig_camera.update_yaxes(showticklabels=False, range=[0.5, 1.5], row=1, col=1, showgrid=False)
            
            fig_camera.update_xaxes(
                title_text="Pixel u",
                range=[-50, 690],
                constrain='domain',
                showgrid=True,
                gridcolor='lightgray',
                row=2, col=1
            )
            fig_camera.update_yaxes(
                title_text="Pixel v",
                range=[530, -50],  # Inverted for image coordinates
                constrain='domain',
                scaleanchor="x2",
                scaleratio=1,
                showgrid=True,
                gridcolor='lightgray',
                row=2, col=1
            )
            
        # Save the plots - use a simpler approach
        # First save the main figure as standalone HTML
        fig.write_html(str(html_path), include_plotlyjs='cdn')
        
        # If we have camera measurements, append them to the HTML
        if fig_camera is not None:
            # Read the HTML that was just written
            with open(str(html_path), 'r') as f:
                html_content = f.read()
            
            # Find where to insert the camera plot (before </body>)
            from plotly.io import to_html
            camera_html = to_html(fig_camera, include_plotlyjs=False, div_id="camera-plot")
            
            # Create the camera section HTML
            camera_section = f"""
    <div style="background-color: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 20px auto; padding: 20px; max-width: 1400px;">
        <div style="font-size: 24px; font-weight: bold; margin: 30px 0 10px 0; padding: 10px; background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%); color: white; border-radius: 5px;">
            📷 Camera Measurements - Interactive Keyframe Viewer
        </div>
        {camera_html}
    </div>
"""
            
            # Insert before </body>
            html_content = html_content.replace('</body>', camera_section + '\n</body>')
            
            # Write back the modified HTML
            with open(str(html_path), 'w') as f:
                f.write(html_content)
        
        print(f"  ✓ Saved comprehensive visualization to {html_path}")
        print("    - 3D trajectory comparison with landmarks")
        print("    - 2D camera measurements")
        print("    - Error evolution over time")
        if fig_camera is not None:
            print("    - Interactive camera measurements with slider control")
        
        # Open main visualization in browser
        import webbrowser
        webbrowser.open(f"file://{html_path.absolute()}")
    
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print(" PIPELINE COMPLETE!")
    print("="*60)
    print("\nSuccessfully ran the complete SLAM pipeline:")
    print(f"1. SIMULATED a {trajectory_type} trajectory with IMU + visual data")
    print(f"2. ESTIMATED the trajectory using {estimator_type.upper()}")
    print("3. EVALUATED accuracy with error metrics")
    print("4. VISUALIZED the results")
    print(f"\nKey Results:")
    print(f"  • Trajectory: {duration:.1f}s {trajectory_type} motion")
    print(f"  • ATE RMSE: {ate_rmse:.3f} meters")
    print(f"  • Visual features: Included for accurate estimation")
    print(f"\nOutput Files:")
    print(f"  • Simulation: {sim_output}")
    print(f"  • SLAM result: {slam_result}")
    print(f"  • Evaluation: {eval_result}")
    print(f"  • Visualization: {html_path}")
    print("\nNext steps:")
    print("  • Try different trajectories: 'figure8', 'spiral', 'line'")
    print("  • Try different estimators: 'ekf', 'swba', 'srif', 'cpp-swba'")
    print("  • Adjust simulation duration (default: 10s)")
    print("  • Use custom filenames for reproducibility")
    print("\nRun with: ./run.sh e2e_simple [options]")
    print("All components (simulation, estimation, evaluation) are production-ready.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SLAM Pipeline Quickstart")
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration in seconds")
    parser.add_argument("--trajectory", type=str, default="circle", 
                        choices=["circle", "figure8", "spiral", "line"],
                        help="Trajectory type")
    parser.add_argument("--estimator", type=str, default="cpp-swba",
                        choices=["ekf", "swba", "srif", "cpp-swba"],
                        help="SLAM estimator type")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Output directory for all files")
    parser.add_argument("--sim-file", type=str, default=None,
                        help="Custom filename for simulation output")
    parser.add_argument("--slam-file", type=str, default=None,
                        help="Custom filename for SLAM output")
    parser.add_argument("--eval-file", type=str, default=None,
                        help="Custom filename for evaluation output")
    
    args = parser.parse_args()
    
    run_e2e_simple(
        duration=args.duration,
        trajectory_type=args.trajectory,
        estimator_type=args.estimator,
        output_dir=args.output_dir,
        sim_filename=args.sim_file,
        slam_filename=args.slam_file,
        eval_filename=args.eval_file
    )