"""
Enhanced plotting functionality for SLAM visualization.
Merged from src/visualization/enhanced_plots.py
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Any
import json

from src.common.json_io import SimulationData
from src.common.data_structures import Trajectory, Map


def create_full_visualization(
    primary_data: SimulationData,
    compare_data: Optional[SimulationData] = None,
    show_trajectory: bool = True,
    show_measurements: bool = True,
    show_imu: bool = True,
    max_keyframes: Optional[int] = None
) -> str:
    """
    Create a complete HTML visualization with all plots.
    
    Args:
        primary_data: Primary simulation or SLAM result data
        compare_data: Optional comparison data
        show_trajectory: Include 3D trajectory plot
        show_measurements: Include 2D measurement visualization
        show_imu: Include IMU data plots
        max_keyframes: Maximum number of keyframes to show (None = all)
    
    Returns:
        HTML string with embedded interactive plots
    """
    figures = []
    
    # 1. 3D Trajectory and Landmarks
    if show_trajectory:
        fig_traj = plot_trajectory_and_landmarks(
            primary_data,
            compare_data,
            title="3D Trajectory and Landmarks"
        )
        figures.append(("trajectory", fig_traj))
    
    # 2. 2D Measurements with Keyframe Selection
    cam_meas = getattr(primary_data, 'camera_measurements', None) or getattr(primary_data, 'camera_data', None)
    if show_measurements and cam_meas:
        fig_meas = plot_measurements_with_keyframes(
            primary_data,
            max_keyframes=max_keyframes
        )
        figures.append(("measurements", fig_meas))
    
    # 3. IMU Data
    imu_meas = getattr(primary_data, 'imu_measurements', None) or getattr(primary_data, 'imu_data', None)
    if show_imu and imu_meas:
        fig_imu = plot_imu_data_enhanced(primary_data, compare_data)
        figures.append(("imu", fig_imu))
    
    # Create HTML with all figures
    html = create_html_dashboard(figures, primary_data, compare_data)
    return html


def plot_trajectory_and_landmarks(
    primary_data: SimulationData,
    compare_data: Optional[SimulationData] = None,
    title: str = "3D Trajectory and Landmarks"
) -> go.Figure:
    """
    Create 3D plot with trajectory and landmarks.
    
    Args:
        primary_data: Primary data to plot
        compare_data: Optional comparison data
        title: Plot title
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Plot primary trajectory - check multiple possible attribute names
    traj = None
    if hasattr(primary_data, 'ground_truth_trajectory'):
        traj = primary_data.ground_truth_trajectory
    elif hasattr(primary_data, 'trajectory'):
        traj = primary_data.trajectory
    elif hasattr(primary_data, 'groundtruth'):
        # For raw SimulationData with groundtruth field
        if hasattr(primary_data.groundtruth, 'trajectory'):
            traj = primary_data.groundtruth.trajectory
    
    if traj and hasattr(traj, 'states'):
        positions = np.array([state.pose.position for state in traj.states])
        
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0].tolist(),
            y=positions[:, 1].tolist(),
            z=positions[:, 2].tolist(),
            mode='lines+markers',
            name='Ground Truth' if compare_data else 'Trajectory',
            line=dict(color='blue', width=3),
            marker=dict(size=2)
        ))
    
    # Plot comparison trajectory if provided
    if compare_data and hasattr(compare_data, 'ground_truth_trajectory') and compare_data.ground_truth_trajectory:
        traj = compare_data.ground_truth_trajectory
        positions = np.array([state.pose.position for state in traj.states])
        
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0].tolist(),
            y=positions[:, 1].tolist(),
            z=positions[:, 2].tolist(),
            mode='lines+markers',
            name='SLAM Estimate',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=2)
        ))
    
    # Plot landmarks
    if hasattr(primary_data, 'landmarks') and primary_data.landmarks:
        # Handle both dict and list formats
        if hasattr(primary_data.landmarks, 'landmarks'):
            if isinstance(primary_data.landmarks.landmarks, dict):
                landmark_positions = np.array([lm.position for lm in primary_data.landmarks.landmarks.values()])
            else:
                landmark_positions = np.array([lm.position for lm in primary_data.landmarks.landmarks])
        else:
            landmark_positions = np.array([])
        
        if len(landmark_positions) > 0:
            fig.add_trace(go.Scatter3d(
                x=landmark_positions[:, 0].tolist(),
                y=landmark_positions[:, 1].tolist(),
                z=landmark_positions[:, 2].tolist(),
                mode='markers',
                name='Landmarks',
                marker=dict(
                    size=4,
                    color='green',
                    symbol='diamond',
                    opacity=0.7
                )
            ))
    
    # Add camera frustums at keyframes
    if hasattr(primary_data, 'camera_measurements') and primary_data.camera_measurements:
        keyframe_indices = np.linspace(
            0, 
            len(primary_data.camera_measurements) - 1,
            min(20, len(primary_data.camera_measurements))
        ).astype(int)
        
        for idx in keyframe_indices:
            frame = primary_data.camera_measurements[idx]
            if hasattr(primary_data, 'ground_truth_trajectory') and primary_data.ground_truth_trajectory:
                # Find corresponding pose
                pose_idx = min(
                    range(len(primary_data.ground_truth_trajectory.states)),
                    key=lambda i: abs(
                        primary_data.ground_truth_trajectory.states[i].pose.timestamp - frame.timestamp
                    )
                )
                pose = primary_data.ground_truth_trajectory.states[pose_idx].pose
                
                # Draw camera frustum
                frustum = create_camera_frustum(pose.position, pose.rotation_matrix, scale=0.2)
                fig.add_trace(go.Mesh3d(
                    x=frustum['x'].tolist() if isinstance(frustum['x'], np.ndarray) else frustum['x'],
                    y=frustum['y'].tolist() if isinstance(frustum['y'], np.ndarray) else frustum['y'],
                    z=frustum['z'].tolist() if isinstance(frustum['z'], np.ndarray) else frustum['z'],
                    i=frustum['i'].tolist() if isinstance(frustum['i'], np.ndarray) else frustum['i'],
                    j=frustum['j'].tolist() if isinstance(frustum['j'], np.ndarray) else frustum['j'],
                    k=frustum['k'].tolist() if isinstance(frustum['k'], np.ndarray) else frustum['k'],
                    color='orange',
                    opacity=0.3,
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True,
        height=700
    )
    
    return fig


def plot_measurements_with_keyframes(
    data: SimulationData,
    max_keyframes: Optional[int] = None
) -> go.Figure:
    """
    Create 2D measurement visualization with keyframe dropdown.
    
    Args:
        data: Simulation data with camera measurements
        max_keyframes: Maximum number of keyframes to include
    
    Returns:
        Plotly figure with dropdown
    """
    if not hasattr(data, 'camera_measurements') or not data.camera_measurements:
        return go.Figure().add_annotation(text="No camera measurements available")
    
    # Get camera calibration
    camera_calib = None
    if hasattr(data, 'camera_calibrations') and data.camera_calibrations:
        camera_calib = list(data.camera_calibrations.values())[0]
    
    # Select keyframes
    n_frames = len(data.camera_measurements)
    if max_keyframes and n_frames > max_keyframes:
        keyframe_indices = np.linspace(0, n_frames - 1, max_keyframes).astype(int)
    else:
        keyframe_indices = range(n_frames)
    
    # Create figure
    fig = go.Figure()
    
    # Prepare data for each keyframe
    buttons = []
    for i, idx in enumerate(keyframe_indices):
        frame = data.camera_measurements[idx]
        
        # Extract measurements
        if frame.observations:
            pixels = np.array([[obs.pixel.u, obs.pixel.v] for obs in frame.observations])
            landmark_ids = [obs.landmark_id for obs in frame.observations]
        else:
            pixels = np.array([]).reshape(0, 2)
            landmark_ids = []
        
        # Add trace for measurements
        visible = i == 0  # Only first frame visible initially
        fig.add_trace(go.Scatter(
            x=pixels[:, 0].tolist() if len(pixels) > 0 else [],
            y=pixels[:, 1].tolist() if len(pixels) > 0 else [],
            mode='markers',
            name=f'Frame {idx}',
            marker=dict(
                size=8,
                color=landmark_ids if landmark_ids else [0],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Landmark ID")
            ),
            text=[f"LM {lid}" for lid in landmark_ids],
            hovertemplate='<b>Landmark %{text}</b><br>u: %{x:.1f}<br>v: %{y:.1f}',
            visible=visible
        ))
        
        # Add projected landmarks if available
        if hasattr(data, 'landmarks') and data.landmarks and camera_calib:
            # Project landmarks to this frame
            projected = project_landmarks_to_frame(
                data.landmarks,
                frame,
                data.ground_truth_trajectory,
                camera_calib
            )
            
            fig.add_trace(go.Scatter(
                x=projected[:, 0].tolist() if len(projected) > 0 else [],
                y=projected[:, 1].tolist() if len(projected) > 0 else [],
                mode='markers',
                name=f'Projected (Frame {idx})',
                marker=dict(
                    size=6,
                    symbol='x',
                    color='red',
                    opacity=0.5
                ),
                visible=visible
            ))
        
        # Create button for this keyframe
        button_visible = [False] * len(fig.data)
        # Calculate the correct indices for this keyframe's traces
        traces_per_frame = 2 if (hasattr(data, 'landmarks') and data.landmarks and camera_calib) else 1
        start_idx = i * traces_per_frame
        button_visible[start_idx] = True  # Measurements
        if traces_per_frame == 2:
            button_visible[start_idx + 1] = True  # Projections
        
        buttons.append(dict(
            label=f'Frame {idx} (t={frame.timestamp:.2f}s)',
            method='update',
            args=[{'visible': button_visible},
                  {'title': f'Camera Measurements - Frame {idx} (t={frame.timestamp:.2f}s)'}]
        ))
    
    # Add dropdown menu
    fig.update_layout(
        updatemenus=[
            dict(
                type='dropdown',
                active=0,
                buttons=buttons,
                x=0.1,
                y=1.15,
                xanchor='left',
                yanchor='top'
            )
        ],
        title='Camera Measurements - Frame 0',
        xaxis_title='u (pixels)',
        yaxis_title='v (pixels)',
        yaxis=dict(autorange='reversed'),  # Image coordinates
        height=600,
        showlegend=True
    )
    
    # Set image bounds if calibration available
    if camera_calib:
        fig.update_xaxes(range=[0, camera_calib.intrinsics.width])
        fig.update_yaxes(range=[camera_calib.intrinsics.height, 0])
    
    return fig


def plot_imu_data_enhanced(
    primary_data: SimulationData,
    compare_data: Optional[SimulationData] = None
) -> go.Figure:
    """
    Create IMU data plots (accelerometer and gyroscope).
    Named with _enhanced to avoid conflict with existing plot_imu_data.
    
    Args:
        primary_data: Primary data with IMU measurements
        compare_data: Optional comparison data
    
    Returns:
        Plotly figure with subplots
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accelerometer X/Y/Z', 'Gyroscope X/Y/Z',
                       'Acceleration Magnitude', 'Angular Velocity Magnitude'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Extract IMU data
    if hasattr(primary_data, 'imu_measurements') and primary_data.imu_measurements:
        timestamps = np.array([m.timestamp for m in primary_data.imu_measurements])
        accel = np.array([m.accelerometer for m in primary_data.imu_measurements])
        gyro = np.array([m.gyroscope for m in primary_data.imu_measurements])
        
        # Plot accelerometer
        for i, axis in enumerate(['X', 'Y', 'Z']):
            fig.add_trace(
                go.Scatter(
                    x=timestamps.tolist(),
                    y=accel[:, i].tolist(),
                    mode='lines',
                    name=f'Accel {axis}',
                    line=dict(width=1),
                    legendgroup='accel'
                ),
                row=1, col=1
            )
        
        # Plot gyroscope
        for i, axis in enumerate(['X', 'Y', 'Z']):
            fig.add_trace(
                go.Scatter(
                    x=timestamps.tolist(),
                    y=gyro[:, i].tolist(),
                    mode='lines',
                    name=f'Gyro {axis}',
                    line=dict(width=1),
                    legendgroup='gyro'
                ),
                row=1, col=2
            )
        
        # Plot magnitudes
        accel_mag = np.linalg.norm(accel, axis=1)
        gyro_mag = np.linalg.norm(gyro, axis=1)
        
        fig.add_trace(
            go.Scatter(
                x=timestamps.tolist(),
                y=accel_mag.tolist(),
                mode='lines',
                name='|Accel|',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=timestamps.tolist(),
                y=gyro_mag.tolist(),
                mode='lines',
                name='|Gyro|',
                line=dict(color='orange', width=2)
            ),
            row=2, col=2
        )
    
    # Add comparison data if provided
    if compare_data and hasattr(compare_data, 'imu_measurements') and compare_data.imu_measurements:
        timestamps_comp = np.array([m.timestamp for m in compare_data.imu_measurements])
        accel_comp = np.array([m.accelerometer for m in compare_data.imu_measurements])
        gyro_comp = np.array([m.gyroscope for m in compare_data.imu_measurements])
        
        # Add comparison traces with dashed lines
        for i, axis in enumerate(['X', 'Y', 'Z']):
            fig.add_trace(
                go.Scatter(
                    x=timestamps_comp.tolist(),
                    y=accel_comp[:, i].tolist(),
                    mode='lines',
                    name=f'Accel {axis} (Est)',
                    line=dict(width=1, dash='dash'),
                    legendgroup='accel_est',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps_comp.tolist(),
                    y=gyro_comp[:, i].tolist(),
                    mode='lines',
                    name=f'Gyro {axis} (Est)',
                    line=dict(width=1, dash='dash'),
                    legendgroup='gyro_est',
                    showlegend=False
                ),
                row=1, col=2
            )
    
    # Update layout
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=1, col=1)
    fig.update_yaxes(title_text="Angular Velocity (rad/s)", row=1, col=2)
    fig.update_yaxes(title_text="Magnitude (m/s²)", row=2, col=1)
    fig.update_yaxes(title_text="Magnitude (rad/s)", row=2, col=2)
    
    fig.update_layout(
        title="IMU Measurements",
        height=800,
        showlegend=True
    )
    
    return fig


def create_camera_frustum(
    position: np.ndarray,
    rotation_matrix: np.ndarray,
    scale: float = 1.0,
    aspect: float = 1.33,
    fov: float = 60.0
) -> Dict[str, np.ndarray]:
    """
    Create camera frustum mesh for visualization.
    
    Args:
        position: Camera position
        rotation_matrix: Camera orientation as SO3 matrix
        scale: Frustum scale
        aspect: Aspect ratio
        fov: Field of view in degrees
    
    Returns:
        Dictionary with mesh vertices and faces
    """
    # Create frustum in camera frame
    fov_rad = np.radians(fov)
    h = scale * np.tan(fov_rad / 2)
    w = h * aspect
    
    # Frustum vertices in camera frame
    vertices_cam = np.array([
        [0, 0, 0],           # Camera center
        [-w, -h, scale],     # Bottom-left
        [w, -h, scale],      # Bottom-right
        [w, h, scale],       # Top-right
        [-w, h, scale]       # Top-left
    ])
    
    # Transform to world frame
    R = rotation_matrix
    vertices_world = (R @ vertices_cam.T).T + position
    
    # Define triangular faces
    faces = np.array([
        [0, 1, 2],  # Bottom
        [0, 2, 3],  # Right
        [0, 3, 4],  # Top
        [0, 4, 1],  # Left
        [1, 2, 3],  # Front face part 1
        [1, 3, 4]   # Front face part 2
    ])
    
    return {
        'x': vertices_world[:, 0],
        'y': vertices_world[:, 1],
        'z': vertices_world[:, 2],
        'i': faces[:, 0],
        'j': faces[:, 1],
        'k': faces[:, 2]
    }


def project_landmarks_to_frame(
    landmarks: Map,
    frame: Any,
    trajectory: Trajectory,
    calibration: Any
) -> np.ndarray:
    """
    Project landmarks to camera frame.
    
    Args:
        landmarks: Map with landmarks
        frame: Camera frame
        trajectory: Trajectory for pose lookup
        calibration: Camera calibration
    
    Returns:
        Array of projected pixels
    """
    from src.estimation.camera_model import CameraMeasurementModel
    
    # Find pose at frame timestamp
    pose_idx = min(
        range(len(trajectory.states)),
        key=lambda i: abs(trajectory.states[i].pose.timestamp - frame.timestamp)
    )
    pose = trajectory.states[pose_idx].pose
    
    # Create camera model
    camera_model = CameraMeasurementModel(calibration)
    
    # Project landmarks
    projected = []
    # Handle both dict and list landmarks
    if isinstance(landmarks.landmarks, dict):
        landmark_list = landmarks.landmarks.values()
    else:
        landmark_list = landmarks.landmarks
    
    for landmark in landmark_list:
        pixel, _, _ = camera_model.project(landmark.position, pose, False)
        if pixel is not None:
            projected.append([pixel.u, pixel.v])
    
    return np.array(projected) if projected else np.array([]).reshape(0, 2)


def create_html_dashboard(
    figures: List[tuple],
    primary_data: SimulationData,
    compare_data: Optional[SimulationData] = None
) -> str:
    """
    Create HTML dashboard with all plots.
    
    Args:
        figures: List of (name, figure) tuples
        primary_data: Primary data
        compare_data: Optional comparison data
    
    Returns:
        Complete HTML string
    """
    # Convert figures to HTML
    plot_htmls = []
    for name, fig in figures:
        # Generate the plot HTML - just use the standard method
        # Use config to ensure the data is included properly
        plot_html = fig.to_html(
            full_html=False,
            include_plotlyjs=False, 
            div_id=f"plot_{name}",
            config={'displayModeBar': True}
        )
        plot_htmls.append(plot_html)
    
    # Create summary statistics
    stats = create_summary_statistics(primary_data, compare_data)
    
    # Build complete HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SLAM Visualization Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                margin: -20px -20px 20px -20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 30px;
            }}
            .stat-card {{
                background: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .stat-label {{
                color: #7f8c8d;
                font-size: 12px;
                text-transform: uppercase;
                margin-bottom: 5px;
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }}
            .plot-container {{
                background: white;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .plot-title {{
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 15px;
                color: #2c3e50;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>SLAM Visualization Dashboard</h1>
            <p>Generated from: {getattr(primary_data, 'metadata', {}).get('source', 'simulation')}</p>
            {f'<p>Comparing with: {getattr(compare_data, "metadata", {}).get("source", "comparison")}</p>' if compare_data else ''}
        </div>
        
        <div class="container">
            <div class="stats-grid">
                {stats}
            </div>
            
            {''.join([f'<div class="plot-container">{html}</div>' for html in plot_htmls])}
        </div>
    </body>
    </html>
    """
    
    return html


def create_summary_statistics(
    primary_data: SimulationData,
    compare_data: Optional[SimulationData] = None
) -> str:
    """Create summary statistics HTML cards."""
    stats = []
    
    # Trajectory length
    if hasattr(primary_data, 'ground_truth_trajectory') and primary_data.ground_truth_trajectory:
        traj_length = compute_trajectory_length(primary_data.ground_truth_trajectory)
        stats.append(f"""
            <div class="stat-card">
                <div class="stat-label">Trajectory Length</div>
                <div class="stat-value">{traj_length:.1f} m</div>
            </div>
        """)
    
    # Number of landmarks
    if hasattr(primary_data, 'landmarks') and primary_data.landmarks:
        if hasattr(primary_data.landmarks, 'landmarks'):
            if isinstance(primary_data.landmarks.landmarks, dict):
                n_landmarks = len(primary_data.landmarks.landmarks)
            else:
                n_landmarks = len(primary_data.landmarks.landmarks)
        else:
            n_landmarks = 0
        stats.append(f"""
            <div class="stat-card">
                <div class="stat-label">Landmarks</div>
                <div class="stat-value">{n_landmarks}</div>
            </div>
        """)
    
    # Number of measurements
    if hasattr(primary_data, 'camera_measurements') and primary_data.camera_measurements:
        n_frames = len(primary_data.camera_measurements)
        total_obs = sum(len(f.observations) for f in primary_data.camera_measurements)
        stats.append(f"""
            <div class="stat-card">
                <div class="stat-label">Camera Frames</div>
                <div class="stat-value">{n_frames}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Observations</div>
                <div class="stat-value">{total_obs}</div>
            </div>
        """)
    
    if hasattr(primary_data, 'imu_measurements') and primary_data.imu_measurements:
        n_imu = len(primary_data.imu_measurements)
        stats.append(f"""
            <div class="stat-card">
                <div class="stat-label">IMU Samples</div>
                <div class="stat-value">{n_imu}</div>
            </div>
        """)
    
    # Duration
    duration = 0
    if hasattr(primary_data, 'ground_truth_trajectory') and primary_data.ground_truth_trajectory and len(primary_data.ground_truth_trajectory.states) > 1:
        duration = (primary_data.ground_truth_trajectory.states[-1].pose.timestamp - 
                   primary_data.ground_truth_trajectory.states[0].pose.timestamp)
        stats.append(f"""
            <div class="stat-card">
                <div class="stat-label">Duration</div>
                <div class="stat-value">{duration:.1f} s</div>
            </div>
        """)
    
    return ''.join(stats)


def compute_trajectory_length(trajectory: Trajectory) -> float:
    """Compute total trajectory length."""
    length = 0.0
    for i in range(1, len(trajectory.states)):
        prev_pos = trajectory.states[i-1].pose.position
        curr_pos = trajectory.states[i].pose.position
        length += np.linalg.norm(curr_pos - prev_pos)
    return length