"""
Sensor data visualization tools.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Any
from pathlib import Path

from src.common.data_structures import (
    IMUData, CameraData, Map, CameraFrame,
    CameraIntrinsics, ImagePoint
)


def plot_camera_tracks(
    camera_data: CameraData,
    intrinsics: CameraIntrinsics,
    max_frames: Optional[int] = None,
    title: str = "Camera Feature Tracks",
    show_grid: bool = True
) -> go.Figure:
    """
    Visualize 2D feature tracks in image plane.
    
    Args:
        camera_data: Camera observations
        intrinsics: Camera intrinsics for image bounds
        max_frames: Maximum number of frames to plot (None for all)
        title: Plot title
        show_grid: Whether to show grid
    
    Returns:
        Plotly figure
    """
    # Group observations by landmark ID
    tracks: Dict[int, List[tuple]] = {}
    
    frames_to_plot = camera_data.frames[:max_frames] if max_frames else camera_data.frames
    
    for frame_idx, frame in enumerate(frames_to_plot):
        for obs in frame.observations:
            if obs.landmark_id not in tracks:
                tracks[obs.landmark_id] = []
            tracks[obs.landmark_id].append((
                frame.timestamp,
                obs.pixel.u if hasattr(obs.pixel, 'u') else obs.pixel[0],
                obs.pixel.v if hasattr(obs.pixel, 'v') else obs.pixel[1],
                frame_idx
            ))
    
    # Create figure
    fig = go.Figure()
    
    # Add rectangle for image bounds
    fig.add_shape(
        type="rect",
        x0=0, y0=0,
        x1=intrinsics.width, y1=intrinsics.height,
        line=dict(color="black", width=2),
        fillcolor="lightgray",
        opacity=0.1
    )
    
    # Plot tracks
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    
    for idx, (landmark_id, track) in enumerate(tracks.items()):
        if len(track) < 2:
            continue  # Skip single observations
        
        color = colors[idx % len(colors)]
        
        # Extract track coordinates
        timestamps = [t[0] for t in track]
        u_coords = [t[1] for t in track]
        v_coords = [t[2] for t in track]
        
        # Add track line
        fig.add_trace(go.Scatter(
            x=u_coords,
            y=v_coords,
            mode='lines+markers',
            name=f'Landmark {landmark_id}',
            line=dict(color=color, width=1),
            marker=dict(size=4, color=color),
            hovertemplate=(
                'Landmark: %{text}<br>' +
                'u: %{x:.1f}<br>' +
                'v: %{y:.1f}<br>' +
                '<extra></extra>'
            ),
            text=[f'ID:{landmark_id}, t:{t:.2f}s' for t in timestamps],
            showlegend=(idx < 10)  # Only show first 10 in legend
        ))
        
        # Mark first observation
        fig.add_trace(go.Scatter(
            x=[u_coords[0]],
            y=[v_coords[0]],
            mode='markers',
            marker=dict(size=8, color=color, symbol='circle-open'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(
            title='u (pixels)',
            range=[0, intrinsics.width],
            showgrid=show_grid,
            constrain='domain'
        ),
        yaxis=dict(
            title='v (pixels)',
            range=[intrinsics.height, 0],  # Invert Y axis (image convention)
            showgrid=show_grid,
            scaleanchor='x',
            scaleratio=1
        ),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        width=800,
        height=600
    )
    
    return fig


def plot_imu_data(
    imu_data: IMUData,
    title: str = "IMU Measurements",
    show_bias: bool = False
) -> go.Figure:
    """
    Create time series plots for IMU data.
    
    Args:
        imu_data: IMU measurements
        title: Plot title
        show_bias: Whether to show bias estimates (placeholder)
    
    Returns:
        Plotly figure with subplots
    """
    # Extract data
    timestamps = [m.timestamp for m in imu_data.measurements]
    
    # Handle empty data
    if len(imu_data.measurements) > 0:
        accels = np.array([m.accelerometer for m in imu_data.measurements])
        gyros = np.array([m.gyroscope for m in imu_data.measurements])
    else:
        # Create dummy data for empty case
        timestamps = [0.0]
        accels = np.array([[0.0, 0.0, 0.0]])
        gyros = np.array([[0.0, 0.0, 0.0]])
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Accelerometer', 'Gyroscope'),
        vertical_spacing=0.12
    )
    
    # Accelerometer subplot
    fig.add_trace(go.Scatter(
        x=timestamps, y=accels[:, 0],
        name='acc_x',
        line=dict(color='red', width=1),
        legendgroup='acc'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=timestamps, y=accels[:, 1],
        name='acc_y',
        line=dict(color='green', width=1),
        legendgroup='acc'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=timestamps, y=accels[:, 2],
        name='acc_z',
        line=dict(color='blue', width=1),
        legendgroup='acc'
    ), row=1, col=1)
    
    # Add gravity magnitude line (approximate)
    gravity_mag = 9.81
    fig.add_trace(go.Scatter(
        x=[timestamps[0], timestamps[-1]],
        y=[gravity_mag, gravity_mag],
        mode='lines',
        name='g',
        line=dict(color='gray', width=1, dash='dash'),
        legendgroup='acc'
    ), row=1, col=1)
    
    # Gyroscope subplot
    fig.add_trace(go.Scatter(
        x=timestamps, y=gyros[:, 0],
        name='gyro_x',
        line=dict(color='red', width=1, dash='dot'),
        legendgroup='gyro'
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=timestamps, y=gyros[:, 1],
        name='gyro_y',
        line=dict(color='green', width=1, dash='dot'),
        legendgroup='gyro'
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=timestamps, y=gyros[:, 2],
        name='gyro_z',
        line=dict(color='blue', width=1, dash='dot'),
        legendgroup='gyro'
    ), row=2, col=1)
    
    # Calculate and display statistics
    accel_norm = np.linalg.norm(accels, axis=1)
    gyro_norm = np.linalg.norm(gyros, axis=1)
    
    stats_text = (
        f"Accel: mean={np.mean(accel_norm):.2f} m/s², "
        f"std={np.std(accel_norm):.2f}<br>"
        f"Gyro: mean={np.mean(gyro_norm):.2f} rad/s, "
        f"std={np.std(gyro_norm):.2f}"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=1, col=1)
    fig.update_yaxes(title_text="Angular velocity (rad/s)", row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=title,
        height=700,
        showlegend=True,
        hovermode='x unified',
        annotations=[
            dict(
                text=stats_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1
            )
        ]
    )
    
    return fig


def plot_landmarks_3d(
    landmarks: Map,
    camera_frames: Optional[List[CameraFrame]] = None,
    title: str = "3D Landmark Map",
    show_ids: bool = False,
    color_by_observations: bool = True
) -> go.Figure:
    """
    Create 3D scatter plot of landmarks.
    
    Args:
        landmarks: Map containing landmarks
        camera_frames: Optional camera observations to color by observation count
        title: Plot title
        show_ids: Whether to show landmark IDs
        color_by_observations: Color landmarks by observation count
    
    Returns:
        Plotly figure
    """
    # Get landmark positions
    positions = landmarks.get_positions()
    landmark_ids = list(landmarks.landmarks.keys())
    
    # Count observations per landmark if camera frames provided
    observation_counts = {lid: 0 for lid in landmark_ids}
    if camera_frames and color_by_observations:
        for frame in camera_frames:
            for obs in frame.observations:
                if obs.landmark_id in observation_counts:
                    observation_counts[obs.landmark_id] += 1
    
    # Prepare colors based on observation count
    if color_by_observations and camera_frames:
        colors = [observation_counts[lid] for lid in landmark_ids]
        colorscale = 'Viridis'
        colorbar_title = 'Observations'
    else:
        colors = 'blue'
        colorscale = None
        colorbar_title = None
    
    # Create figure
    fig = go.Figure()
    
    # Add landmarks
    hover_text = []
    for lid in landmark_ids:
        text = f"ID: {lid}"
        if color_by_observations and camera_frames:
            text += f"<br>Observations: {observation_counts[lid]}"
        hover_text.append(text)
    
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='markers+text' if show_ids else 'markers',
        text=[str(lid) for lid in landmark_ids] if show_ids else None,
        textposition='top center',
        marker=dict(
            size=5,
            color=colors,
            colorscale=colorscale,
            showscale=bool(color_by_observations and camera_frames),
            colorbar=dict(title=colorbar_title) if colorbar_title else None,
            opacity=0.8
        ),
        hovertemplate='%{hovertext}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>',
        hovertext=hover_text,
        name='Landmarks'
    ))
    
    # Add statistics
    stats_text = f"Total landmarks: {len(landmark_ids)}<br>"
    
    if len(positions) > 0:
        stats_text += (
            f"X range: [{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}]<br>"
            f"Y range: [{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}]<br>"
            f"Z range: [{positions[:, 2].min():.1f}, {positions[:, 2].max():.1f}]"
        )
    else:
        stats_text += "No landmarks to display"
    
    if camera_frames and color_by_observations:
        observed = sum(1 for c in observation_counts.values() if c > 0)
        avg_obs = np.mean(list(observation_counts.values()))
        stats_text += f"<br>Observed: {observed}/{len(landmark_ids)}<br>"
        stats_text += f"Avg observations: {avg_obs:.1f}"
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X (m)', showgrid=True),
            yaxis=dict(title='Y (m)', showgrid=True),
            zaxis=dict(title='Z (m)', showgrid=True),
            aspectmode='data'
        ),
        showlegend=False,
        annotations=[
            dict(
                text=stats_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="gray",
                borderwidth=1
            )
        ],
        height=600
    )
    
    return fig


def plot_camera_coverage(
    camera_frames: List[CameraFrame],
    intrinsics: CameraIntrinsics,
    title: str = "Camera Coverage Heatmap"
) -> go.Figure:
    """
    Create heatmap showing camera coverage over image plane.
    
    Args:
        camera_frames: Camera observations
        intrinsics: Camera intrinsics
        title: Plot title
    
    Returns:
        Plotly figure
    """
    # Create 2D histogram of observations
    bin_size = 20  # pixels
    x_bins = int(intrinsics.width / bin_size)
    y_bins = int(intrinsics.height / bin_size)
    
    coverage = np.zeros((y_bins, x_bins))
    
    for frame in camera_frames:
        for obs in frame.observations:
            u = obs.pixel.u if hasattr(obs.pixel, 'u') else obs.pixel[0]
            v = obs.pixel.v if hasattr(obs.pixel, 'v') else obs.pixel[1]
            
            x_idx = min(int(u / bin_size), x_bins - 1)
            y_idx = min(int(v / bin_size), y_bins - 1)
            
            if 0 <= x_idx < x_bins and 0 <= y_idx < y_bins:
                coverage[y_idx, x_idx] += 1
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=coverage,
        x=np.arange(x_bins) * bin_size + bin_size/2,
        y=np.arange(y_bins) * bin_size + bin_size/2,
        colorscale='Hot',
        reversescale=True,
        colorbar=dict(title='Observations'),
        hovertemplate='u: %{x:.0f}<br>v: %{y:.0f}<br>Count: %{z}<extra></extra>'
    ))
    
    # Add image boundary
    fig.add_shape(
        type="rect",
        x0=0, y0=0,
        x1=intrinsics.width, y1=intrinsics.height,
        line=dict(color="white", width=2)
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(
            title='u (pixels)',
            range=[0, intrinsics.width],
            constrain='domain'
        ),
        yaxis=dict(
            title='v (pixels)',
            range=[intrinsics.height, 0],  # Invert Y
            scaleanchor='x',
            scaleratio=1
        ),
        width=800,
        height=600
    )
    
    return fig