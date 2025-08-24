"""
Trajectory visualization tools using Plotly.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

from src.common.data_structures import Trajectory, Pose


def plot_trajectory_3d(
    trajectory: Trajectory,
    title: str = "3D Trajectory",
    show_orientation: bool = True,
    orientation_scale: float = 0.5,
    color: str = "blue",
    name: str = "Trajectory",
    show_grid: bool = True,
    show_axes: bool = True
) -> go.Figure:
    """
    Create 3D trajectory plot using Plotly.
    
    Args:
        trajectory: Trajectory to plot
        title: Plot title
        show_orientation: Whether to show orientation arrows
        orientation_scale: Scale factor for orientation arrows
        color: Trajectory color
        name: Trajectory name for legend
        show_grid: Whether to show grid
        show_axes: Whether to show axes
    
    Returns:
        Plotly figure object
    """
    # Extract positions
    if len(trajectory.states) > 0:
        positions = np.array([state.pose.position for state in trajectory.states])
    else:
        # Handle empty trajectory
        positions = np.array([[0, 0, 0]])  # Single point at origin
    
    # Create figure
    fig = go.Figure()
    
    # Add trajectory line
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0] if len(positions) > 0 else [0],
        y=positions[:, 1] if len(positions) > 0 else [0],
        z=positions[:, 2] if len(positions) > 0 else [0],
        mode='lines+markers',
        name=name,
        line=dict(
            color=color,
            width=3
        ),
        marker=dict(
            size=3,
            color=color,
            opacity=0.8
        ),
        hovertemplate=(
            '<b>%{text}</b><br>' +
            'X: %{x:.3f}<br>' +
            'Y: %{y:.3f}<br>' +
            'Z: %{z:.3f}<br>' +
            '<extra></extra>'
        ),
        text=[f't={state.pose.timestamp:.2f}s' for state in trajectory.states]
    ))
    
    # Add start and end markers (if trajectory is not empty)
    if len(trajectory.states) > 0:
        fig.add_trace(go.Scatter3d(
            x=[positions[0, 0]],
            y=[positions[0, 1]],
            z=[positions[0, 2]],
            mode='markers',
            name='Start',
            marker=dict(
                size=10,
                color='green',
                symbol='diamond'
            ),
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[positions[-1, 0]],
            y=[positions[-1, 1]],
            z=[positions[-1, 2]],
            mode='markers',
            name='End',
            marker=dict(
                size=10,
                color='red',
                symbol='square'
            ),
            showlegend=True
        ))
    
    # Add orientation arrows if requested
    if show_orientation:
        # Sample poses to avoid cluttering
        num_arrows = min(20, len(trajectory.states))
        indices = np.linspace(0, len(trajectory.states)-1, num_arrows, dtype=int)
        
        for idx in indices:
            state = trajectory.states[idx]
            pos = state.pose.position
            
            # Get rotation matrix
            R = state.pose.rotation_matrix
            
            # Extract forward direction (x-axis in body frame)
            forward = R[:, 0] * orientation_scale
            
            # Add arrow as a line
            fig.add_trace(go.Scatter3d(
                x=[pos[0], pos[0] + forward[0]],
                y=[pos[1], pos[1] + forward[1]],
                z=[pos[2], pos[2] + forward[2]],
                mode='lines',
                line=dict(
                    color='orange',
                    width=2
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Add cone at arrow tip
            fig.add_trace(go.Cone(
                x=[pos[0] + forward[0]],
                y=[pos[1] + forward[1]],
                z=[pos[2] + forward[2]],
                u=[forward[0] * 0.2],
                v=[forward[1] * 0.2],
                w=[forward[2] * 0.2],
                colorscale=[[0, 'orange'], [1, 'orange']],
                showscale=False,
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(
                title='X (m)',
                showgrid=show_grid,
                showaxeslabels=show_axes
            ),
            yaxis=dict(
                title='Y (m)',
                showgrid=show_grid,
                showaxeslabels=show_axes
            ),
            zaxis=dict(
                title='Z (m)',
                showgrid=show_grid,
                showaxeslabels=show_axes
            ),
            aspectmode='data',  # Equal aspect ratio
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode='closest'
    )
    
    return fig


def plot_trajectory_comparison(
    ground_truth: Trajectory,
    estimated: Optional[Trajectory] = None,
    title: str = "Trajectory Comparison",
    show_orientation: bool = False,
    show_error: bool = True
) -> go.Figure:
    """
    Plot ground truth vs estimated trajectory comparison.
    
    Args:
        ground_truth: Ground truth trajectory
        estimated: Estimated trajectory (optional, placeholder for now)
        title: Plot title
        show_orientation: Whether to show orientation arrows
        show_error: Whether to show error plot (if estimated provided)
    
    Returns:
        Plotly figure object
    """
    if estimated is None:
        # Just plot ground truth for now (placeholder)
        return plot_trajectory_3d(
            ground_truth,
            title=f"{title} (Ground Truth Only)",
            show_orientation=show_orientation,
            color='blue',
            name='Ground Truth'
        )
    
    # Create figure with subplots if showing error
    if show_error:
        fig = make_subplots(
            rows=2, cols=2,
            row_heights=[0.7, 0.3],
            column_widths=[0.7, 0.3],
            specs=[
                [{'type': 'scatter3d', 'rowspan': 1, 'colspan': 1}, 
                 {'type': 'scatter', 'rowspan': 1, 'colspan': 1}],
                [{'type': 'scatter', 'colspan': 2}, None]
            ],
            subplot_titles=('3D Trajectories', 'Position Error', 'Error over Time')
        )
    else:
        fig = go.Figure()
    
    # Extract positions
    gt_positions = np.array([state.pose.position for state in ground_truth.states])
    est_positions = np.array([state.pose.position for state in estimated.states])
    
    # Add ground truth trajectory
    fig.add_trace(go.Scatter3d(
        x=gt_positions[:, 0],
        y=gt_positions[:, 1],
        z=gt_positions[:, 2],
        mode='lines',
        name='Ground Truth',
        line=dict(
            color='blue',
            width=3
        ),
        opacity=0.7
    ), row=1 if show_error else None, col=1 if show_error else None)
    
    # Add estimated trajectory
    fig.add_trace(go.Scatter3d(
        x=est_positions[:, 0],
        y=est_positions[:, 1],
        z=est_positions[:, 2],
        mode='lines',
        name='Estimated',
        line=dict(
            color='red',
            width=3,
            dash='dash'
        ),
        opacity=0.7
    ), row=1 if show_error else None, col=1 if show_error else None)
    
    if show_error and len(est_positions) > 0:
        # Calculate errors at estimated timestamps
        est_timestamps = [state.pose.timestamp for state in estimated.states]
        errors = []
        
        # For each estimated state, find closest ground truth and compute error
        for est_state in estimated.states:
            est_pos = est_state.pose.position
            est_time = est_state.pose.timestamp
            
            # Find closest ground truth state by timestamp
            closest_gt_state = min(ground_truth.states, 
                                   key=lambda s: abs(s.pose.timestamp - est_time))
            gt_pos = closest_gt_state.pose.position
            
            # Compute error
            error = np.linalg.norm(est_pos - gt_pos)
            errors.append(error)
        
        errors = np.array(errors)
        timestamps = est_timestamps
        
        # Add error histogram
        fig.add_trace(go.Histogram(
            x=errors,
            nbinsx=30,
            name='Error Distribution',
            marker_color='purple',
            showlegend=False
        ), row=1, col=2)
        
        # Add error over time
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=errors,
            mode='lines',
            name='Position Error',
            line=dict(color='purple', width=2),
            showlegend=False
        ), row=2, col=1)
        
        # Add mean error line
        mean_error = np.mean(errors)
        fig.add_trace(go.Scatter(
            x=[timestamps[0], timestamps[-1]],
            y=[mean_error, mean_error],
            mode='lines',
            name=f'Mean Error: {mean_error:.3f}m',
            line=dict(color='orange', width=2, dash='dash'),
            showlegend=True
        ), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=title,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        height=800 if show_error else 600
    )
    
    # Update 3D scene
    if show_error:
        fig.update_scenes(
            dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            ),
            row=1, col=1
        )
        
        # Update 2D axes
        fig.update_xaxes(title_text="Error (m)", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Error (m)", row=2, col=1)
    else:
        fig.update_layout(
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            )
        )
    
    return fig


def save_trajectory_plot(
    fig: go.Figure,
    filepath: Union[str, Path],
    include_plotlyjs: str = 'cdn',
    auto_open: bool = False
) -> Path:
    """
    Save trajectory plot to HTML file.
    
    Args:
        fig: Plotly figure to save
        filepath: Output file path
        include_plotlyjs: How to include plotly.js ('cdn', 'directory', 'inline', etc.)
        auto_open: Whether to open in browser after saving
    
    Returns:
        Path to saved file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure .html extension
    if filepath.suffix != '.html':
        filepath = filepath.with_suffix('.html')
    
    # Save figure
    fig.write_html(
        str(filepath),
        include_plotlyjs=include_plotlyjs,
        auto_open=auto_open
    )
    
    return filepath


def plot_trajectory_components(
    trajectory: Trajectory,
    title: str = "Trajectory Components"
) -> go.Figure:
    """
    Plot trajectory components (position, velocity, orientation) over time.
    
    Args:
        trajectory: Trajectory to plot
        title: Plot title
    
    Returns:
        Plotly figure with subplots
    """
    # Extract data
    timestamps = [state.pose.timestamp for state in trajectory.states]
    positions = np.array([state.pose.position for state in trajectory.states])
    velocities = np.array([state.velocity if state.velocity is not None else np.zeros(3) 
                          for state in trajectory.states])
    
    # Extract Euler angles from rotation matrices
    from src.utils.math_utils import rotation_matrix_to_euler
    euler_angles = np.array([rotation_matrix_to_euler(state.pose.rotation_matrix) 
                             for state in trajectory.states])
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Position', 'Velocity', 'Orientation (Euler Angles)'),
        vertical_spacing=0.1
    )
    
    # Position subplot
    fig.add_trace(go.Scatter(x=timestamps, y=positions[:, 0], name='X', 
                             line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=positions[:, 1], name='Y',
                             line=dict(color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=positions[:, 2], name='Z',
                             line=dict(color='blue')), row=1, col=1)
    
    # Velocity subplot
    fig.add_trace(go.Scatter(x=timestamps, y=velocities[:, 0], name='Vx',
                             line=dict(color='red', dash='dash')), row=2, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=velocities[:, 1], name='Vy',
                             line=dict(color='green', dash='dash')), row=2, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=velocities[:, 2], name='Vz',
                             line=dict(color='blue', dash='dash')), row=2, col=1)
    
    # Orientation subplot (Euler angles in degrees)
    fig.add_trace(go.Scatter(x=timestamps, y=np.rad2deg(euler_angles[:, 0]), name='Roll',
                             line=dict(color='red', dash='dot')), row=3, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=np.rad2deg(euler_angles[:, 1]), name='Pitch',
                             line=dict(color='green', dash='dot')), row=3, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=np.rad2deg(euler_angles[:, 2]), name='Yaw',
                             line=dict(color='blue', dash='dot')), row=3, col=1)
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Position (m)", row=1, col=1)
    fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=1)
    fig.update_yaxes(title_text="Angle (deg)", row=3, col=1)
    
    # Update layout
    fig.update_layout(
        title=title,
        height=800,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def plot_trajectory_with_keyframes(
    trajectory: Trajectory,
    keyframe_indices: List[int],
    title: str = "Trajectory with Keyframes",
    show_orientation: bool = True,
    orientation_scale: float = 0.5,
    show_grid: bool = True,
    show_axes: bool = True
) -> go.Figure:
    """
    Create 3D trajectory plot with keyframes highlighted.
    
    Args:
        trajectory: Trajectory to plot
        keyframe_indices: Indices of keyframes in trajectory
        title: Plot title
        show_orientation: Whether to show orientation arrows
        orientation_scale: Scale factor for orientation arrows
        show_grid: Whether to show grid
        show_axes: Whether to show axes
    
    Returns:
        Plotly figure object with keyframes highlighted
    """
    # Extract positions
    if len(trajectory.states) > 0:
        positions = np.array([state.pose.position for state in trajectory.states])
        timestamps = [state.pose.timestamp for state in trajectory.states]
    else:
        positions = np.array([[0, 0, 0]])
        timestamps = [0.0]
    
    # Separate keyframe and non-keyframe positions
    keyframe_mask = np.zeros(len(positions), dtype=bool)
    for idx in keyframe_indices:
        if 0 <= idx < len(positions):
            keyframe_mask[idx] = True
    
    keyframe_positions = positions[keyframe_mask]
    keyframe_times = [timestamps[i] for i in range(len(timestamps)) if keyframe_mask[i]]
    
    # Create figure
    fig = go.Figure()
    
    # Add full trajectory line
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='lines',
        name='Trajectory',
        line=dict(
            color='lightblue',
            width=2
        ),
        hovertemplate=(
            '<b>Trajectory</b><br>' +
            'Time: %{text}<br>' +
            'X: %{x:.3f}<br>' +
            'Y: %{y:.3f}<br>' +
            'Z: %{z:.3f}<br>' +
            '<extra></extra>'
        ),
        text=[f'{t:.2f}s' for t in timestamps]
    ))
    
    # Add keyframe markers
    if len(keyframe_positions) > 0:
        fig.add_trace(go.Scatter3d(
            x=keyframe_positions[:, 0],
            y=keyframe_positions[:, 1],
            z=keyframe_positions[:, 2],
            mode='markers',
            name=f'Keyframes ({len(keyframe_positions)})',
            marker=dict(
                size=8,
                color='red',
                symbol='diamond',
                line=dict(color='darkred', width=1)
            ),
            hovertemplate=(
                '<b>Keyframe</b><br>' +
                'Time: %{text}<br>' +
                'X: %{x:.3f}<br>' +
                'Y: %{y:.3f}<br>' +
                'Z: %{z:.3f}<br>' +
                '<extra></extra>'
            ),
            text=[f'{t:.2f}s' for t in keyframe_times]
        ))
    
    # Add orientation arrows if requested
    if show_orientation and len(keyframe_positions) > 0:
        arrow_traces = []
        for i, idx in enumerate(keyframe_indices):
            if 0 <= idx < len(trajectory.states):
                state = trajectory.states[idx]
                pos = state.pose.position
                R = state.pose.rotation_matrix
                
                # Create arrow for forward direction (x-axis)
                direction = R[:, 0] * orientation_scale
                arrow_traces.append(go.Cone(
                    x=[pos[0] + direction[0]/2],
                    y=[pos[1] + direction[1]/2],
                    z=[pos[2] + direction[2]/2],
                    u=[direction[0]],
                    v=[direction[1]],
                    w=[direction[2]],
                    sizemode='absolute',
                    sizeref=orientation_scale/2,
                    showscale=False,
                    colorscale=[[0, 'red'], [1, 'red']],
                    name='Keyframe Orientation',
                    showlegend=i == 0  # Only show legend for first arrow
                ))
        
        for trace in arrow_traces:
            fig.add_trace(trace)
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(
                title='X (m)',
                showgrid=show_grid,
                showbackground=True,
                backgroundcolor='rgba(230, 230, 230, 0.5)',
                visible=show_axes
            ),
            yaxis=dict(
                title='Y (m)',
                showgrid=show_grid,
                showbackground=True,
                backgroundcolor='rgba(230, 230, 230, 0.5)',
                visible=show_axes
            ),
            zaxis=dict(
                title='Z (m)',
                showgrid=show_grid,
                showbackground=True,
                backgroundcolor='rgba(230, 230, 230, 0.5)',
                visible=show_axes
            ),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        hovermode='closest'
    )
    
    return fig