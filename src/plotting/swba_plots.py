"""
Plotting functions for SWBA (Sliding Window Bundle Adjustment) visualization.
Uses Plotly for interactive visualizations.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Any
from pathlib import Path


def plot_sliding_window_state(
    keyframe_positions: np.ndarray,
    keyframe_times: np.ndarray,
    keyframe_ids: List[int],
    iteration: int,
    window_size: int,
    output_path: Optional[str] = None,
    show: bool = False
) -> go.Figure:
    """
    Visualize the current sliding window state.
    
    Args:
        keyframe_positions: Positions of keyframes in window (Nx3)
        keyframe_times: Timestamps of keyframes
        keyframe_ids: IDs of keyframes
        iteration: Current iteration number
        window_size: Maximum window size
        output_path: Optional path to save HTML
        show: Whether to display the figure
    
    Returns:
        Plotly figure object
    """
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=(
            f'Sliding Window Trajectory (Iteration {iteration})',
            f'Window Timeline ({len(keyframe_ids)}/{window_size} keyframes)'
        ),
        specs=[[{'type': 'scatter'}, {'type': 'bar'}]]
    )
    
    # Plot 1: 2D trajectory with sliding window
    fig.add_trace(
        go.Scatter(
            x=keyframe_positions[:, 0],
            y=keyframe_positions[:, 1],
            mode='lines+markers',
            name='Window Trajectory',
            line=dict(color='blue', width=3),
            marker=dict(size=8, color='blue'),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Highlight start and end of window
    if len(keyframe_positions) > 0:
        fig.add_trace(
            go.Scatter(
                x=[keyframe_positions[0, 0]],
                y=[keyframe_positions[0, 1]],
                mode='markers',
                name='Window Start',
                marker=dict(size=15, color='green', symbol='circle'),
                showlegend=True
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[keyframe_positions[-1, 0]],
                y=[keyframe_positions[-1, 1]],
                mode='markers',
                name='Window End',
                marker=dict(size=15, color='red', symbol='square'),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Update subplot 1 layout
    fig.update_xaxes(title_text="X (m)", row=1, col=1)
    fig.update_yaxes(title_text="Y (m)", row=1, col=1)
    
    # Plot 2: Window timeline
    time_in_window = keyframe_times - keyframe_times[0] if len(keyframe_times) > 0 else []
    labels = [f'KF {id}' for id in keyframe_ids]
    
    fig.add_trace(
        go.Bar(
            y=labels,
            x=time_in_window,
            orientation='h',
            marker=dict(color='blue', opacity=0.6),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Update subplot 2 layout
    fig.update_xaxes(title_text="Time in Window (s)", row=1, col=2)
    
    # Update overall layout
    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        title_text=f"SWBA Sliding Window State"
    )
    
    if output_path:
        fig.write_html(output_path)
    
    if show:
        fig.show()
    
    return fig


def plot_swba_optimization_analysis(
    window_sizes: List[int],
    optimization_iterations: List[int],
    position_errors: np.ndarray,
    timestamps: np.ndarray,
    gt_positions: np.ndarray,
    est_positions: np.ndarray,
    rmse: float,
    mean_error: float,
    output_path: Optional[str] = None,
    show: bool = True
) -> go.Figure:
    """
    Create comprehensive SWBA optimization analysis plots.
    
    Args:
        window_sizes: Window size at each iteration
        optimization_iterations: Iterations where optimization occurred
        position_errors: Position errors over time
        timestamps: Timestamps for errors
        gt_positions: Ground truth positions
        est_positions: Estimated positions
        rmse: Root mean square error
        mean_error: Mean error
        output_path: Optional path to save HTML
        show: Whether to display the figure
    
    Returns:
        Plotly figure object
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Window Size Evolution',
            'Optimization Points',
            '2D Trajectory Comparison',
            'Position Error Over Time',
            'Position Components',
            'Error Distribution'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'scatter'}],
            [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'histogram'}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Plot 1: Window size evolution
    fig.add_trace(
        go.Scatter(
            x=list(range(len(window_sizes))),
            y=window_sizes,
            mode='lines',
            name='Window Size',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Add max window size line
    if window_sizes:
        max_size = max(window_sizes)
        fig.add_trace(
            go.Scatter(
                x=[0, len(window_sizes)-1],
                y=[max_size, max_size],
                mode='lines',
                name=f'Max: {max_size}',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
    
    # Plot 2: Optimization points
    if optimization_iterations:
        fig.add_trace(
            go.Bar(
                x=optimization_iterations,
                y=[window_sizes[i-1] if i-1 < len(window_sizes) else 0 
                   for i in optimization_iterations],
                name='Optimization',
                marker=dict(color='green', opacity=0.7)
            ),
            row=1, col=2
        )
    
    # Plot 3: 2D trajectory comparison
    fig.add_trace(
        go.Scatter(
            x=gt_positions[:, 0],
            y=gt_positions[:, 1],
            mode='lines',
            name='Ground Truth',
            line=dict(color='green', width=2)
        ),
        row=1, col=3
    )
    
    fig.add_trace(
        go.Scatter(
            x=est_positions[:, 0],
            y=est_positions[:, 1],
            mode='lines',
            name='SWBA',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=1, col=3
    )
    
    # Plot 4: Position error over time
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=position_errors,
            mode='lines',
            name='Error',
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.1)'
        ),
        row=2, col=1
    )
    
    # Add RMSE line
    fig.add_trace(
        go.Scatter(
            x=[timestamps[0], timestamps[-1]],
            y=[rmse, rmse],
            mode='lines',
            name=f'RMSE: {rmse:.3f}m',
            line=dict(color='black', dash='dash')
        ),
        row=2, col=1
    )
    
    # Plot 5: Position components
    colors = ['green', 'blue', 'red']
    labels = ['X', 'Y', 'Z']
    
    for i, (color, label) in enumerate(zip(colors, labels)):
        # Ground truth
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=gt_positions[:, i],
                mode='lines',
                name=f'{label} (GT)',
                line=dict(color=color, width=2),
                opacity=0.7
            ),
            row=2, col=2
        )
        
        # Estimated
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=est_positions[:, i],
                mode='lines',
                name=f'{label} (Est)',
                line=dict(color=color, width=2, dash='dash')
            ),
            row=2, col=2
        )
    
    # Plot 6: Error histogram
    fig.add_trace(
        go.Histogram(
            x=position_errors,
            nbinsx=25,
            name='Error Distribution',
            marker=dict(color='red', opacity=0.7),
            showlegend=False
        ),
        row=2, col=3
    )
    
    # Add RMSE and mean lines
    fig.add_trace(
        go.Scatter(
            x=[rmse, rmse],
            y=[0, len(position_errors)/5],  # Approximate height
            mode='lines',
            name=f'RMSE: {rmse:.3f}m',
            line=dict(color='black', dash='dash', width=2)
        ),
        row=2, col=3
    )
    
    fig.add_trace(
        go.Scatter(
            x=[mean_error, mean_error],
            y=[0, len(position_errors)/5],
            mode='lines',
            name=f'Mean: {mean_error:.3f}m',
            line=dict(color='blue', dash='dash', width=2)
        ),
        row=2, col=3
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Keyframe", row=1, col=1)
    fig.update_yaxes(title_text="Window Size", row=1, col=1)
    
    fig.update_xaxes(title_text="Keyframe", row=1, col=2)
    fig.update_yaxes(title_text="Window Size", row=1, col=2)
    
    fig.update_xaxes(title_text="X (m)", row=1, col=3)
    fig.update_yaxes(title_text="Y (m)", row=1, col=3)
    
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Position Error (m)", row=2, col=1)
    
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Position (m)", row=2, col=2)
    
    fig.update_xaxes(title_text="Position Error (m)", row=2, col=3)
    fig.update_yaxes(title_text="Frequency", row=2, col=3)
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(x=1.02, y=0.5),
        title_text="SWBA Optimization Analysis"
    )
    
    if output_path:
        fig.write_html(output_path)
    
    if show:
        fig.show()
    
    return fig


def create_swba_animation(
    window_snapshots: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    show: bool = True
) -> go.Figure:
    """
    Create an animated visualization of sliding window evolution.
    
    Args:
        window_snapshots: List of window states over time
        output_path: Optional path to save HTML
        show: Whether to display the figure
    
    Returns:
        Plotly figure object with animation
    """
    if not window_snapshots:
        return go.Figure()
    
    # Create frames for animation
    frames = []
    for i, snapshot in enumerate(window_snapshots):
        positions = snapshot['positions']
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    mode='lines+markers',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                ),
                go.Scatter(
                    x=[positions[0, 0]],
                    y=[positions[0, 1]],
                    mode='markers',
                    marker=dict(size=15, color='green', symbol='circle')
                ),
                go.Scatter(
                    x=[positions[-1, 0]],
                    y=[positions[-1, 1]],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='square')
                )
            ],
            name=f'frame_{i}',
            layout=go.Layout(
                title=f"Sliding Window - Iteration {snapshot['iteration']}"
            )
        )
        frames.append(frame)
    
    # Initial data
    initial_pos = window_snapshots[0]['positions']
    
    fig = go.Figure(
        data=[
            go.Scatter(
                x=initial_pos[:, 0],
                y=initial_pos[:, 1],
                mode='lines+markers',
                name='Window',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ),
            go.Scatter(
                x=[initial_pos[0, 0]],
                y=[initial_pos[0, 1]],
                mode='markers',
                name='Start',
                marker=dict(size=15, color='green', symbol='circle')
            ),
            go.Scatter(
                x=[initial_pos[-1, 0]],
                y=[initial_pos[-1, 1]],
                mode='markers',
                name='End',
                marker=dict(size=15, color='red', symbol='square')
            )
        ],
        frames=frames
    )
    
    # Add animation controls
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(label='Play',
                         method='animate',
                         args=[None, {'frame': {'duration': 500}}]),
                    dict(label='Pause',
                         method='animate',
                         args=[[None], {'frame': {'duration': 0}, 'mode': 'immediate'}])
                ]
            )
        ],
        sliders=[{
            'steps': [
                {
                    'args': [[f'frame_{i}'], {'frame': {'duration': 0}, 'mode': 'immediate'}],
                    'label': f'{i}',
                    'method': 'animate'
                }
                for i in range(len(frames))
            ],
            'active': 0,
            'y': 0,
            'len': 0.9,
            'x': 0.05,
            'xanchor': 'left',
            'y': 0,
            'yanchor': 'top'
        }],
        xaxis=dict(title="X (m)"),
        yaxis=dict(title="Y (m)", scaleanchor="x"),
        title="SWBA Sliding Window Animation",
        height=600
    )
    
    if output_path:
        fig.write_html(output_path)
    
    if show:
        fig.show()
    
    return fig