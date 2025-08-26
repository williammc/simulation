"""
Visual debugging tools for IMU rotation accumulation and preintegration.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional

from src.common.data_structures import IMUMeasurement, PreintegratedIMUData
from src.utils.math_utils import rotation_matrix_to_euler
from scipy.spatial.transform import Rotation


def plot_rotation_accumulation(
    measurements: List[IMUMeasurement],
    preintegrated: Optional[PreintegratedIMUData] = None,
    title: str = "Rotation Accumulation Debug"
) -> go.Figure:
    """
    Plot rotation accumulation from IMU measurements.
    
    Shows:
    - Gyroscope measurements over time
    - Accumulated rotation angle (using rotation matrices)
    - Expected vs actual rotation
    - Rotation matrix components
    
    Args:
        measurements: List of IMU measurements
        preintegrated: Optional preintegrated result to compare
        title: Plot title
    
    Returns:
        Plotly figure with debug visualization
    """
    if not measurements:
        raise ValueError("No measurements provided")
    
    # Extract data
    timestamps = np.array([m.timestamp for m in measurements])
    gyros = np.array([m.gyroscope for m in measurements])
    dt = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 0.01
    
    # Accumulate rotation matrices
    R_accum = np.eye(3)
    rotation_matrices = [R_accum.copy()]
    accumulated_angles = [0.0]
    euler_angles = [[0, 0, 0]]
    
    for i, gyro in enumerate(gyros[1:], 1):
        # Create incremental rotation
        omega_dt = gyro * dt
        rot = Rotation.from_rotvec(omega_dt)
        R_dt = rot.as_matrix()
        
        # Accumulate
        R_accum = R_accum @ R_dt
        rotation_matrices.append(R_accum.copy())
        
        # Extract Euler angles for visualization
        euler = rotation_matrix_to_euler(R_accum)
        euler_angles.append(euler)
        
        # Track accumulated angle around Z (for circular motion)
        # Using atan2 for continuous angle tracking beyond ±180°
        accumulated_angles.append(np.arctan2(R_accum[1, 0], R_accum[0, 0]))
    
    rotation_matrices = np.array(rotation_matrices)
    euler_angles = np.array(euler_angles)
    accumulated_angles = np.array(accumulated_angles)
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            'Gyroscope Measurements',
            'Rotation Matrix Components',
            'Euler Angles (Roll, Pitch, Yaw)',
            'Accumulated Z-axis Rotation'
        ),
        vertical_spacing=0.08
    )
    
    # 1. Gyroscope measurements
    fig.add_trace(go.Scatter(x=timestamps, y=gyros[:, 0], name='ωx',
                             line=dict(color='red', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=gyros[:, 1], name='ωy',
                             line=dict(color='green', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=gyros[:, 2], name='ωz',
                             line=dict(color='blue', width=1)), row=1, col=1)
    
    # 2. Rotation matrix components (showing R[0,0] and R[0,1] for Z rotation)
    fig.add_trace(go.Scatter(x=timestamps, y=rotation_matrices[:, 0, 0], 
                             name='R[0,0] (cos θ)',
                             line=dict(color='purple', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=rotation_matrices[:, 0, 1],
                             name='R[0,1] (-sin θ)',
                             line=dict(color='orange', width=2)), row=2, col=1)
    
    # Add expected values if constant rotation
    if len(gyros) > 0 and np.std(gyros[:, 2]) < 0.1:  # Roughly constant Z rotation
        omega_z = np.mean(gyros[:, 2])
        expected_angles = omega_z * timestamps
        expected_cos = np.cos(expected_angles)
        expected_sin = -np.sin(expected_angles)
        fig.add_trace(go.Scatter(x=timestamps, y=expected_cos,
                                name='Expected cos',
                                line=dict(color='purple', dash='dash', width=1)), 
                     row=2, col=1)
        fig.add_trace(go.Scatter(x=timestamps, y=expected_sin,
                                name='Expected -sin',
                                line=dict(color='orange', dash='dash', width=1)), 
                     row=2, col=1)
    
    # 3. Euler angles
    fig.add_trace(go.Scatter(x=timestamps, y=np.rad2deg(euler_angles[:, 0]),
                             name='Roll', line=dict(color='red')), row=3, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=np.rad2deg(euler_angles[:, 1]),
                             name='Pitch', line=dict(color='green')), row=3, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=np.rad2deg(euler_angles[:, 2]),
                             name='Yaw', line=dict(color='blue')), row=3, col=1)
    
    # 4. Accumulated rotation angle (continuous)
    fig.add_trace(go.Scatter(x=timestamps, y=np.rad2deg(accumulated_angles),
                             name='Accumulated angle',
                             line=dict(color='darkblue', width=2)), row=4, col=1)
    
    # Add expected linear accumulation if constant rotation
    if len(gyros) > 0 and np.std(gyros[:, 2]) < 0.1:
        omega_z = np.mean(gyros[:, 2])
        expected_accumulated = np.rad2deg(omega_z * timestamps)
        fig.add_trace(go.Scatter(x=timestamps, y=expected_accumulated,
                                name='Expected (linear)',
                                line=dict(color='red', dash='dash', width=1)), 
                     row=4, col=1)
    
    # Add preintegrated result if provided
    if preintegrated:
        R_preint = preintegrated.delta_rotation
        angle_preint = np.arctan2(R_preint[1, 0], R_preint[0, 0])
        fig.add_hline(y=np.rad2deg(angle_preint), 
                     line=dict(color='green', dash='dot', width=2),
                     annotation_text=f"Preintegrated: {np.rad2deg(angle_preint):.1f}°",
                     row=4, col=1)
    
    # Update axes
    fig.update_xaxes(title_text="Time (s)", row=4, col=1)
    fig.update_yaxes(title_text="ω (rad/s)", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    fig.update_yaxes(title_text="Angle (deg)", row=3, col=1)
    fig.update_yaxes(title_text="Angle (deg)", row=4, col=1)
    
    # Update layout
    fig.update_layout(
        title=title,
        height=1000,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def plot_rotation_matrix_validation(
    R_actual: np.ndarray,
    R_expected: np.ndarray,
    angle_deg: float,
    title: Optional[str] = None
) -> go.Figure:
    """
    Visualize rotation matrix comparison for validation.
    
    Args:
        R_actual: Actual rotation matrix (3x3)
        R_expected: Expected rotation matrix (3x3)
        angle_deg: Rotation angle in degrees
        title: Optional plot title
    
    Returns:
        Plotly figure showing matrix comparison
    """
    if title is None:
        title = f"Rotation Matrix Validation - {angle_deg}°"
    
    # Create subplots for matrices
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Actual Matrix', 'Expected Matrix', 'Error Matrix',
            'Vector Test: [1,0,0]', 'Vector Test: [0,1,0]', 'Properties Check'
        ),
        specs=[
            [{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}],
            [{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'table'}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Row 1: Matrix heatmaps
    fig.add_trace(go.Heatmap(z=R_actual, colorscale='RdBu', zmid=0,
                             text=np.round(R_actual, 3),
                             texttemplate='%{text}',
                             showscale=False), row=1, col=1)
    
    fig.add_trace(go.Heatmap(z=R_expected, colorscale='RdBu', zmid=0,
                             text=np.round(R_expected, 3),
                             texttemplate='%{text}',
                             showscale=False), row=1, col=2)
    
    error_matrix = R_actual - R_expected
    fig.add_trace(go.Heatmap(z=error_matrix, colorscale='RdYlGn_r',
                             text=np.round(error_matrix, 4),
                             texttemplate='%{text}',
                             showscale=True,
                             colorbar=dict(title='Error')), row=1, col=3)
    
    # Row 2, Col 1-2: Vector rotation visualization
    vectors = [np.array([1, 0, 0]), np.array([0, 1, 0])]
    colors = ['red', 'green']
    
    for idx, (vec, color) in enumerate(zip(vectors, colors)):
        v_actual = R_actual @ vec
        v_expected = R_expected @ vec
        
        # Original vector
        fig.add_trace(go.Scatter3d(
            x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]],
            mode='lines+markers',
            line=dict(color=color, width=5),
            marker=dict(size=5),
            name='Original',
            showlegend=(idx == 0)
        ), row=2, col=idx+1)
        
        # Expected rotation
        fig.add_trace(go.Scatter3d(
            x=[0, v_expected[0]], y=[0, v_expected[1]], z=[0, v_expected[2]],
            mode='lines+markers',
            line=dict(color=color, width=5, dash='dash'),
            marker=dict(size=5),
            name='Expected',
            showlegend=(idx == 0)
        ), row=2, col=idx+1)
        
        # Actual rotation
        fig.add_trace(go.Scatter3d(
            x=[0, v_actual[0]], y=[0, v_actual[1]], z=[0, v_actual[2]],
            mode='lines+markers',
            line=dict(color='blue', width=3),
            marker=dict(size=5),
            name='Actual',
            showlegend=(idx == 0)
        ), row=2, col=idx+1)
    
    # Row 2, Col 3: Properties check
    det_actual = np.linalg.det(R_actual)
    det_expected = np.linalg.det(R_expected)
    orth_error = np.linalg.norm(R_actual @ R_actual.T - np.eye(3), 'fro')
    frob_error = np.linalg.norm(error_matrix, 'fro')
    
    # Extract angle using minimal representation
    rot_actual = Rotation.from_matrix(R_actual)
    angle_actual = np.linalg.norm(rot_actual.as_rotvec())
    
    properties_data = [
        ['Determinant (actual)', f'{det_actual:.6f}', '✓' if abs(det_actual - 1) < 1e-6 else '✗'],
        ['Determinant (expected)', f'{det_expected:.6f}', '✓'],
        ['Orthogonality error', f'{orth_error:.6e}', '✓' if orth_error < 1e-6 else '✗'],
        ['Frobenius norm error', f'{frob_error:.6f}', '✓' if frob_error < 0.01 else '✗'],
        ['Angle (minimal)', f'{np.degrees(angle_actual):.2f}°', '—'],
        ['Expected angle', f'{angle_deg:.2f}°', '—']
    ]
    
    fig.add_trace(go.Table(
        cells=dict(
            values=list(zip(*properties_data)),
            align='left',
            font=dict(size=10),
            height=25
        )
    ), row=2, col=3)
    
    # Update 3D axes
    for col in [1, 2]:
        fig.update_scenes(
            xaxis=dict(range=[-1.2, 1.2], title='X'),
            yaxis=dict(range=[-1.2, 1.2], title='Y'),
            zaxis=dict(range=[-1.2, 1.2], title='Z'),
            aspectmode='cube',
            row=2, col=col
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=800,
        showlegend=True
    )
    
    return fig


def compare_preintegration_methods(
    measurements: List[IMUMeasurement],
    results: dict,
    title: str = "Preintegration Methods Comparison"
) -> go.Figure:
    """
    Compare different preintegration methods or implementations.
    
    Args:
        measurements: List of IMU measurements
        results: Dictionary of method_name -> PreintegratedIMUData
        title: Plot title
    
    Returns:
        Plotly figure comparing results
    """
    timestamps = np.array([m.timestamp for m in measurements])
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Delta Position', 'Delta Velocity', 'Rotation Angle'
        ),
        vertical_spacing=0.1
    )
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for idx, (method_name, result) in enumerate(results.items()):
        color = colors[idx % len(colors)]
        
        # Position
        pos_norm = np.linalg.norm(result.delta_position)
        fig.add_trace(go.Scatter(
            x=[timestamps[-1]], y=[pos_norm],
            mode='markers',
            marker=dict(color=color, size=10),
            name=f'{method_name} position'
        ), row=1, col=1)
        
        # Velocity
        vel_norm = np.linalg.norm(result.delta_velocity)
        fig.add_trace(go.Scatter(
            x=[timestamps[-1]], y=[vel_norm],
            mode='markers',
            marker=dict(color=color, size=10),
            name=f'{method_name} velocity'
        ), row=2, col=1)
        
        # Rotation angle
        R = result.delta_rotation
        angle = np.arctan2(R[1, 0], R[0, 0])  # For Z-axis rotation
        fig.add_trace(go.Scatter(
            x=[timestamps[-1]], y=[np.degrees(angle)],
            mode='markers',
            marker=dict(color=color, size=10),
            name=f'{method_name} angle'
        ), row=3, col=1)
    
    # Update axes
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Position (m)", row=1, col=1)
    fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=1)
    fig.update_yaxes(title_text="Angle (deg)", row=3, col=1)
    
    fig.update_layout(
        title=title,
        height=800,
        showlegend=True
    )
    
    return fig