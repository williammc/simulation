"""
Dashboard generation for SLAM visualization.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import json

from src.common.data_structures import (
    Trajectory, IMUData, CameraData, Map,
    CameraCalibration, IMUCalibration
)
from src.common.json_io import load_simulation_data
from .trajectory_plot import plot_trajectory_3d, plot_trajectory_components
from .sensor_plot import plot_imu_data, plot_camera_tracks, plot_landmarks_3d


@dataclass
class DashboardConfig:
    """Configuration for dashboard generation."""
    show_trajectory_3d: bool = True
    show_trajectory_components: bool = True
    show_imu_data: bool = True
    show_camera_tracks: bool = True
    show_landmarks: bool = True
    show_kpis: bool = True
    show_metadata: bool = True
    title: str = "SLAM Simulation Dashboard"
    height_per_row: int = 400


def create_dashboard(
    simulation_file: Path,
    output_file: Optional[Path] = None,
    config: Optional[DashboardConfig] = None,
    estimated_trajectory: Optional[Trajectory] = None
) -> Path:
    """
    Create comprehensive dashboard from simulation data.
    
    Args:
        simulation_file: Path to simulation JSON file
        output_file: Output HTML file path (default: same name as input)
        config: Dashboard configuration
        estimated_trajectory: Optional estimated trajectory for comparison
    
    Returns:
        Path to generated dashboard HTML
    """
    config = config or DashboardConfig()
    
    # Load simulation data
    data = load_simulation_data(simulation_file)
    trajectory = data['trajectory']
    landmarks = data['landmarks']
    imu_data = data['imu_data']
    camera_data = data['camera_data']
    camera_calibrations = data['camera_calibrations']
    imu_calibrations = data['imu_calibrations']
    metadata = data['metadata']
    
    # Determine output file
    if output_file is None:
        output_file = simulation_file.with_suffix('.dashboard.html')
    else:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create HTML content
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{config.title}</title>
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
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
        }}
        .header p {{
            margin: 5px 0;
            opacity: 0.9;
        }}
        .kpi-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .kpi-card {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .kpi-title {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        .kpi-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .kpi-unit {{
            font-size: 14px;
            color: #95a5a6;
            margin-left: 5px;
        }}
        .plot-container {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .plot-title {{
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
        }}
        .metadata-container {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }}
        .metadata-item {{
            padding: 5px;
        }}
        .metadata-label {{
            font-weight: bold;
            color: #34495e;
        }}
        .metadata-value {{
            color: #7f8c8d;
            margin-left: 10px;
        }}
        .footer {{
            text-align: center;
            color: #95a5a6;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{config.title}</h1>
        <p>Simulation: {simulation_file.name}</p>
        <p>Generated: {metadata.get('timestamp', 'N/A')}</p>
    </div>
"""
    
    # Add KPIs section
    if config.show_kpis:
        html_content += """
    <div class="kpi-container">
"""
        # Calculate KPIs
        duration = trajectory.get_time_range()[1] - trajectory.get_time_range()[0]
        num_poses = len(trajectory.states)
        num_landmarks = len(landmarks.landmarks)
        num_imu = len(imu_data.measurements) if imu_data else 0
        num_camera = len(camera_data.frames) if camera_data else 0
        
        # Total observations
        total_obs = 0
        if camera_data:
            for frame in camera_data.frames:
                total_obs += len(frame.observations)
        
        kpis = [
            ("Duration", f"{duration:.1f}", "seconds"),
            ("Poses", str(num_poses), ""),
            ("Landmarks", str(num_landmarks), ""),
            ("IMU Samples", str(num_imu), ""),
            ("Camera Frames", str(num_camera), ""),
            ("Observations", str(total_obs), ""),
        ]
        
        # Add placeholder KPIs for estimation (will be filled when we have estimators)
        if estimated_trajectory:
            kpis.extend([
                ("ATE RMSE", "N/A", "m"),
                ("RPE RMSE", "N/A", "m"),
                ("Runtime", "N/A", "ms"),
            ])
        
        for title, value, unit in kpis:
            html_content += f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}<span class="kpi-unit">{unit}</span></div>
        </div>
"""
        
        html_content += """
    </div>
"""
    
    # Add metadata section
    if config.show_metadata:
        html_content += """
    <div class="metadata-container">
        <div class="plot-title">Simulation Metadata</div>
        <div class="metadata-grid">
"""
        metadata_items = [
            ("Trajectory Type", metadata.get('trajectory_type', 'N/A')),
            ("Coordinate System", metadata.get('coordinate_system', 'N/A')),
            ("Random Seed", str(metadata.get('seed', 'N/A'))),
            ("Camera Rate", f"{camera_data.rate if camera_data else 'N/A'} Hz"),
            ("IMU Rate", f"{imu_data.rate if imu_data else 'N/A'} Hz"),
            ("Camera Model", camera_calibrations[0].intrinsics.model.value if camera_calibrations else 'N/A'),
        ]
        
        for label, value in metadata_items:
            html_content += f"""
            <div class="metadata-item">
                <span class="metadata-label">{label}:</span>
                <span class="metadata-value">{value}</span>
            </div>
"""
        
        html_content += """
        </div>
    </div>
"""
    
    # Generate plots
    plots = []
    
    # 3D Trajectory
    if config.show_trajectory_3d:
        fig = plot_trajectory_3d(trajectory, title="Ground Truth Trajectory")
        plots.append(("3D Trajectory", fig))
    
    # Trajectory Components
    if config.show_trajectory_components:
        fig = plot_trajectory_components(trajectory)
        plots.append(("Trajectory Components", fig))
    
    # IMU Data
    if config.show_imu_data and imu_data:
        fig = plot_imu_data(imu_data)
        plots.append(("IMU Measurements", fig))
    
    # Camera Tracks
    if config.show_camera_tracks and camera_data and camera_calibrations:
        fig = plot_camera_tracks(
            camera_data,
            camera_calibrations[0].intrinsics,
            max_frames=50  # Limit for performance
        )
        plots.append(("Camera Feature Tracks", fig))
    
    # Landmarks
    if config.show_landmarks and landmarks:
        fig = plot_landmarks_3d(
            landmarks,
            camera_data.frames if camera_data else None
        )
        plots.append(("3D Landmark Map", fig))
    
    # Add plots to HTML
    for plot_title, fig in plots:
        # Convert figure to HTML div
        plot_html = fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id=plot_title.replace(" ", "_").lower()
        )
        
        html_content += f"""
    <div class="plot-container">
        <div class="plot-title">{plot_title}</div>
        {plot_html}
    </div>
"""
    
    # Add footer
    html_content += """
    <div class="footer">
        <p>SLAM Simulation System - Dashboard Generated with Plotly</p>
    </div>
</body>
</html>
"""
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    return output_file


def create_kpi_summary(
    simulation_file: Path,
    estimated_file: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Generate KPI summary from simulation and optionally estimation results.
    
    Args:
        simulation_file: Path to simulation JSON
        estimated_file: Optional path to estimation results JSON
    
    Returns:
        Dictionary of KPIs
    """
    # Load simulation data
    data = load_simulation_data(simulation_file)
    trajectory = data['trajectory']
    landmarks = data['landmarks']
    imu_data = data['imu_data']
    camera_data = data['camera_data']
    metadata = data['metadata']
    
    # Calculate basic KPIs
    time_range = trajectory.get_time_range()
    duration = time_range[1] - time_range[0]
    
    # Count observations
    total_observations = 0
    unique_landmarks_observed = set()
    if camera_data:
        for frame in camera_data.frames:
            total_observations += len(frame.observations)
            for obs in frame.observations:
                unique_landmarks_observed.add(obs.landmark_id)
    
    kpis = {
        "simulation": {
            "file": simulation_file.name,
            "trajectory_type": metadata.get('trajectory_type', 'unknown'),
            "duration": duration,
            "num_poses": len(trajectory.states),
            "num_landmarks": len(landmarks.landmarks),
            "num_imu_measurements": len(imu_data.measurements) if imu_data else 0,
            "num_camera_frames": len(camera_data.frames) if camera_data else 0,
            "total_observations": total_observations,
            "unique_landmarks_observed": len(unique_landmarks_observed),
            "observation_rate": total_observations / duration if duration > 0 else 0,
        }
    }
    
    # Add estimation KPIs if available (placeholder for now)
    if estimated_file and estimated_file.exists():
        kpis["estimation"] = {
            "file": estimated_file.name,
            "ate_rmse": 0.0,  # Placeholder
            "ate_max": 0.0,   # Placeholder
            "ate_median": 0.0, # Placeholder
            "rpe_rmse": 0.0,  # Placeholder
            "rpe_max": 0.0,   # Placeholder
            "runtime_ms": 0.0, # Placeholder
            "success": True    # Placeholder
        }
    
    return kpis