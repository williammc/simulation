"""
Visualization and plotting tools for SLAM simulation.
"""

from .trajectory_plot import (
    plot_trajectory_3d,
    plot_trajectory_comparison,
    save_trajectory_plot
)
from .sensor_plot import (
    plot_imu_data,
    plot_camera_tracks,
    plot_landmarks_3d
)
from .dashboard import (
    create_dashboard,
    DashboardConfig
)

__all__ = [
    'plot_trajectory_3d',
    'plot_trajectory_comparison',
    'save_trajectory_plot',
    'plot_imu_data',
    'plot_camera_tracks',
    'plot_landmarks_3d',
    'create_dashboard',
    'DashboardConfig'
]