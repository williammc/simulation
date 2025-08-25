# Plotting Module Documentation

## Overview

The plotting module (`src/plotting/`) provides comprehensive visualization tools for SLAM data analysis, including trajectories, sensor measurements, estimation results, and performance metrics. It supports both static plots for reports and interactive visualizations for debugging.

## Table of Contents
- [Architecture](#architecture)
- [Trajectory Visualization](#trajectory-visualization)
- [Sensor Data Plots](#sensor-data-plots)
- [Estimation Results](#estimation-results)
- [Error Analysis](#error-analysis)
- [Interactive Visualizations](#interactive-visualizations)
- [Dashboard Generation](#dashboard-generation)
- [Usage Examples](#usage-examples)

## Architecture

```
src/plotting/
├── __init__.py
├── trajectory_plotter.py     # 3D trajectory visualization
├── sensor_plotter.py        # IMU/camera data plots
├── error_plotter.py         # Error metrics visualization
├── landmark_plotter.py      # Map and landmark visualization
├── dashboard_generator.py   # Multi-panel dashboards
├── interactive_viewer.py    # Real-time interactive plots
└── utils.py                # Common plotting utilities
```

## Trajectory Visualization

### 3D Trajectory Plots

The `TrajectoryPlotter` class provides comprehensive trajectory visualization:

```python
from src.plotting import TrajectoryPlotter

plotter = TrajectoryPlotter(figsize=(12, 8))

# Basic trajectory plot
plotter.plot_trajectory(
    trajectory,
    color='blue',
    label='Estimated',
    show_orientation=True
)

# Compare multiple trajectories
plotter.plot_comparison(
    ground_truth=gt_trajectory,
    estimated=est_trajectory,
    show_error=True
)

# Animated trajectory
plotter.animate_trajectory(
    trajectory,
    speed=2.0,
    save_path='trajectory.mp4'
)
```

### Trajectory Features

#### Position and Orientation
```python
# Plot with coordinate frames
plotter.plot_with_frames(
    trajectory,
    frame_spacing=10,  # Show frame every 10 poses
    frame_size=0.5
)

# Color by velocity
plotter.plot_colored_by_velocity(
    trajectory,
    cmap='viridis',
    show_colorbar=True
)

# Plot with uncertainty ellipsoids
plotter.plot_with_covariance(
    trajectory,
    covariances,
    confidence=0.95,
    alpha=0.3
)
```

#### Top-down View
```python
# 2D bird's eye view
plotter.plot_top_down(
    trajectory,
    show_heading=True,
    grid=True
)

# With landmarks
plotter.plot_map_view(
    trajectory,
    landmarks,
    show_visibility=True
)
```

## Sensor Data Plots

### IMU Visualization

The `SensorPlotter` handles IMU data visualization:

```python
from src.plotting import SensorPlotter

sensor_plotter = SensorPlotter()

# Plot IMU measurements
fig = sensor_plotter.plot_imu_data(
    imu_measurements,
    components=['accel', 'gyro'],
    show_bias=True
)

# Acceleration components
sensor_plotter.plot_acceleration(
    imu_measurements,
    show_magnitude=True,
    show_gravity=True
)

# Angular velocity
sensor_plotter.plot_angular_velocity(
    imu_measurements,
    units='deg/s'
)

# IMU bias evolution
sensor_plotter.plot_bias_evolution(
    bias_estimates,
    show_3sigma_bounds=True
)
```

### Camera Observations

```python
# Plot camera observations
sensor_plotter.plot_camera_observations(
    camera_frames,
    image_size=(640, 480),
    show_tracks=True
)

# Feature tracks
sensor_plotter.plot_feature_tracks(
    observations,
    max_track_length=50,
    color_by_length=True
)

# Reprojection errors
sensor_plotter.plot_reprojection_errors(
    observations,
    predictions,
    histogram=True
)
```

### Multi-Sensor Timeline

```python
# Sensor timing diagram
sensor_plotter.plot_sensor_timeline(
    imu_times,
    camera_times,
    keyframe_times,
    duration=10.0
)

# Data rate analysis
sensor_plotter.plot_data_rates(
    sensor_data,
    window_size=1.0  # 1 second windows
)
```

## Estimation Results

### State Evolution

```python
from src.plotting import EstimationPlotter

est_plotter = EstimationPlotter()

# Plot state evolution
est_plotter.plot_state_evolution(
    states,
    components=['position', 'velocity', 'orientation']
)

# Pose components over time
fig, axes = est_plotter.plot_pose_components(
    trajectory,
    include_rates=True  # Show velocity and angular velocity
)

# Landmark estimates
est_plotter.plot_landmark_evolution(
    landmark_history,
    show_initialization=True,
    show_final=True
)
```

### Optimization Metrics

```python
# Cost function evolution
est_plotter.plot_optimization_cost(
    iteration_costs,
    log_scale=True
)

# Factor graph structure
est_plotter.plot_factor_graph(
    graph,
    show_factors=True,
    show_variables=True,
    layout='spring'
)

# Information matrix
est_plotter.plot_information_matrix(
    information_matrix,
    log_scale=True,
    colormap='hot'
)
```

## Error Analysis

### Trajectory Errors

The `ErrorPlotter` provides comprehensive error visualization:

```python
from src.plotting import ErrorPlotter

error_plotter = ErrorPlotter()

# Absolute trajectory error
error_plotter.plot_ate(
    ground_truth,
    estimated,
    plot_type='time_series'
)

# Relative pose error
error_plotter.plot_rpe(
    ground_truth,
    estimated,
    segment_length=1.0  # 1 meter segments
)

# Error statistics
error_plotter.plot_error_statistics(
    errors,
    metrics=['mean', 'std', 'max', 'rmse']
)
```

### Error Distributions

```python
# Error histograms
error_plotter.plot_error_distribution(
    position_errors,
    orientation_errors,
    fit_distribution='normal'
)

# Box plots by segment
error_plotter.plot_error_boxplot(
    errors_by_segment,
    group_by='distance'
)

# Error heatmap
error_plotter.plot_error_heatmap(
    x_errors,
    y_errors,
    bins=50
)
```

### Drift Analysis

```python
# Cumulative drift
error_plotter.plot_drift(
    trajectory,
    ground_truth,
    components=['x', 'y', 'z', 'yaw']
)

# Loop closure analysis
error_plotter.plot_loop_closure_errors(
    loop_closures,
    before_optimization=True,
    after_optimization=True
)
```

## Interactive Visualizations

### Real-time Viewer

```python
from src.plotting import InteractiveViewer

viewer = InteractiveViewer()

# Start real-time visualization
viewer.start()

# Update in loop
for data in data_stream:
    viewer.update_trajectory(data.pose)
    viewer.update_landmarks(data.landmarks)
    viewer.update_observations(data.observations)
    
    # Update metrics panel
    viewer.update_metrics({
        'speed': data.velocity.norm(),
        'landmarks': len(data.landmarks),
        'error': compute_error(data)
    })

viewer.stop()
```

### Interactive 3D Plot

```python
# Plotly-based interactive plot
from src.plotting import create_interactive_plot

fig = create_interactive_plot(
    trajectory=trajectory,
    landmarks=landmarks,
    observations=observations
)

# Add controls
fig.add_slider('time', 0, duration)
fig.add_checkbox('show_landmarks')
fig.add_dropdown('color_by', ['time', 'velocity', 'error'])

fig.show()
```

### Debugging Visualizer

```python
# Debug specific keyframes
from src.plotting import DebugVisualizer

debugger = DebugVisualizer()

# Visualize single keyframe
debugger.show_keyframe(
    keyframe,
    show_observations=True,
    show_preintegration=True,
    show_factors=True
)

# Step through keyframes
debugger.step_through_keyframes(
    keyframes,
    pause_duration=1.0
)
```

## Dashboard Generation

### Multi-Panel Dashboards

```python
from src.plotting import DashboardGenerator

dashboard = DashboardGenerator(figsize=(20, 12))

# Configure layout
dashboard.set_layout([
    ['trajectory_3d', 'trajectory_top'],
    ['position_error', 'orientation_error'],
    ['imu_accel', 'imu_gyro'],
    ['metrics_table', 'statistics']
])

# Generate dashboard
dashboard.generate(
    simulation_data=sim_data,
    estimation_result=result,
    ground_truth=gt_data
)

# Save in multiple formats
dashboard.save('dashboard.png', dpi=150)
dashboard.save('dashboard.pdf')
dashboard.save_html('dashboard.html', interactive=True)
```

### Comparison Dashboard

```python
# Compare multiple algorithms
comparison_dashboard = dashboard.create_comparison(
    results={
        'EKF': ekf_result,
        'GTSAM': gtsam_result,
        'SWBA': swba_result
    },
    ground_truth=gt_data
)

comparison_dashboard.add_performance_table()
comparison_dashboard.add_error_plots()
comparison_dashboard.add_trajectory_comparison()
```

## Usage Examples

### Complete Analysis Pipeline

```python
from src.plotting import create_analysis_plots

# Load data
sim_data = load_simulation('simulation.json')
result = load_estimation_result('result.json')

# Generate all plots
plots = create_analysis_plots(
    simulation=sim_data,
    result=result,
    output_dir='plots/'
)

# Specific analyses
plots.trajectory_analysis()
plots.error_analysis()
plots.sensor_analysis()
plots.optimization_analysis()

# Generate report
plots.generate_report('analysis_report.pdf')
```

### Custom Plotting

```python
import matplotlib.pyplot as plt
from src.plotting.utils import setup_3d_axes, add_grid

# Custom 3D plot
fig = plt.figure(figsize=(10, 8))
ax = setup_3d_axes(fig)

# Plot trajectory
ax.plot3D(
    trajectory.positions[:, 0],
    trajectory.positions[:, 1],
    trajectory.positions[:, 2],
    'b-', linewidth=2
)

# Add custom elements
add_grid(ax, size=10, spacing=1)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')

# Custom colormap
colors = compute_error_colors(errors)
ax.scatter3D(positions[:, 0], positions[:, 1], positions[:, 2], 
            c=colors, cmap='RdYlGn_r')

plt.show()
```

### Batch Processing

```python
from src.plotting import BatchPlotter

batch_plotter = BatchPlotter()

# Process multiple runs
for run_id in run_ids:
    data = load_run(run_id)
    
    batch_plotter.add_run(run_id, data)

# Generate comparative plots
batch_plotter.plot_all_trajectories()
batch_plotter.plot_error_statistics()
batch_plotter.plot_performance_trends()

# Save summary
batch_plotter.save_summary('batch_analysis/')
```

## Plot Customization

### Style Configuration

```python
from src.plotting import set_plot_style

# Set global style
set_plot_style('publication')  # or 'presentation', 'notebook'

# Custom style
custom_style = {
    'figure.figsize': (10, 6),
    'lines.linewidth': 2,
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.3
}
set_plot_style(custom_style)
```

### Color Schemes

```python
from src.plotting.utils import get_color_palette

# Get predefined palette
colors = get_color_palette('qualitative', n_colors=5)

# Custom gradient
gradient = create_gradient_colormap(
    start_color='blue',
    end_color='red',
    n_steps=100
)
```

### Export Options

```python
# High-quality export
plotter.save_figure(
    'figure.pdf',
    dpi=300,
    bbox_inches='tight',
    transparent=True
)

# Multi-format export
formats = ['png', 'pdf', 'svg', 'eps']
for fmt in formats:
    plotter.save_figure(f'figure.{fmt}')

# LaTeX-ready export
plotter.export_tikz('figure.tex', standalone=True)
```

## Performance Optimization

### Large Dataset Handling

```python
# Downsample for visualization
from src.plotting.utils import downsample_trajectory

# Adaptive downsampling
downsampled = downsample_trajectory(
    trajectory,
    max_points=1000,
    preserve_keypoints=True
)

# Level-of-detail rendering
plotter.plot_with_lod(
    trajectory,
    zoom_level=zoom,
    detail_threshold=100
)
```

### Memory Management

```python
# Streaming plot updates
plotter = StreamingPlotter(max_points=10000)

for chunk in data_chunks:
    plotter.update(chunk)
    plotter.render()
    
# Clear old data
plotter.clear_buffer()
```

## Troubleshooting

### Common Issues

1. **Memory Issues with Large Datasets**
   - Use downsampling
   - Plot in chunks
   - Use backend: 'Agg' for non-interactive

2. **Slow Rendering**
   - Reduce number of points
   - Use rasterization for dense plots
   - Disable anti-aliasing

3. **Export Quality**
   - Increase DPI for prints
   - Use vector formats (PDF/SVG)
   - Check font embedding

4. **3D Navigation**
   - Use `matplotlib` backend: 'TkAgg'
   - Enable mouse interaction
   - Set initial view angle

## References

- Matplotlib Documentation: https://matplotlib.org/
- Plotly for Interactive Plots: https://plotly.com/
- Scientific Visualization: "Visualization Analysis and Design" (Munzner)