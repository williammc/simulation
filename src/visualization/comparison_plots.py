"""
Visualization tools for comparing multiple SLAM estimators.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

from src.evaluation.comparison import ComparisonResult, EstimatorPerformance
from src.common.data_structures import Trajectory


def create_comparison_dashboard(
    comparison_result: ComparisonResult,
    output_path: Optional[str] = None,
    show: bool = True
) -> go.Figure:
    """
    Create interactive dashboard comparing estimator performances.
    
    Args:
        comparison_result: Results from estimator comparison
        output_path: Optional path to save HTML file
        show: Whether to display the figure
    
    Returns:
        Plotly figure object
    """
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Trajectory Error (ATE)",
            "Runtime Performance",
            "Relative Pose Error",
            "Memory Usage",
            "Consistency (NEES)",
            "Overall Comparison"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "scatter"}]
        ]
    )
    
    # Extract data
    estimators = list(comparison_result.performances.keys())
    colors = px.colors.qualitative.Set1[:len(estimators)]
    
    # 1. ATE comparison
    ate_rmse = [perf.trajectory_metrics.ate_rmse for perf in comparison_result.performances.values()]
    ate_mean = [perf.trajectory_metrics.ate_mean for perf in comparison_result.performances.values()]
    
    fig.add_trace(
        go.Bar(name='ATE RMSE', x=estimators, y=ate_rmse, 
               marker_color=colors[0], showlegend=False),
        row=1, col=1
    )
    
    # 2. Runtime performance
    runtime = [perf.runtime_ms for perf in comparison_result.performances.values()]
    fig.add_trace(
        go.Bar(name='Runtime', x=estimators, y=runtime,
               marker_color=colors[1], showlegend=False),
        row=1, col=2
    )
    
    # 3. RPE comparison
    rpe_trans = [perf.trajectory_metrics.rpe_trans_rmse for perf in comparison_result.performances.values()]
    rpe_rot = [perf.trajectory_metrics.rpe_rot_rmse for perf in comparison_result.performances.values()]
    
    fig.add_trace(
        go.Bar(name='RPE Trans', x=estimators, y=rpe_trans,
               marker_color=colors[2], showlegend=False),
        row=2, col=1
    )
    
    # 4. Memory usage
    memory = [perf.peak_memory_mb for perf in comparison_result.performances.values()]
    fig.add_trace(
        go.Bar(name='Memory', x=estimators, y=memory,
               marker_color=colors[3], showlegend=False),
        row=2, col=2
    )
    
    # 5. NEES consistency
    nees_mean = []
    for perf in comparison_result.performances.values():
        if perf.consistency_metrics:
            nees_mean.append(perf.consistency_metrics.nees_mean)
        else:
            nees_mean.append(None)
    
    valid_nees = [(e, n) for e, n in zip(estimators, nees_mean) if n is not None]
    if valid_nees:
        est_with_nees, nees_values = zip(*valid_nees)
        fig.add_trace(
            go.Bar(name='NEES', x=est_with_nees, y=nees_values,
                   marker_color=colors[4], showlegend=False),
            row=3, col=1
        )
        # Add chi-squared bounds (3 DOF)
        fig.add_hline(y=3.0, line_dash="dash", line_color="green", 
                     annotation_text="Expected", row=3, col=1)
    
    # 6. Overall comparison (radar chart style as scatter)
    # Normalize metrics to 0-1 scale for comparison
    metrics_normalized = {}
    for name, perf in comparison_result.performances.items():
        metrics_normalized[name] = {
            'ATE': 1.0 - min(perf.trajectory_metrics.ate_rmse / max(ate_rmse), 1.0) if max(ate_rmse) > 0 else 0,
            'Speed': 1.0 - min(perf.runtime_ms / max(runtime), 1.0) if max(runtime) > 0 else 0,
            'Memory': 1.0 - min(perf.peak_memory_mb / max(memory), 1.0) if max(memory) > 0 else 0,
            'RPE': 1.0 - min(perf.trajectory_metrics.rpe_trans_rmse / max(rpe_trans), 1.0) if max(rpe_trans) > 0 else 0
        }
    
    for i, (name, metrics) in enumerate(metrics_normalized.items()):
        x_vals = list(metrics.keys())
        y_vals = list(metrics.values())
        fig.add_trace(
            go.Scatter(name=name, x=x_vals, y=y_vals,
                      mode='lines+markers', line=dict(color=colors[i]),
                      showlegend=True),
            row=3, col=2
        )
    
    # Update layout
    fig.update_layout(
        title="Estimator Comparison Dashboard",
        height=900,
        showlegend=True,
        legend=dict(x=0.85, y=0.15)
    )
    
    # Update axes
    fig.update_yaxes(title_text="Error (m)", row=1, col=1)
    fig.update_yaxes(title_text="Time (ms)", row=1, col=2)
    fig.update_yaxes(title_text="Error (m)", row=2, col=1)
    fig.update_yaxes(title_text="Memory (MB)", row=2, col=2)
    fig.update_yaxes(title_text="NEES", row=3, col=1)
    fig.update_yaxes(title_text="Score", range=[0, 1], row=3, col=2)
    
    # Save if requested
    if output_path:
        fig.write_html(output_path)
    
    if show:
        fig.show()
    
    return fig


def plot_trajectory_comparison(
    trajectories: Dict[str, Trajectory],
    ground_truth: Optional[Trajectory] = None,
    output_path: Optional[str] = None,
    show: bool = True
) -> go.Figure:
    """
    Plot multiple trajectories for visual comparison.
    
    Args:
        trajectories: Dictionary of trajectories by estimator name
        ground_truth: Optional ground truth trajectory
        output_path: Optional path to save HTML file
        show: Whether to display the figure
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    # Plot ground truth if available
    if ground_truth:
        positions = np.array([state.pose.position for state in ground_truth.states])
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='lines',
            name='Ground Truth',
            line=dict(color='black', width=3, dash='dash')
        ))
    
    # Plot estimated trajectories
    for i, (name, traj) in enumerate(trajectories.items()):
        positions = np.array([state.pose.position for state in traj.states])
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='lines+markers',
            name=name,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=2)
        ))
    
    fig.update_layout(
        title="Trajectory Comparison",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode='data'
        ),
        height=700
    )
    
    if output_path:
        fig.write_html(output_path)
    
    if show:
        fig.show()
    
    return fig


def plot_error_over_time(
    comparison_result: ComparisonResult,
    error_data: Dict[str, np.ndarray],
    timestamps: np.ndarray,
    output_path: Optional[str] = None,
    show: bool = True
) -> go.Figure:
    """
    Plot error evolution over time for each estimator.
    
    Args:
        comparison_result: Comparison results
        error_data: Dictionary of error arrays by estimator name
        timestamps: Time stamps for error values
        output_path: Optional path to save HTML file
        show: Whether to display the figure
    
    Returns:
        Plotly figure object
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Position Error Over Time", "Error Statistics"),
        row_heights=[0.7, 0.3]
    )
    
    colors = px.colors.qualitative.Set1
    
    # Plot error over time
    for i, (name, errors) in enumerate(error_data.items()):
        fig.add_trace(
            go.Scatter(
                x=timestamps[:len(errors)],
                y=errors,
                mode='lines',
                name=name,
                line=dict(color=colors[i % len(colors)])
            ),
            row=1, col=1
        )
    
    # Add statistics box plot
    error_values = []
    labels = []
    for name, errors in error_data.items():
        error_values.extend(errors.tolist())
        labels.extend([name] * len(errors))
    
    fig.add_trace(
        go.Box(
            y=error_values,
            x=labels,
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Error (m)", row=1, col=1)
    fig.update_xaxes(title_text="Estimator", row=2, col=1)
    fig.update_yaxes(title_text="Error (m)", row=2, col=1)
    
    fig.update_layout(
        title="Error Analysis Over Time",
        height=700
    )
    
    if output_path:
        fig.write_html(output_path)
    
    if show:
        fig.show()
    
    return fig


def plot_performance_metrics(
    comparison_result: ComparisonResult,
    output_path: Optional[str] = None,
    show: bool = True
) -> go.Figure:
    """
    Create detailed performance metrics visualization.
    
    Args:
        comparison_result: Comparison results
        output_path: Optional path to save HTML file
        show: Whether to display the figure
    
    Returns:
        Plotly figure object
    """
    # Convert to dataframe for easier plotting
    df = comparison_result.to_dataframe()
    
    # Create subplot for each metric
    metrics = ['ate_rmse', 'runtime_ms', 'memory_mb', 'rpe_trans_rmse']
    titles = ['ATE RMSE (m)', 'Runtime (ms)', 'Memory (MB)', 'RPE Translation (m)']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=titles
    )
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        if metric in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df['estimator'],
                    y=df[metric],
                    name=title,
                    showlegend=False,
                    text=df[metric].round(3),
                    textposition='auto'
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title="Performance Metrics Comparison",
        height=600
    )
    
    if output_path:
        fig.write_html(output_path)
    
    if show:
        fig.show()
    
    return fig


def generate_html_report(
    comparison_result: ComparisonResult,
    output_dir: str,
    include_plots: bool = True
) -> str:
    """
    Generate comprehensive HTML report.
    
    Args:
        comparison_result: Comparison results
        output_dir: Directory to save report files
        include_plots: Whether to include interactive plots
    
    Returns:
        Path to generated HTML report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots if requested
    plot_files = []
    if include_plots:
        # Dashboard
        dashboard_fig = create_comparison_dashboard(comparison_result, show=False)
        dashboard_path = output_dir / "dashboard.html"
        dashboard_fig.write_html(str(dashboard_path))
        plot_files.append("dashboard.html")
        
        # Performance metrics
        perf_fig = plot_performance_metrics(comparison_result, show=False)
        perf_path = output_dir / "performance.html"
        perf_fig.write_html(str(perf_path))
        plot_files.append("performance.html")
    
    # Generate main report HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SLAM Estimator Comparison Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #333;
                border-bottom: 2px solid #4CAF50;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #555;
                margin-top: 30px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                background-color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .best {{
                background-color: #d4edda !important;
                font-weight: bold;
            }}
            .plot-container {{
                margin: 20px 0;
                padding: 20px;
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metadata {{
                background-color: #e9ecef;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <h1>SLAM Estimator Comparison Report</h1>
        
        <div class="metadata">
            <h2>Simulation Metadata</h2>
            <p><strong>Number of Poses:</strong> {comparison_result.simulation_metadata.get('num_poses', 'N/A')}</p>
            <p><strong>Number of Landmarks:</strong> {comparison_result.simulation_metadata.get('num_landmarks', 'N/A')}</p>
            <p><strong>Trajectory Length:</strong> {comparison_result.simulation_metadata.get('trajectory_length', 0):.2f} m</p>
            <p><strong>Best Estimator:</strong> {comparison_result.best_estimator or 'N/A'}</p>
        </div>
        
        <h2>Performance Summary</h2>
        {_generate_html_table(comparison_result)}
        
        <h2>Statistical Tests</h2>
        {_generate_statistical_tests_html(comparison_result.statistical_tests)}
    """
    
    # Add plots if generated
    if plot_files:
        html_content += """
        <h2>Interactive Visualizations</h2>
        """
        for plot_file in plot_files:
            html_content += f"""
        <div class="plot-container">
            <iframe src="{plot_file}" width="100%" height="700" frameborder="0"></iframe>
        </div>
            """
    
    html_content += """
    </body>
    </html>
    """
    
    # Save report
    report_path = output_dir / "report.html"
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    return str(report_path)


def _generate_html_table(comparison_result: ComparisonResult) -> str:
    """Generate HTML table from comparison results."""
    df = comparison_result.to_dataframe()
    
    html = "<table>\n<tr>"
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr>\n"
    
    best_estimator = comparison_result.best_estimator
    
    for _, row in df.iterrows():
        row_class = ' class="best"' if row['estimator'] == best_estimator else ''
        html += f"<tr{row_class}>"
        for val in row:
            html += f"<td>{val}</td>"
        html += "</tr>\n"
    
    html += "</table>"
    return html


def _generate_statistical_tests_html(tests: Dict) -> str:
    """Generate HTML for statistical test results."""
    if not tests:
        return "<p>No statistical tests performed.</p>"
    
    html = "<table>\n<tr><th>Comparison</th><th>ATE Difference</th><th>Percent Difference</th><th>Better</th></tr>\n"
    
    for key, value in tests.items():
        if key == "ranking":
            continue
        if isinstance(value, dict) and 'ate_difference' in value:
            html += f"""<tr>
                <td>{key.replace('_', ' vs ')}</td>
                <td>{value['ate_difference']:.4f}</td>
                <td>{value['percent_difference']:.1f}%</td>
                <td>{value['better']}</td>
            </tr>\n"""
    
    html += "</table>\n"
    
    if "ranking" in tests:
        html += f"<p><strong>Overall Ranking:</strong> {' > '.join(tests['ranking'])}</p>"
    
    return html