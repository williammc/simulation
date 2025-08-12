"""
Dashboard generation for SLAM evaluation results.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import json


def create_evaluation_dashboard(
    comparison_results: Dict,
    kpis: Dict,
    config: Dict,
    output_dir: Path
) -> Path:
    """
    Create a comprehensive evaluation dashboard.
    
    Args:
        comparison_results: Comparison results for all datasets
        kpis: Extracted KPIs
        config: Dashboard configuration
        output_dir: Output directory for dashboard files
        
    Returns:
        Path to the main dashboard HTML file
    """
    dashboard = EvaluationDashboard(comparison_results, kpis, config)
    return dashboard.generate(output_dir)


class EvaluationDashboard:
    """Generates comprehensive evaluation dashboard."""
    
    def __init__(self, comparison_results: Dict, kpis: Dict, config: Dict):
        """
        Initialize dashboard generator.
        
        Args:
            comparison_results: All comparison results
            kpis: Extracted KPIs
            config: Dashboard configuration
        """
        self.comparison_results = comparison_results
        self.kpis = kpis
        self.config = config
        self.figures = []
        
    def generate(self, output_dir: Path) -> Path:
        """
        Generate the complete dashboard.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Path to main dashboard file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate each section
        for section in self.config.get('sections', []):
            self._generate_section(section, output_dir)
        
        # Create main dashboard HTML
        dashboard_path = self._create_main_dashboard(output_dir)
        
        return dashboard_path
    
    def _generate_section(self, section: Dict, output_dir: Path):
        """Generate a dashboard section."""
        section_type = section.get('type')
        
        if section_type == 'summary_table':
            fig = self._create_summary_table(section)
        elif section_type == '3d_plot':
            fig = self._create_trajectory_comparison(section)
        elif section_type == 'time_series':
            fig = self._create_error_evolution(section)
        elif section_type == 'heatmap':
            fig = self._create_performance_heatmap(section)
        elif section_type == 'bar_chart':
            fig = self._create_bar_chart(section)
        elif section_type == 'nees_plot':
            fig = self._create_nees_plot(section)
        elif section_type == 'ranking_table':
            fig = self._create_ranking_table(section)
        else:
            return
        
        if fig:
            self.figures.append({
                'name': section.get('name', 'Unnamed'),
                'figure': fig,
                'type': section_type
            })
    
    def _create_summary_table(self, config: Dict) -> go.Figure:
        """Create a summary table of key metrics."""
        metrics = config.get('metrics', ['ate_rmse', 'runtime_ms'])
        
        # Prepare data for table
        headers = ['Dataset', 'Estimator'] + [m.upper() for m in metrics]
        cells = [[] for _ in headers]
        
        for dataset_name, comparison in self.comparison_results.items():
            if not comparison or not comparison.performances:
                continue
                
            for estimator_name, performance in comparison.performances.items():
                cells[0].append(dataset_name)
                cells[1].append(estimator_name)
                
                for i, metric in enumerate(metrics):
                    value = self._extract_metric_value(performance, metric)
                    cells[i + 2].append(f"{value:.3f}" if value is not None else "N/A")
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=headers,
                fill_color='paleturquoise',
                align='left',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=cells,
                fill_color='lavender',
                align='left',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title="Performance Summary",
            height=400
        )
        
        return fig
    
    def _create_trajectory_comparison(self, config: Dict) -> go.Figure:
        """Create 3D trajectory comparison plot."""
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        color_idx = 0
        
        # Plot trajectories for each dataset
        for dataset_name, comparison in self.comparison_results.items():
            if config.get('datasets') != 'all' and dataset_name not in config.get('datasets', []):
                continue
                
            if not comparison:
                continue
            
            # Plot ground truth if available
            if hasattr(comparison, 'ground_truth') and comparison.ground_truth:
                positions = np.array([
                    state.pose.position for state in comparison.ground_truth.states
                ])
                fig.add_trace(go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=positions[:, 2],
                    mode='lines',
                    name=f"{dataset_name} - Ground Truth",
                    line=dict(color='black', width=2, dash='dash'),
                    showlegend=True
                ))
            
            # Plot estimated trajectories
            if hasattr(comparison, 'estimator_results'):
                for estimator_name, result in comparison.estimator_results.items():
                    if 'trajectory' in result:
                        traj = result['trajectory']
                        positions = np.array([
                            state.pose.position for state in traj.states
                        ])
                        fig.add_trace(go.Scatter3d(
                            x=positions[:, 0],
                            y=positions[:, 1],
                            z=positions[:, 2],
                            mode='lines',
                            name=f"{dataset_name} - {estimator_name}",
                            line=dict(color=colors[color_idx % len(colors)], width=2),
                            showlegend=True
                        ))
                        color_idx += 1
        
        fig.update_layout(
            title="Trajectory Comparison",
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                aspectmode='data'
            ),
            height=600
        )
        
        return fig
    
    def _create_error_evolution(self, config: Dict) -> go.Figure:
        """Create error evolution over time plot."""
        metrics = config.get('metrics', ['position_error', 'orientation_error'])
        
        fig = make_subplots(
            rows=len(metrics), cols=1,
            subplot_titles=[m.replace('_', ' ').title() for m in metrics],
            shared_xaxes=True
        )
        
        colors = px.colors.qualitative.Set2
        
        for row, metric in enumerate(metrics, 1):
            color_idx = 0
            for dataset_name, comparison in self.comparison_results.items():
                if not comparison or not hasattr(comparison, 'error_evolution'):
                    continue
                    
                for estimator_name, errors in comparison.error_evolution.get(metric, {}).items():
                    if 'timestamps' in errors and 'values' in errors:
                        fig.add_trace(
                            go.Scatter(
                                x=errors['timestamps'],
                                y=errors['values'],
                                mode='lines',
                                name=f"{dataset_name[:10]}-{estimator_name}",
                                line=dict(color=colors[color_idx % len(colors)]),
                                showlegend=(row == 1)
                            ),
                            row=row, col=1
                        )
                        color_idx += 1
        
        fig.update_xaxes(title_text="Time (s)", row=len(metrics), col=1)
        for row in range(1, len(metrics) + 1):
            fig.update_yaxes(title_text="Error", row=row, col=1)
        
        fig.update_layout(
            title="Error Evolution Over Time",
            height=300 * len(metrics)
        )
        
        return fig
    
    def _create_performance_heatmap(self, config: Dict) -> go.Figure:
        """Create performance heatmap."""
        metric = config.get('metric', 'ate_rmse')
        
        # Prepare data matrix
        datasets = sorted(set(self.comparison_results.keys()))
        estimators = sorted(set(
            est for comp in self.comparison_results.values()
            if comp and comp.performances
            for est in comp.performances.keys()
        ))
        
        matrix = np.full((len(datasets), len(estimators)), np.nan)
        
        for i, dataset in enumerate(datasets):
            comparison = self.comparison_results.get(dataset)
            if not comparison or not comparison.performances:
                continue
                
            for j, estimator in enumerate(estimators):
                if estimator in comparison.performances:
                    perf = comparison.performances[estimator]
                    value = self._extract_metric_value(perf, metric)
                    if value is not None:
                        matrix[i, j] = value
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=estimators,
            y=datasets,
            colorscale='RdYlGn_r',
            text=np.round(matrix, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title=metric.upper())
        ))
        
        fig.update_layout(
            title=f"Performance Matrix - {metric.upper()}",
            xaxis_title="Estimators",
            yaxis_title="Datasets",
            height=400 + 20 * len(datasets)
        )
        
        return fig
    
    def _create_bar_chart(self, config: Dict) -> go.Figure:
        """Create bar chart for specific metric."""
        metric = config.get('metric', 'runtime_ms')
        group_by = config.get('group_by', 'estimator')
        
        # Prepare data
        data = []
        for dataset_name, comparison in self.comparison_results.items():
            if not comparison or not comparison.performances:
                continue
                
            for estimator_name, performance in comparison.performances.items():
                value = self._extract_metric_value(performance, metric)
                if value is not None:
                    data.append({
                        'dataset': dataset_name,
                        'estimator': estimator_name,
                        'value': value
                    })
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        
        # Create grouped bar chart
        if group_by == 'estimator':
            fig = px.bar(
                df,
                x='estimator',
                y='value',
                color='dataset',
                barmode='group',
                title=f"{metric.replace('_', ' ').title()} by Estimator"
            )
        else:
            fig = px.bar(
                df,
                x='dataset',
                y='value',
                color='estimator',
                barmode='group',
                title=f"{metric.replace('_', ' ').title()} by Dataset"
            )
        
        fig.update_layout(
            yaxis_title=metric.replace('_', ' ').title(),
            height=400
        )
        
        return fig
    
    def _create_nees_plot(self, config: Dict) -> go.Figure:
        """Create NEES consistency plot."""
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        color_idx = 0
        
        for dataset_name, comparison in self.comparison_results.items():
            if not comparison or not comparison.performances:
                continue
                
            for estimator_name, performance in comparison.performances.items():
                if performance.consistency_metrics:
                    nees = performance.consistency_metrics.nees_history
                    if nees is not None and len(nees) > 0:
                        fig.add_trace(go.Scatter(
                            y=nees,
                            mode='lines',
                            name=f"{dataset_name[:15]}-{estimator_name}",
                            line=dict(color=colors[color_idx % len(colors)])
                        ))
                        color_idx += 1
        
        # Add chi-squared bounds
        if config.get('show_bounds', True):
            n_points = 1000  # Approximate
            chi2_lower = 2.7  # Chi-squared 5% bound for 3 DOF
            chi2_upper = 7.8  # Chi-squared 95% bound for 3 DOF
            
            fig.add_hline(
                y=chi2_lower, line_dash="dash", line_color="green",
                annotation_text="5% bound"
            )
            fig.add_hline(
                y=chi2_upper, line_dash="dash", line_color="red",
                annotation_text="95% bound"
            )
            fig.add_hline(
                y=3.0, line_dash="dot", line_color="blue",
                annotation_text="Expected (3 DOF)"
            )
        
        fig.update_layout(
            title="NEES Consistency Check",
            xaxis_title="Time Step",
            yaxis_title="NEES",
            height=400
        )
        
        return fig
    
    def _create_ranking_table(self, config: Dict) -> go.Figure:
        """Create ranking table based on weighted criteria."""
        criteria = config.get('criteria', ['ate_rmse', 'runtime_ms'])
        weights = config.get('weights', [1.0] * len(criteria))
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Compute scores for each estimator
        scores = {}
        for dataset_name, comparison in self.comparison_results.items():
            if not comparison or not comparison.performances:
                continue
                
            for estimator_name, performance in comparison.performances.items():
                if estimator_name not in scores:
                    scores[estimator_name] = []
                
                # Compute weighted score
                score = 0
                for criterion, weight in zip(criteria, weights):
                    value = self._extract_metric_value(performance, criterion)
                    if value is not None:
                        # Normalize (lower is better for most metrics)
                        normalized = 1.0 / (1.0 + value)
                        score += weight * normalized
                
                scores[estimator_name].append(score)
        
        # Average scores across datasets
        ranking = []
        for estimator, estimator_scores in scores.items():
            if estimator_scores:
                avg_score = np.mean(estimator_scores)
                ranking.append({
                    'Estimator': estimator,
                    'Score': avg_score,
                    'Datasets': len(estimator_scores)
                })
        
        # Sort by score
        ranking.sort(key=lambda x: x['Score'], reverse=True)
        
        # Add rank
        for i, item in enumerate(ranking):
            item['Rank'] = i + 1
        
        # Create table
        if ranking:
            df = pd.DataFrame(ranking)
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=['Rank', 'Estimator', 'Score', 'Datasets'],
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[
                        df['Rank'],
                        df['Estimator'],
                        df['Score'].round(3),
                        df['Datasets']
                    ],
                    fill_color=['lightgreen' if r == 1 else 'lavender' 
                               for r in df['Rank']],
                    align='left'
                )
            )])
            
            fig.update_layout(
                title="Overall Ranking",
                height=300
            )
            
            return fig
        
        return None
    
    def _extract_metric_value(self, performance: Any, metric: str) -> Optional[float]:
        """Extract a metric value from performance object."""
        if metric == 'ate_rmse':
            return performance.trajectory_metrics.ate_rmse
        elif metric == 'ate_mean':
            return performance.trajectory_metrics.ate_mean
        elif metric == 'rpe_translation_rmse' or metric == 'rpe_trans_rmse':
            return performance.trajectory_metrics.rpe_trans_rmse
        elif metric == 'runtime_ms':
            return performance.runtime_ms
        elif metric == 'peak_memory_mb' or metric == 'memory_mb':
            return performance.peak_memory_mb
        elif metric == 'success_rate' or metric == 'convergence_rate':
            return 1.0 if performance.converged else 0.0
        else:
            return None
    
    def _create_main_dashboard(self, output_dir: Path) -> Path:
        """Create the main dashboard HTML file."""
        dashboard_file = output_dir / "index.html"
        
        # Generate individual plot files
        plot_files = []
        for i, fig_info in enumerate(self.figures):
            plot_file = output_dir / f"plot_{i}.html"
            fig_info['figure'].write_html(str(plot_file))
            plot_files.append(plot_file.name)
        
        # Create main HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.config.get('title', 'SLAM Evaluation Dashboard')}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }}
                .header {{
                    background: rgba(255, 255, 255, 0.95);
                    padding: 20px;
                    text-align: center;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2d3748;
                    margin: 0;
                    font-size: 2.5em;
                }}
                .subtitle {{
                    color: #718096;
                    margin-top: 10px;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }}
                .card {{
                    background: white;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                .card-header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 15px;
                    font-size: 1.2em;
                    font-weight: bold;
                }}
                .card-content {{
                    padding: 10px;
                }}
                iframe {{
                    width: 100%;
                    height: 500px;
                    border: none;
                }}
                .kpi-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                .kpi-card {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .kpi-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #667eea;
                }}
                .kpi-label {{
                    color: #718096;
                    margin-top: 5px;
                }}
                .footer {{
                    text-align: center;
                    padding: 20px;
                    color: white;
                    margin-top: 40px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{self.config.get('title', 'SLAM Evaluation Dashboard')}</h1>
                <div class="subtitle">Comprehensive Performance Analysis</div>
            </div>
            
            <div class="container">
                {self._generate_kpi_summary()}
                
                <div class="{'grid' if self.config.get('layout') == 'grid' else 'tabs'}">
        """
        
        # Add plots
        for i, (fig_info, plot_file) in enumerate(zip(self.figures, plot_files)):
            html_content += f"""
                <div class="card">
                    <div class="card-header">{fig_info['name']}</div>
                    <div class="card-content">
                        <iframe src="{plot_file}"></iframe>
                    </div>
                </div>
            """
        
        html_content += """
                </div>
            </div>
            
            <div class="footer">
                <p>Generated with SLAM Evaluation Pipeline</p>
            </div>
        </body>
        </html>
        """
        
        with open(dashboard_file, 'w') as f:
            f.write(html_content)
        
        return dashboard_file
    
    def _generate_kpi_summary(self) -> str:
        """Generate KPI summary HTML."""
        if not self.kpis or 'summary' not in self.kpis:
            return ""
        
        html = '<div class="kpi-grid">'
        
        # Select top KPIs to display
        top_kpis = ['ate_rmse', 'runtime_ms', 'peak_memory_mb']
        
        for kpi in top_kpis:
            if kpi in self.kpis['summary']:
                # Find best performer
                best_estimator = None
                best_value = float('inf')
                
                for estimator, stats in self.kpis['summary'][kpi].items():
                    if stats['mean'] < best_value:
                        best_value = stats['mean']
                        best_estimator = estimator
                
                if best_estimator:
                    html += f"""
                    <div class="kpi-card">
                        <div class="kpi-value">{best_value:.3f}</div>
                        <div class="kpi-label">Best {kpi.replace('_', ' ').upper()}</div>
                        <div style="color: #a0aec0; font-size: 0.9em; margin-top: 5px;">
                            {best_estimator}
                        </div>
                    </div>
                    """
        
        html += '</div>'
        return html