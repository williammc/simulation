"""
Dashboard command implementation.
Generates HTML dashboard from SLAM KPI results or simulation data.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from rich.console import Console
from rich.table import Table
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our new plotting module
try:
    from src.plotting.dashboard import create_dashboard, DashboardConfig, create_kpi_summary
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

console = Console()


def generate_dashboard(input_dir: Path, output: Path) -> int:
    """
    Generate dashboard from SLAM KPIs or simulation data.
    
    Args:
        input_dir: Directory containing SLAM KPI JSON files or path to simulation JSON
        output: Output HTML file path
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    if not input_dir.exists():
        console.print(f"[red]✗ Error: Input path not found: {input_dir}[/red]")
        return 1
    
    console.print("[bold green]Generating Dashboard[/bold green]")
    console.print(f"  Input: [cyan]{input_dir}[/cyan]")
    console.print(f"  Output: [cyan]{output}[/cyan]")
    
    # Check if input is a simulation JSON file
    if input_dir.is_file() and input_dir.suffix == '.json' and PLOTTING_AVAILABLE:
        try:
            # Detect if it's a simulation file by checking for trajectory/landmarks keys
            with open(input_dir, 'r') as f:
                data = json.load(f)
                if 'groundtruth' in data and 'measurements' in data:
                    # This is a simulation file, use new dashboard
                    console.print("  Type: [cyan]Simulation data[/cyan]")
                    config = DashboardConfig(
                        title=f"SLAM Simulation Dashboard - {input_dir.stem}"
                    )
                    dashboard_path = create_dashboard(
                        simulation_file=input_dir,
                        output_file=output,
                        config=config
                    )
                    console.print(f"\n[green]✓[/green] Dashboard generated: [cyan]{dashboard_path}[/cyan]")
                    
                    # Show KPI summary
                    kpis = create_kpi_summary(input_dir)
                    console.print("\n[bold]Simulation KPIs:[/bold]")
                    for key, value in kpis['simulation'].items():
                        console.print(f"  {key}: [cyan]{value}[/cyan]")
                    return 0
        except Exception as e:
            console.print(f"[yellow]⚠ Could not parse as simulation file: {e}[/yellow]")
            # Fall through to KPI dashboard
    
    # Load all KPI files
    kpi_files = list(input_dir.glob("*.json"))
    if not kpi_files:
        console.print(f"[yellow]⚠ No KPI files found in {input_dir}[/yellow]")
        console.print("[yellow]  Run SLAM estimators first to generate KPI data[/yellow]")
        return 1
    
    console.print(f"  Found [cyan]{len(kpi_files)}[/cyan] KPI files")
    
    # Load KPI data
    all_kpis = []
    for kpi_file in sorted(kpi_files):
        try:
            with open(kpi_file, 'r') as f:
                kpi_data = json.load(f)
                all_kpis.append(kpi_data)
        except Exception as e:
            console.print(f"[yellow]⚠ Error reading {kpi_file.name}: {e}[/yellow]")
    
    if not all_kpis:
        console.print("[red]✗ No valid KPI data found[/red]")
        return 1
    
    # Display loaded runs
    display_loaded_runs(all_kpis)
    
    # Generate dashboard HTML
    html_content = create_dashboard_html(all_kpis)
    
    # Save HTML file
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        f.write(html_content)
    
    console.print(f"\n[green]✓[/green] Dashboard generated: [cyan]{output}[/cyan]")
    console.print(f"  File size: {output.stat().st_size / 1024:.1f} KB")
    console.print(f"  Open in browser: [cyan]file://{output.absolute()}[/cyan]")
    
    return 0


def display_loaded_runs(kpis: List[Dict[str, Any]]) -> None:
    """Display a table of loaded SLAM runs."""
    table = Table(title="Loaded SLAM Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Algorithm", style="magenta")
    table.add_column("Trajectory", style="green")
    table.add_column("ATE RMSE", style="yellow")
    table.add_column("Time", style="blue")
    
    for kpi in kpis:
        table.add_row(
            kpi.get("run_id", "N/A"),
            kpi.get("algorithm", "N/A").upper(),
            kpi.get("configuration", {}).get("trajectory_type", "N/A"),
            f"{kpi.get('metrics', {}).get('trajectory_error', {}).get('ate_rmse', 0):.4f}",
            f"{kpi.get('metrics', {}).get('computational', {}).get('total_time', 0):.2f}s"
        )
    
    console.print(table)


def create_dashboard_html(kpis: List[Dict[str, Any]]) -> str:
    """
    Create HTML dashboard with Plotly visualizations.
    
    Args:
        kpis: List of KPI dictionaries
    
    Returns:
        HTML string with embedded visualizations
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Trajectory Error Comparison",
            "Computational Performance",
            "Error Distribution",
            "Convergence Analysis"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "box"}, {"type": "scatter"}]
        ]
    )
    
    # Prepare data for plots
    algorithms = []
    ate_rmse = []
    rpe_rmse = []
    comp_times = []
    iterations = []
    
    for kpi in kpis:
        alg = kpi.get("algorithm", "unknown").upper()
        algorithms.append(f"{alg}_{kpi.get('timestamp', '')[:8]}")
        
        metrics = kpi.get("metrics", {})
        ate_rmse.append(metrics.get("trajectory_error", {}).get("ate_rmse", 0))
        rpe_rmse.append(metrics.get("trajectory_error", {}).get("rpe_rmse", 0))
        comp_times.append(metrics.get("computational", {}).get("total_time", 0))
        iterations.append(metrics.get("convergence", {}).get("iterations", 0))
    
    # 1. Trajectory Error Comparison
    fig.add_trace(
        go.Bar(name="ATE RMSE", x=algorithms, y=ate_rmse, marker_color='lightblue'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name="RPE RMSE", x=algorithms, y=rpe_rmse, marker_color='lightcoral'),
        row=1, col=1
    )
    
    # 2. Computational Performance
    fig.add_trace(
        go.Bar(x=algorithms, y=comp_times, marker_color='lightgreen', showlegend=False),
        row=1, col=2
    )
    
    # 3. Error Distribution (Box plot)
    for i, alg in enumerate(algorithms):
        if i < len(kpis):
            error_data = [
                kpis[i].get("metrics", {}).get("trajectory_error", {}).get("ate_mean", 0),
                kpis[i].get("metrics", {}).get("trajectory_error", {}).get("ate_rmse", 0),
                kpis[i].get("metrics", {}).get("trajectory_error", {}).get("ate_max", 0),
            ]
            fig.add_trace(
                go.Box(y=error_data, name=alg, showlegend=False),
                row=2, col=1
            )
    
    # 4. Convergence Analysis
    fig.add_trace(
        go.Scatter(
            x=algorithms, 
            y=iterations, 
            mode='markers+lines',
            marker=dict(size=10, color='purple'),
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="SLAM Estimator Performance Dashboard",
        title_font_size=24,
        height=800,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Run", row=1, col=1)
    fig.update_yaxes(title_text="Error (m)", row=1, col=1)
    fig.update_xaxes(title_text="Run", row=1, col=2)
    fig.update_yaxes(title_text="Time (s)", row=1, col=2)
    fig.update_xaxes(title_text="Algorithm", row=2, col=1)
    fig.update_yaxes(title_text="Error (m)", row=2, col=1)
    fig.update_xaxes(title_text="Run", row=2, col=2)
    fig.update_yaxes(title_text="Iterations", row=2, col=2)
    
    # Convert to HTML
    graph_html = fig.to_html(include_plotlyjs='cdn')
    
    # Create summary statistics table
    summary_html = create_summary_table(kpis)
    
    # Combine into full HTML document
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SLAM Dashboard</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                padding: 30px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                text-align: center;
                margin-bottom: 10px;
            }}
            .timestamp {{
                text-align: center;
                color: #666;
                margin-bottom: 30px;
            }}
            .summary {{
                margin: 30px 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #667eea;
                color: white;
                font-weight: bold;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .metric-good {{
                color: #28a745;
                font-weight: bold;
            }}
            .metric-warning {{
                color: #ffc107;
                font-weight: bold;
            }}
            .metric-bad {{
                color: #dc3545;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 SLAM Estimator Performance Dashboard</h1>
            <div class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
            
            <div class="summary">
                <h2>Summary Statistics</h2>
                {summary_html}
            </div>
            
            <h2>Performance Visualizations</h2>
            {graph_html}
        </div>
    </body>
    </html>
    """
    
    return html_template


def create_summary_table(kpis: List[Dict[str, Any]]) -> str:
    """Create HTML table with summary statistics."""
    if not kpis:
        return "<p>No data available</p>"
    
    # Calculate statistics
    best_ate = min(kpis, key=lambda x: x.get("metrics", {}).get("trajectory_error", {}).get("ate_rmse", float('inf')))
    best_time = min(kpis, key=lambda x: x.get("metrics", {}).get("computational", {}).get("total_time", float('inf')))
    
    table_html = """
    <table>
        <thead>
            <tr>
                <th>Algorithm</th>
                <th>Run ID</th>
                <th>ATE RMSE (m)</th>
                <th>RPE RMSE (m)</th>
                <th>Time (s)</th>
                <th>Converged</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for kpi in sorted(kpis, key=lambda x: x.get("algorithm", "")):
        alg = kpi.get("algorithm", "unknown").upper()
        run_id = kpi.get("run_id", "N/A")
        metrics = kpi.get("metrics", {})
        
        ate = metrics.get("trajectory_error", {}).get("ate_rmse", 0)
        rpe = metrics.get("trajectory_error", {}).get("rpe_rmse", 0)
        time = metrics.get("computational", {}).get("total_time", 0)
        converged = metrics.get("convergence", {}).get("converged", False)
        
        # Color coding for metrics
        ate_class = "metric-good" if ate < 0.05 else "metric-warning" if ate < 0.1 else "metric-bad"
        time_class = "metric-good" if time < 2 else "metric-warning" if time < 5 else "metric-bad"
        conv_symbol = "✓" if converged else "✗"
        conv_class = "metric-good" if converged else "metric-bad"
        
        table_html += f"""
            <tr>
                <td><strong>{alg}</strong></td>
                <td>{run_id}</td>
                <td class="{ate_class}">{ate:.4f}</td>
                <td>{rpe:.4f}</td>
                <td class="{time_class}">{time:.2f}</td>
                <td class="{conv_class}">{conv_symbol}</td>
            </tr>
        """
    
    table_html += """
        </tbody>
    </table>
    """
    
    # Add best performers note
    best_ate_alg = best_ate.get("algorithm", "unknown").upper()
    best_time_alg = best_time.get("algorithm", "unknown").upper()
    
    table_html += f"""
    <p style="margin-top: 20px;">
        <strong>Best Accuracy:</strong> {best_ate_alg} (ATE RMSE: {best_ate.get("metrics", {}).get("trajectory_error", {}).get("ate_rmse", 0):.4f}m)<br>
        <strong>Fastest:</strong> {best_time_alg} ({best_time.get("metrics", {}).get("computational", {}).get("total_time", 0):.2f}s)
    </p>
    """
    
    return table_html