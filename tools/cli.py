#!/usr/bin/env python3
"""
SLAM Simulation System - Command Line Interface
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

# Import command implementations
from tools.simulate import run_simulation
from tools.slam import run_slam
from tools.dashboard import generate_dashboard
from tools.download import download_dataset, list_available_datasets

app = typer.Typer(
    name="slam-sim",
    help="SLAM Simulation System CLI",
    add_completion=False,
)
console = Console()


@app.command()
def simulate(
    trajectory: str = typer.Argument(
        "circle",
        help="Trajectory type: circle, figure8, spiral, line, random_walk"
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to simulation config YAML file"
    ),
    duration: float = typer.Option(
        20.0,
        "--duration", "-d",
        help="Simulation duration in seconds"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory for simulation data"
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed", "-s",
        help="Random seed for reproducibility"
    ),
    noise_config: Optional[Path] = typer.Option(
        None,
        "--noise-config", "-n",
        help="Path to noise configuration YAML file"
    ),
    add_noise: bool = typer.Option(
        False,
        "--add-noise",
        help="Add noise to sensor measurements"
    ),
):
    """Run simulation to generate synthetic SLAM data."""
    exit_code = run_simulation(trajectory, config, duration, output, seed, noise_config, add_noise)
    if exit_code != 0:
        raise typer.Exit(exit_code)


@app.command()
def slam(
    estimator: str = typer.Argument(
        "ekf",
        help="Estimator type: ekf, swba, srif"
    ),
    input_data: Path = typer.Option(
        ...,
        "--input", "-i",
        help="Path to simulation data JSON file"
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to estimator config YAML file"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory for SLAM results"
    ),
):
    """Run SLAM estimator on simulation data."""
    exit_code = run_slam(estimator, input_data, config, output)
    if exit_code != 0:
        raise typer.Exit(exit_code)


@app.command()
def dashboard(
    input_dir: Path = typer.Option(
        Path("data/SLAM"),
        "--input", "-i",
        help="Directory containing SLAM KPI JSON files"
    ),
    output: Path = typer.Option(
        Path("output/dashboard.html"),
        "--output", "-o",
        help="Output HTML file path"
    ),
):
    """Generate dashboard from SLAM KPIs."""
    exit_code = generate_dashboard(input_dir, output)
    if exit_code != 0:
        raise typer.Exit(exit_code)


@app.command()
def download(
    dataset: str = typer.Argument(
        ...,
        help="Dataset name (e.g., 'tum-vie', 'euroc')"
    ),
    sequence: str = typer.Argument(
        ...,
        help="Sequence name (e.g., 'mocap-desk', 'mh-01')"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory for downloaded data"
    ),
):
    """Download public datasets."""
    exit_code = download_dataset(dataset, sequence, output)
    if exit_code != 0:
        raise typer.Exit(exit_code)


@app.command("list-runs")
def list_runs(
    directory: Path = typer.Option(
        Path("data/SLAM"),
        "--dir", "-d",
        help="Directory containing SLAM run results"
    ),
):
    """List all SLAM estimation runs."""
    console.print("[green]SLAM Estimation Runs:[/green]")
    
    if not directory.exists():
        console.print(f"[yellow]No runs found in {directory}[/yellow]")
        return
    
    # Create table
    table = Table(title="SLAM Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Algorithm", style="magenta")
    table.add_column("Timestamp", style="green")
    table.add_column("ATE RMSE", style="yellow")
    
    # Load and display runs
    import json
    for json_file in sorted(directory.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
                table.add_row(
                    data.get("run_id", "N/A"),
                    data.get("algorithm", "N/A"),
                    data.get("timestamp", "N/A"),
                    f"{data.get('metrics', {}).get('trajectory_error', {}).get('ate_rmse', 0):.4f}"
                )
        except Exception as e:
            console.print(f"[red]Error reading {json_file}: {e}[/red]")
    
    console.print(table)


@app.command("list-datasets")
def list_datasets():
    """List available datasets for download."""
    list_available_datasets()


@app.command()
def test(
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Verbose test output"
    ),
    coverage: bool = typer.Option(
        False,
        "--coverage",
        help="Run with coverage report"
    ),
):
    """Run unit tests."""
    import subprocess
    
    cmd = [sys.executable, "-m", "pytest", "tests/"]
    if verbose:
        cmd.append("-v")
    if coverage:
        cmd.extend(["--cov=src", "--cov=tools", "--cov-report=term-missing"])
    
    console.print("[green]Running tests...[/green]")
    result = subprocess.run(cmd)
    raise typer.Exit(result.returncode)


@app.command()
def compare(
    simulation_data: Path = typer.Argument(
        ...,
        help="Path to simulation data JSON file"
    ),
    estimators: Optional[str] = typer.Option(
        None,
        "--estimators", "-e",
        help="Comma-separated list of estimators (default: all)"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory for comparison results"
    ),
    generate_report: bool = typer.Option(
        True,
        "--report/--no-report",
        help="Generate HTML report with plots"
    ),
    show_table: bool = typer.Option(
        True,
        "--table/--no-table",
        help="Display comparison table in terminal"
    )
):
    """Compare multiple SLAM estimators on the same data."""
    from src.evaluation.comparison import compare_estimators, generate_comparison_table
    from src.visualization.comparison_plots import generate_html_report
    
    # Parse estimators
    estimator_list = None
    if estimators:
        estimator_list = [e.strip().upper() for e in estimators.split(",")]
    
    # Set output directory
    if output_dir is None:
        output_dir = Path("output/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[green]Comparing estimators on {simulation_data}[/green]")
    
    try:
        # Run comparison
        results = compare_estimators(
            str(simulation_data),
            estimators=estimator_list,
            output_path=str(output_dir / "comparison_results.json")
        )
        
        # Display table if requested
        if show_table:
            console.print("\n[bold]Comparison Results:[/bold]")
            table_str = generate_comparison_table(results)
            console.print(table_str)
        
        # Generate report if requested
        if generate_report:
            console.print("\n[green]Generating HTML report...[/green]")
            report_path = generate_html_report(
                results,
                str(output_dir),
                include_plots=True
            )
            console.print(f"[green]✓ Report saved to: {report_path}[/green]")
        
        # Print summary
        console.print(f"\n[bold]Best Estimator:[/bold] {results.best_estimator}")
        console.print(f"[bold]Results saved to:[/bold] {output_dir}")
        
    except Exception as e:
        console.print(f"[red]Error during comparison: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def plot(
    input_file: Path = typer.Argument(..., help="Input JSON file (simulation or SLAM result)"),
    compare_file: Optional[Path] = typer.Option(None, "--compare", help="Second file for comparison"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output HTML file"),
    show_trajectory: bool = typer.Option(True, "--trajectory/--no-trajectory", help="Show 3D trajectory"),
    show_measurements: bool = typer.Option(True, "--measurements/--no-measurements", help="Show 2D measurements"),
    show_imu: bool = typer.Option(True, "--imu/--no-imu", help="Show IMU data"),
    keyframes: Optional[int] = typer.Option(None, "--keyframes", "-k", help="Number of keyframes to show"),
    open_browser: bool = typer.Option(True, "--browser/--no-browser", help="Open in browser")
):
    """Generate interactive plots from simulation or SLAM results."""
    from src.plotting.enhanced_plots import create_full_visualization
    from src.common.json_io import load_simulation_data, SimulationData
    
    console.print("[bold cyan]Generating plots...[/bold cyan]")
    
    # Load primary data
    try:
        data_dict = load_simulation_data(str(input_file))
        # Always create a simple object with the expected attributes
        # (The raw SimulationData doesn't have the right attribute names)
        class SimData:
            pass
        primary_data = SimData()
        # Map the loaded data to expected attributes
        primary_data.ground_truth_trajectory = data_dict.get('trajectory')
        primary_data.landmarks = data_dict.get('landmarks')
        primary_data.imu_measurements = None
        primary_data.camera_measurements = None
        
        # Handle IMU data
        if data_dict.get('imu_data'):
            imu_data = data_dict['imu_data']
            if hasattr(imu_data, 'measurements'):
                primary_data.imu_measurements = imu_data.measurements
        
        # Handle camera data
        if data_dict.get('camera_data'):
            cam_data = data_dict['camera_data']
            if hasattr(cam_data, 'frames'):
                primary_data.camera_measurements = cam_data.frames
        
        primary_data.camera_calibrations = data_dict.get('camera_calibrations')
        primary_data.metadata = data_dict.get('metadata', {})
        console.print(f"✓ Loaded: {input_file.name}")
    except Exception as e:
        console.print(f"[red]Error loading {input_file}: {e}[/red]")
        raise typer.Exit(1)
    
    # Load comparison data if provided
    compare_data = None
    if compare_file:
        try:
            comp_dict = load_simulation_data(str(compare_file))
            # Always create a simple object with the expected attributes
            class SimData:
                pass
            compare_data = SimData()
            compare_data.ground_truth_trajectory = comp_dict.get('trajectory')
            compare_data.landmarks = comp_dict.get('landmarks')
            compare_data.imu_measurements = None
            compare_data.camera_measurements = None
            
            if comp_dict.get('imu_data'):
                imu_data = comp_dict['imu_data']
                if hasattr(imu_data, 'measurements'):
                    compare_data.imu_measurements = imu_data.measurements
            
            if comp_dict.get('camera_data'):
                cam_data = comp_dict['camera_data']
                if hasattr(cam_data, 'frames'):
                    compare_data.camera_measurements = cam_data.frames
            
            compare_data.camera_calibrations = comp_dict.get('camera_calibrations')
            compare_data.metadata = comp_dict.get('metadata', {})
            console.print(f"✓ Loaded comparison: {compare_file.name}")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load {compare_file}: {e}[/yellow]")
    
    # Generate output filename
    if output is None:
        output = Path("output") / f"{input_file.stem}_plots.html"
        output.parent.mkdir(exist_ok=True)
    
    # Create visualization
    try:
        html_content = create_full_visualization(
            primary_data,
            compare_data=compare_data,
            show_trajectory=show_trajectory,
            show_measurements=show_measurements,
            show_imu=show_imu,
            max_keyframes=keyframes
        )
        
        # Save HTML
        with open(output, 'w') as f:
            f.write(html_content)
        
        console.print(f"[green]✓ Plots saved to: {output}[/green]")
        
        # Open in browser
        if open_browser:
            import webbrowser
            webbrowser.open(f"file://{output.absolute()}")
            console.print("[cyan]Opening in browser...[/cyan]")
        
    except Exception as e:
        console.print(f"[red]Error creating plots: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def clean():
    """Clean generated files and caches."""
    import shutil
    from rich.prompt import Confirm
    
    console.print("[yellow]This will remove:[/yellow]")
    console.print("  • Output files (output/*.json, output/*.html)")
    console.print("  • SLAM results (data/SLAM/*.json)")
    console.print("  • Python caches (__pycache__)")
    console.print("  • Test caches (.pytest_cache)")
    
    if not Confirm.ask("\nContinue?"):
        console.print("[yellow]Cancelled[/yellow]")
        raise typer.Exit(0)
    
    # Clean output files
    for pattern in ["output/*.json", "output/*.html", "data/SLAM/*.json"]:
        for file in Path(".").glob(pattern):
            file.unlink()
            console.print(f"  Removed: {file}")
    
    # Clean cache directories
    for cache_dir in Path(".").rglob("__pycache__"):
        shutil.rmtree(cache_dir)
        console.print(f"  Removed: {cache_dir}")
    
    if Path(".pytest_cache").exists():
        shutil.rmtree(".pytest_cache")
        console.print("  Removed: .pytest_cache")
    
    console.print("[green]✓ Cleaned![/green]")


@app.command()
def info():
    """Show system information and configuration."""
    
    console.print("[bold cyan]SLAM Simulation System[/bold cyan]")
    console.print()
    
    # System info
    table = Table(title="System Information")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    table.add_row("Python", f"{sys.version.split()[0]}")
    table.add_row("Project Root", str(Path.cwd()))
    table.add_row("Config Directory", "config/")
    table.add_row("Output Directory", "output/")
    table.add_row("Data Directory", "data/")
    
    console.print(table)
    
    # Available trajectories
    console.print("\n[bold]Available Trajectories:[/bold]")
    trajectories = ["circle", "figure8", "spiral", "line", "random_walk"]
    for traj in trajectories:
        console.print(f"  • {traj}")
    
    # Available estimators
    console.print("\n[bold]Available Estimators:[/bold]")
    estimators = ["ekf", "swba", "srif"]
    for est in estimators:
        console.print(f"  • {est.upper()}")
    
    # Available datasets
    console.print("\n[bold]Available Datasets:[/bold]")
    console.print("  Run './run.sh list-datasets' for details")


@app.command()
def evaluation(
    config_file: Path = typer.Argument(
        Path("config/evaluation_config.yaml"),
        help="Path to evaluation configuration YAML file"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Override output directory from config"
    ),
    parallel_jobs: Optional[int] = typer.Option(
        None,
        "--parallel", "-j",
        help="Number of parallel estimation jobs"
    ),
    datasets: Optional[str] = typer.Option(
        None,
        "--datasets", "-d",
        help="Comma-separated list of datasets to evaluate (default: all)"
    ),
    estimators: Optional[str] = typer.Option(
        None,
        "--estimators", "-e",
        help="Comma-separated list of estimators to run (default: all enabled)"
    ),
    skip_generation: bool = typer.Option(
        False,
        "--skip-generation",
        help="Skip dataset generation even if missing"
    ),
    skip_dashboard: bool = typer.Option(
        False,
        "--skip-dashboard",
        help="Skip dashboard generation"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be done without actually running"
    )
):
    """Run comprehensive evaluation pipeline across all datasets and estimators."""
    from src.evaluation.orchestrator import EvaluationOrchestrator
    import yaml
    
    # Check if config file exists
    if not config_file.exists():
        console.print(f"[red]Error: Configuration file not found: {config_file}[/red]")
        console.print("\n[yellow]Tip: Use the default config at config/evaluation_config.yaml[/yellow]")
        raise typer.Exit(1)
    
    # Load and potentially modify config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override settings from command line
    if output_dir:
        config['evaluation']['output_dir'] = str(output_dir)
    if parallel_jobs:
        config['evaluation']['parallel_jobs'] = parallel_jobs
    if skip_generation:
        config['datasets']['simulated']['generate_if_missing'] = False
        config['datasets']['tum_vi']['download_if_missing'] = False
    
    # Filter datasets if specified
    if datasets:
        dataset_list = [d.strip() for d in datasets.split(",")]
        # Filter simulated datasets
        if 'simulated' in config['datasets']:
            config['datasets']['simulated']['types'] = [
                d for d in config['datasets']['simulated']['types']
                if d['name'] in dataset_list
            ]
        # Filter TUM-VI sequences
        if 'tum_vi' in config['datasets']:
            config['datasets']['tum_vi']['sequences'] = [
                s for s in config['datasets']['tum_vi']['sequences']
                if s['name'] in dataset_list
            ]
    
    # Filter estimators if specified
    if estimators:
        estimator_list = [e.strip().lower() for e in estimators.split(",")]
        for est_name in config['estimators'].keys():
            if est_name not in estimator_list:
                config['estimators'][est_name]['enabled'] = False
    
    # Disable dashboard if requested
    if skip_dashboard:
        config['dashboard']['sections'] = []
    
    # Show configuration summary
    console.print("\n[bold]Evaluation Configuration:[/bold]")
    console.print(f"  Output Directory: {config['evaluation']['output_dir']}")
    console.print(f"  Parallel Jobs: {config['evaluation']['parallel_jobs']}")
    
    # Count enabled items
    num_sim = len(config['datasets'].get('simulated', {}).get('types', []))
    num_tum = len(config['datasets'].get('tum_vi', {}).get('sequences', []))
    num_datasets = num_sim + num_tum
    
    enabled_estimators = [
        name for name, cfg in config['estimators'].items()
        if cfg.get('enabled', True)
    ]
    
    console.print(f"  Datasets: {num_datasets} ({num_sim} simulated, {num_tum} TUM-VI)")
    console.print(f"  Estimators: {len(enabled_estimators)} ({', '.join(enabled_estimators)})")
    console.print(f"  Total Runs: {num_datasets * len(enabled_estimators)}")
    
    if dry_run:
        console.print("\n[yellow]Dry run mode - no actual execution[/yellow]")
        console.print("\n[bold]Would evaluate:[/bold]")
        
        # Show datasets
        console.print("\n[cyan]Simulated Datasets:[/cyan]")
        for dataset in config['datasets'].get('simulated', {}).get('types', []):
            console.print(f"  - {dataset['name']}")
        
        console.print("\n[cyan]TUM-VI Sequences:[/cyan]")
        for seq in config['datasets'].get('tum_vi', {}).get('sequences', []):
            console.print(f"  - {seq['name']}")
        
        console.print("\n[cyan]Estimators:[/cyan]")
        for est in enabled_estimators:
            console.print(f"  - {est}")
        
        return
    
    # Run evaluation
    try:
        console.print("\n[green]Starting evaluation pipeline...[/green]")
        
        # Save modified config to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_config_path = f.name
        
        try:
            # Create orchestrator with modified config
            orchestrator = EvaluationOrchestrator(temp_config_path)
            
            # Run pipeline
            results = orchestrator.run()
            
            console.print("\n[green]✓ Evaluation complete![/green]")
            
            if 'dashboard' in results:
                console.print(f"\n[bold]Dashboard:[/bold] {results['dashboard']}")
                
                # Optionally open in browser
                import webbrowser
                if not skip_dashboard:
                    webbrowser.open(f"file://{Path(results['dashboard']).absolute()}")
            
            # Show KPI summary
            if 'kpis' in results and 'summary' in results['kpis']:
                console.print("\n[bold]Top KPIs:[/bold]")
                kpi_summary = results['kpis']['summary']
                
                # Find best performer for each KPI
                for kpi_name, estimator_stats in kpi_summary.items():
                    if estimator_stats:
                        best_est = min(estimator_stats.items(), key=lambda x: x[1]['mean'])
                        console.print(f"  {kpi_name}: {best_est[0]} ({best_est[1]['mean']:.3f})")
        finally:
            # Clean up temp config file
            import os
            if 'temp_config_path' in locals():
                try:
                    os.unlink(temp_config_path)
                except:
                    pass
        
    except Exception as e:
        console.print(f"\n[red]Error during evaluation: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def convert(
    dataset_type: str = typer.Argument(
        ...,
        help="Dataset type to convert (tumvie, euroc, kitti)"
    ),
    input_path: Path = typer.Argument(
        ...,
        help="Path to input dataset directory"
    ),
    output_file: Path = typer.Argument(
        ...,
        help="Path to output JSON file"
    ),
    num_landmarks: int = typer.Option(
        200,
        "--num-landmarks", "-n",
        help="Number of synthetic 3D landmarks to generate"
    ),
    keyframe_interval: float = typer.Option(
        0.1,
        "--keyframe-interval", "-k",
        help="Time between keyframes in seconds"
    ),
    pixel_noise: float = typer.Option(
        1.0,
        "--pixel-noise", "-p",
        help="Standard deviation of pixel measurement noise"
    )
):
    """Convert external datasets to common JSON format."""
    
    # Check dataset type
    supported_types = ["tumvie", "euroc", "kitti"]
    if dataset_type.lower() not in supported_types:
        console.print(f"[red]Error: Unsupported dataset type: {dataset_type}[/red]")
        console.print(f"[yellow]Supported types: {', '.join(supported_types)}[/yellow]")
        raise typer.Exit(1)
    
    # Check input path exists
    if not input_path.exists():
        console.print(f"[red]Error: Input path not found: {input_path}[/red]")
        raise typer.Exit(1)
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if dataset_type.lower() == "tumvie":
            console.print(f"[cyan]Converting TUM-VIE dataset:[/cyan]")
            console.print(f"  Input: {input_path}")
            console.print(f"  Output: {output_file}")
            console.print(f"  Landmarks: {num_landmarks}")
            console.print(f"  Keyframe interval: {keyframe_interval}s")
            console.print(f"  Pixel noise std: {pixel_noise} pixels")
            
            from src.utils.tumvie_converter import convert_tumvie_dataset
            
            # Show progress
            with console.status("[bold green]Converting dataset...") as status:
                convert_tumvie_dataset(
                    input_path, 
                    output_file,
                    num_landmarks=num_landmarks,
                    keyframe_interval=keyframe_interval,
                    pixel_noise_std=pixel_noise
                )
            
            console.print(f"[green]✓ Dataset converted successfully![/green]")
            
            # Show summary
            import json
            with open(output_file, 'r') as f:
                data = json.load(f)
                
            if 'metadata' in data:
                console.print("\n[bold]Dataset Summary:[/bold]")
                meta = data['metadata']
                console.print(f"  Source: {meta.get('source', 'Unknown')}")
                console.print(f"  Duration: {meta.get('duration', 0):.1f}s")
                console.print(f"  Poses: {meta.get('num_poses', 0)}")
                console.print(f"  Landmarks: {meta.get('num_landmarks', 0)}")
                console.print(f"  IMU Measurements: {meta.get('num_imu_measurements', 0)}")
                console.print(f"  Camera Frames: {meta.get('num_camera_frames', 0)}")
        
        elif dataset_type.lower() == "euroc":
            console.print(f"[yellow]EuRoC conversion not yet implemented[/yellow]")
            raise typer.Exit(1)
            
        elif dataset_type.lower() == "kitti":
            console.print(f"[yellow]KITTI conversion not yet implemented[/yellow]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error during conversion: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()