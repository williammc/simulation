#!/usr/bin/env python3
"""
SLAM Simulation System - Command Line Interface
"""

import sys
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
# Add parent directory to path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add tools directory for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate import run_simulation
from slam import run_slam
from dashboard import generate_dashboard
from e2e_pipeline import run_e2e
from evaluate import run_evaluate

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
    enable_preintegration: bool = typer.Option(
        False,
        "--preintegrate",
        help="Enable IMU preintegration between keyframes"
    ),
    keyframe_strategy: str = typer.Option(
        "fixed_interval",
        "--keyframe-strategy", "-ks",
        help="Keyframe selection strategy: fixed_interval, motion_based, hybrid"
    ),
    keyframe_interval: int = typer.Option(
        10,
        "--keyframe-interval", "-ki",
        help="Interval for fixed keyframe selection"
    ),
):
    """Run simulation to generate synthetic SLAM data."""
    from src.common.config import KeyframeSelectionConfig, KeyframeSelectionStrategy
    
    # Create keyframe config
    try:
        strategy = KeyframeSelectionStrategy(keyframe_strategy)
    except ValueError:
        console.print(f"[red]Invalid keyframe strategy: {keyframe_strategy}[/red]")
        raise typer.Exit(1)
    
    keyframe_config = KeyframeSelectionConfig(
        strategy=strategy,
        fixed_interval=keyframe_interval
    )
    
    exit_code = run_simulation(
        trajectory, config, duration, output, seed, noise_config, add_noise,
        enable_preintegration, keyframe_config
    )
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
    result_file = run_slam(estimator, input_data, config, output)
    if result_file is None:
        raise typer.Exit(1)


@app.command()
def evaluate(
    result_file: Path = typer.Argument(
        ...,
        help="Path to SLAM result JSON file"
    ),
    ground_truth: Optional[Path] = typer.Option(
        None,
        "--ground-truth", "-g",
        help="Path to ground truth data (auto-detected if not provided)"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory for evaluation metrics"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed metrics"
    ),
):
    """Evaluate SLAM results against ground truth."""
    metrics_file = run_evaluate(result_file, ground_truth, output, verbose)
    if metrics_file is None:
        raise typer.Exit(1)


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
    include_gtsam: bool = typer.Option(
        True,
        "--gtsam/--no-gtsam",
        help="Include GTSAM comparison tests"
    ),
    cpp: bool = typer.Option(
        True,
        "--cpp/--no-cpp",
        help="Build and test C++ code"
    ),
    cpp_only: bool = typer.Option(
        False,
        "--cpp-only",
        help="Only build and test C++ code, skip Python tests"
    ),
):
    """Run unit tests (Python and optionally C++)."""
    import subprocess
    
    # Track overall success
    all_passed = True
    
    # Build and test C++ if requested
    if cpp or cpp_only:
        console.print("[bold cyan]Building C++ code...[/bold cyan]")
        
        # Check if cpp_estimation exists
        cpp_dir = Path("cpp_estimation")
        if not cpp_dir.exists():
            console.print("[yellow]Warning: cpp_estimation directory not found, skipping C++ tests[/yellow]")
        else:
            # Create build directory
            build_dir = cpp_dir / "build"
            build_dir.mkdir(exist_ok=True)
            
            # Run cmake
            console.print("[green]Running cmake...[/green]")
            cmake_result = subprocess.run(
                ["cmake", ".."],
                cwd=build_dir,
                capture_output=True,
                text=True
            )
            
            if cmake_result.returncode != 0:
                console.print(f"[red]CMake failed:[/red]\n{cmake_result.stderr}")
                all_passed = False
            else:
                # Build
                console.print("[green]Building C++ project...[/green]")
                make_result = subprocess.run(
                    ["make", "-j4"],
                    cwd=build_dir,
                    capture_output=True,
                    text=True
                )
                
                if make_result.returncode != 0:
                    console.print(f"[red]Build failed:[/red]\n{make_result.stderr}")
                    all_passed = False
                else:
                    console.print("[green]✓ C++ build successful[/green]")
                    
                    # Run C++ tests if they exist
                    test_exe = build_dir / "tests" / "test_simulation_io"
                    if test_exe.exists():
                        console.print("[green]Running C++ tests...[/green]")
                        test_result = subprocess.run(
                            [str(test_exe)],
                            cwd=build_dir,
                            capture_output=True,
                            text=True
                        )
                        
                        if test_result.returncode != 0:
                            console.print(f"[red]C++ tests failed:[/red]\n{test_result.stdout}")
                            all_passed = False
                        else:
                            console.print(f"[green]✓ C++ tests passed[/green]\n{test_result.stdout}")
    
    # Run Python tests unless --cpp-only was specified
    if not cpp_only:
        console.print("\n[bold cyan]Running Python tests...[/bold cyan]")
        
        if include_gtsam:
            cmd = [sys.executable, "-m", "pytest", "tests/"]
            console.print("[green]Running all tests (including GTSAM comparisons)...[/green]")
        else:
            cmd = [sys.executable, "-m", "pytest", "tests/", "--ignore=tests/gtsam-comparison"]
            console.print("[green]Running tests (excluding GTSAM comparisons)...[/green]")
        
        if verbose:
            cmd.append("-v")
        if coverage:
            cmd.extend(["--cov=src", "--cov=tools", "--cov-report=term-missing"])
        
        result = subprocess.run(cmd)
        if result.returncode != 0:
            all_passed = False
    
    # Exit with appropriate code
    raise typer.Exit(0 if all_passed else 1)


@app.command()
def test_gtsam(
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Verbose test output"
    ),
    plot: bool = typer.Option(
        True,
        "--plot/--no-plot",
        help="Generate interactive Plotly visualizations"
    ),
):
    """Run GTSAM comparison tests to verify IMU preintegration implementation."""
    import subprocess
    
    console.print("[cyan]Running GTSAM Comparison Tests[/cyan]")
    console.print("This verifies our IMU preintegration matches GTSAM (gold standard)")
    
    cmd = [sys.executable, "-m", "pytest", "tests/gtsam-comparison/"]
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        console.print("[green]✓ All GTSAM comparison tests passed![/green]")
        if plot:
            output_dir = Path("tests/gtsam-comparison/outputs")
            if output_dir.exists():
                console.print(f"\n[yellow]Interactive plots generated in:[/yellow]")
                for html_file in output_dir.glob("*.html"):
                    console.print(f"  • {html_file}")
                console.print(f"\n[cyan]Open {output_dir / 'master_dashboard.html'} for summary[/cyan]")
    else:
        console.print("[red]✗ Some tests failed[/red]")
    
    raise typer.Exit(result.returncode)


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
        # Create a simple object with the expected attributes
        # (The raw SimulationData doesn't have the right attribute names)
        from types import SimpleNamespace
        primary_data = SimpleNamespace(
            ground_truth_trajectory=data_dict.get('trajectory'),
            landmarks=data_dict.get('landmarks'),
            imu_measurements=None,
            camera_measurements=None
        )
        
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
            # Create a simple object with the expected attributes
            from types import SimpleNamespace
            compare_data = SimpleNamespace(
                ground_truth_trajectory=comp_dict.get('trajectory'),
                landmarks=comp_dict.get('landmarks'),
                imu_measurements=None,
                camera_measurements=None
            )
            
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


@app.command(name="e2e")
def e2e_command(
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
    """Run end-to-end (e2e) pipeline across all datasets and estimators."""
    # Call the e2e module
    exit_code = run_e2e(
        config_file=config_file,
        output_dir=output_dir,
        parallel_jobs=parallel_jobs,
        datasets=datasets,
        estimators=estimators,
        skip_generation=skip_generation,
        skip_dashboard=skip_dashboard,
        dry_run=dry_run
    )
    if exit_code != 0:
        raise typer.Exit(exit_code)


@app.command("e2e-simple")
def e2e_simple(
    duration: float = typer.Option(
        10.0,
        "--duration", "-d",
        help="Simulation duration in seconds"
    ),
    trajectory: str = typer.Option(
        "circle",
        "--trajectory", "-t",
        help="Trajectory type: circle, figure8, spiral, line"
    ),
    estimator: str = typer.Option(
        "ekf",
        "--estimator", "-e", 
        help="SLAM estimator: ekf, swba, srif"
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory for all files"
    ),
    sim_file: Optional[str] = typer.Option(
        None,
        "--sim-file",
        help="Custom filename for simulation output"
    ),
    slam_file: Optional[str] = typer.Option(
        None,
        "--slam-file",
        help="Custom filename for SLAM output"
    ),
    eval_file: Optional[str] = typer.Option(
        None,
        "--eval-file",
        help="Custom filename for evaluation output"
    ),
):
    """Run end-to-end simple SLAM pipeline: simulate → estimate → evaluate."""
    from e2e_simple import run_e2e_simple
    
    # Convert Path to string if provided
    output_dir_str = str(output_dir) if output_dir else "output"
    
    # Run the end-to-end pipeline
    run_e2e_simple(
        duration=duration,
        trajectory_type=trajectory,
        estimator_type=estimator,
        output_dir=output_dir_str,
        sim_filename=sim_file,
        slam_filename=slam_file,
        eval_filename=eval_file
    )


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
