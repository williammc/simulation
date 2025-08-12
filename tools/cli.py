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


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()