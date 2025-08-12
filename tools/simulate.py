"""
Simulation command implementation.
Generates synthetic SLAM data with configurable trajectories and sensors.
"""

import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.simulation.trajectory_generator import generate_trajectory
from src.common.json_io import save_simulation_data

console = Console()


def run_simulation(
    trajectory: str,
    config: Optional[Path],
    duration: float,
    output: Optional[Path],
    seed: Optional[int],
) -> int:
    """
    Run simulation to generate synthetic SLAM data.
    
    Args:
        trajectory: Trajectory type (circle, figure8, spiral, line)
        config: Path to simulation config YAML file
        duration: Simulation duration in seconds
        output: Output directory for simulation data
        seed: Random seed for reproducibility
    
    Returns:
        Exit code (0 for success)
    """
    console.print(f"[bold green]Running simulation[/bold green]")
    console.print(f"  Trajectory: [cyan]{trajectory}[/cyan]")
    console.print(f"  Duration: [cyan]{duration}s[/cyan]")
    
    if config:
        console.print(f"  Config: [cyan]{config}[/cyan]")
    if seed is not None:
        console.print(f"  Seed: [cyan]{seed}[/cyan]")
    
    output_dir = output or Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Load config or use defaults
    if config and config.exists():
        with open(config, 'r') as f:
            params = yaml.safe_load(f)
    else:
        # Default parameters for each trajectory type
        if trajectory == "circle":
            params = {
                "radius": 2.0,
                "height": 1.5,
            }
        elif trajectory == "figure8":
            params = {
                "scale_x": 3.0,
                "scale_y": 2.0,
                "height": 1.5,
            }
        elif trajectory == "spiral":
            params = {
                "initial_radius": 0.5,
                "final_radius": 3.0,
                "initial_height": 0.5,
                "final_height": 3.0,
            }
        elif trajectory == "line":
            params = {
                "start_position": [0, 0, 1],
                "end_position": [10, 0, 1],
            }
        else:
            console.print(f"[red]Error: Unknown trajectory type: {trajectory}[/red]")
            return 1
    
    # Add common parameters
    params["duration"] = duration
    params["rate"] = 100.0  # Hz
    params["start_time"] = 0.0
    
    # Simulate with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating trajectory...", total=None)
        
        try:
            # Generate trajectory
            traj = generate_trajectory(trajectory, params)
            
            progress.update(task, description="Saving data...")
            
            # Generate output file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"simulation_{trajectory}_{timestamp}.json"
            
            # Save to JSON
            metadata = {
                "trajectory_type": trajectory,
                "duration": duration,
                "coordinate_system": "ENU",
                "seed": seed
            }
            
            save_simulation_data(
                filepath=output_file,
                trajectory=traj,
                metadata=metadata
            )
            
        except Exception as e:
            console.print(f"[red]Error during simulation: {e}[/red]")
            return 1
    
    console.print(f"[green]✓[/green] Simulation complete: [cyan]{output_file}[/cyan]")
    
    # Show summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Output size: {output_file.stat().st_size:,} bytes")
    console.print(f"  Trajectory points: {len(traj.states)}")
    time_range = traj.get_time_range()
    console.print(f"  Time range: {time_range[0]:.2f}s - {time_range[1]:.2f}s")
    
    return 0