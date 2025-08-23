"""
Evaluate SLAM results against ground truth.
Computes metrics like ATE, RPE, and consistency.
"""

from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
import json

console = Console()


def run_evaluate(
    result_file: Path,
    ground_truth: Optional[Path] = None,
    output: Optional[Path] = None,
    verbose: bool = False
) -> Optional[Path]:
    """
    Evaluate SLAM results against ground truth.
    
    Args:
        result_file: Path to SLAM result JSON file
        ground_truth: Path to ground truth data (if not in result file)
        output: Output directory for evaluation metrics
        verbose: Show detailed metrics
    
    Returns:
        Path to metrics file if successful, None if error
    """
    from src.estimation.result_io import EstimatorResultStorage
    from src.common.json_io import load_simulation_data
    from src.evaluation.metrics import compute_ate, compute_rpe
    
    # Load SLAM results using EstimatorResultStorage
    if not result_file.exists():
        console.print(f"[red]✗ Error: Result file not found: {result_file}[/red]")
        return None
    
    try:
        result_data = EstimatorResultStorage.load_result(result_file)
    except Exception as e:
        console.print(f"[red]✗ Error loading result file: {e}[/red]")
        return None
    
    # Extract estimated trajectory and landmarks
    estimated_trajectory = result_data.get('trajectory')
    estimated_landmarks = result_data.get('landmarks')
    
    if not estimated_trajectory:
        console.print("[red]✗ Error: No estimated trajectory in result file[/red]")
        return None
    
    # Get ground truth
    ground_truth_trajectory = None
    
    # First check if ground truth path is in the simulation metadata
    if ground_truth is None:
        sim_metadata = result_data.get('simulation', {})
        input_file = sim_metadata.get('input_file')
        if input_file:
            ground_truth = Path(input_file)
    
    # Load ground truth data
    if ground_truth and ground_truth.exists():
        console.print(f"[cyan]Loading ground truth from: {ground_truth}[/cyan]")
        sim_data = load_simulation_data(str(ground_truth))
        
        # Extract trajectory based on data format
        if isinstance(sim_data, dict):
            ground_truth_trajectory = sim_data.get('trajectory')
        else:
            ground_truth_trajectory = getattr(sim_data, 'ground_truth_trajectory', None)
    else:
        console.print("[yellow]Warning: No ground truth data available[/yellow]")
        console.print("[yellow]Cannot compute trajectory error metrics without ground truth[/yellow]")
        return None
    
    if not ground_truth_trajectory:
        console.print("[red]✗ Error: Could not load ground truth trajectory[/red]")
        return None
    
    # Compute metrics
    console.print("\n[bold]Computing Evaluation Metrics[/bold]")
    
    # Compute ATE
    _, ate_metrics = compute_ate(estimated_trajectory, ground_truth_trajectory)
    
    # Compute RPE
    _, _, rpe_metrics = compute_rpe(
        estimated_trajectory, ground_truth_trajectory, delta=1
    )
    
    # Create metrics dictionary
    metrics = {
        "ate": {
            "rmse": float(ate_metrics.ate_rmse),
            "mean": float(ate_metrics.ate_mean),
            "median": float(ate_metrics.ate_median),
            "std": float(ate_metrics.ate_std),
            "min": float(ate_metrics.ate_min),
            "max": float(ate_metrics.ate_max)
        },
        "rpe_translation": {
            "rmse": float(rpe_metrics.rpe_trans_rmse),
            "mean": float(rpe_metrics.rpe_trans_mean),
            "median": float(rpe_metrics.rpe_trans_median),
            "std": float(rpe_metrics.rpe_trans_std)
        },
        "rpe_rotation": {
            "rmse": float(rpe_metrics.rpe_rot_rmse),
            "mean": float(rpe_metrics.rpe_rot_mean),
            "median": float(rpe_metrics.rpe_rot_median),
            "std": float(rpe_metrics.rpe_rot_std)
        },
        "trajectory": {
            "estimated_states": len(estimated_trajectory.states),
            "ground_truth_states": len(ground_truth_trajectory.states),
            "trajectory_length": float(ate_metrics.trajectory_length)
        }
    }
    
    # Add landmark metrics if available
    if estimated_landmarks:
        true_landmark_count = 0
        if isinstance(sim_data, dict):
            landmarks_data = sim_data.get('landmarks', {})
            if isinstance(landmarks_data, dict):
                true_landmark_count = len(landmarks_data.get('landmarks', []))
            elif isinstance(landmarks_data, list):
                true_landmark_count = len(landmarks_data)
        
        metrics["landmarks"] = {
            "estimated_count": len(estimated_landmarks.landmarks),
            "true_count": true_landmark_count
        }
    
    # Display results
    table = Table(title="Evaluation Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # ATE metrics
    table.add_row("ATE RMSE", f"{metrics['ate']['rmse']:.4f} m")
    table.add_row("ATE Mean", f"{metrics['ate']['mean']:.4f} m")
    table.add_row("ATE Std", f"{metrics['ate']['std']:.4f} m")
    
    # RPE metrics
    table.add_row("RPE Trans RMSE", f"{metrics['rpe_translation']['rmse']:.4f} m")
    table.add_row("RPE Rot RMSE", f"{metrics['rpe_rotation']['rmse']:.4f} rad")
    
    # Trajectory info
    table.add_row("Estimated States", str(metrics['trajectory']['estimated_states']))
    table.add_row("Ground Truth States", str(metrics['trajectory']['ground_truth_states']))
    
    # Landmark info if available
    if 'landmarks' in metrics:
        table.add_row("Estimated Landmarks", str(metrics['landmarks']['estimated_count']))
        table.add_row("Ground Truth Landmarks", str(metrics['landmarks']['true_count']))
    
    console.print("\n")
    console.print(table)
    
    # Save metrics
    output_dir = output or Path("output/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename based on result file
    result_name = result_file.stem
    metrics_file = output_dir / f"metrics_{result_name}.json"
    
    # Create full evaluation output
    evaluation_output = {
        "result_file": str(result_file),
        "ground_truth_file": str(ground_truth) if ground_truth else None,
        "metrics": metrics,
        "estimator_info": {
            "algorithm": result_data.get("algorithm"),
            "run_id": result_data.get("run_id"),
            "timestamp": result_data.get("timestamp"),
            "configuration": result_data.get("configuration", {})
        },
        "simulation_metadata": result_data.get("simulation", {})
    }
    
    # Add computational metrics from the result
    if "results" in result_data:
        evaluation_output["computational_metrics"] = {
            "runtime_ms": result_data["results"].get("runtime_ms"),
            "iterations": result_data["results"].get("iterations"),
            "converged": result_data["results"].get("converged"),
            "final_cost": result_data["results"].get("final_cost")
        }
    
    with open(metrics_file, 'w') as f:
        json.dump(evaluation_output, f, indent=2)
    
    console.print(f"\n[green]✓ Metrics saved to:[/green] {metrics_file}")
    
    # Verbose output
    if verbose:
        console.print("\n[bold]Detailed Metrics:[/bold]")
        console.print(f"  ATE: min={metrics['ate']['min']:.4f}, max={metrics['ate']['max']:.4f}")
        console.print(f"  RPE Trans: median={metrics['rpe_translation']['median']:.4f}")
        console.print(f"  RPE Rot: median={metrics['rpe_rotation']['median']:.4f}")
        
        if "computational_metrics" in evaluation_output:
            comp = evaluation_output["computational_metrics"]
            if comp.get("runtime_ms"):
                console.print(f"  Runtime: {comp['runtime_ms']/1000:.2f} seconds")
            if comp.get("iterations"):
                console.print(f"  Iterations: {comp['iterations']}")
            if comp.get("converged") is not None:
                console.print(f"  Converged: {comp['converged']}")
    
    # Use EstimatorResultStorage to create KPI summary
    try:
        kpi_summary = EstimatorResultStorage.create_kpi_summary(result_file, ground_truth)
        if verbose:
            console.print("\n[bold]KPI Summary:[/bold]")
            console.print(f"  Algorithm: {kpi_summary.get('algorithm')}")
            console.print(f"  Run ID: {kpi_summary.get('run_id')}")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not create KPI summary: {e}[/yellow]")
    
    return metrics_file