"""
Evaluate SLAM results against ground truth.
Computes metrics like ATE, RPE, and consistency.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
from rich.console import Console
from rich.table import Table

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
    from src.common.json_io import load_simulation_data
    from src.common.data_structures import Trajectory
    from src.evaluation.metrics import compute_ate, compute_rpe
    
    # Load SLAM results
    if not result_file.exists():
        console.print(f"[red]✗ Error: Result file not found: {result_file}[/red]")
        return None
    
    with open(result_file, 'r') as f:
        slam_result = json.load(f)
    
    # Extract estimated trajectory
    if 'estimated_trajectory' not in slam_result:
        console.print("[red]✗ Error: No estimated trajectory in result file[/red]")
        return None
    
    # Convert trajectory dict to Trajectory object
    estimated_trajectory = Trajectory()
    est_traj_data = slam_result['estimated_trajectory']
    if 'states' in est_traj_data:
        for state_dict in est_traj_data['states']:
            from src.common.data_structures import TrajectoryState, Pose
            import numpy as np
            
            pose_data = state_dict['pose']
            state = TrajectoryState(
                pose=Pose(
                    timestamp=pose_data['timestamp'],
                    position=np.array(pose_data['position']),
                    rotation_matrix=np.array(pose_data['rotation_matrix'])
                ),
                velocity=np.array(state_dict.get('velocity', [0, 0, 0]))
            )
            estimated_trajectory.add_state(state)
    
    # Get ground truth
    ground_truth_trajectory = None
    
    # First check if ground truth is in the SLAM result metadata
    if ground_truth is None and 'metadata' in slam_result:
        input_file = slam_result['metadata'].get('input_file')
        if input_file:
            ground_truth = Path(input_file)
    
    # Load ground truth data
    if ground_truth and ground_truth.exists():
        console.print(f"[cyan]Loading ground truth from: {ground_truth}[/cyan]")
        sim_data = load_simulation_data(str(ground_truth))
        ground_truth_trajectory = sim_data.get('trajectory')
    else:
        console.print("[yellow]Warning: No ground truth data available[/yellow]")
        return None
    
    if not ground_truth_trajectory:
        console.print("[red]✗ Error: Could not load ground truth trajectory[/red]")
        return None
    
    # Compute metrics
    console.print("\n[bold]Computing Evaluation Metrics[/bold]")
    
    # Compute ATE
    ate_errors, ate_metrics = compute_ate(estimated_trajectory, ground_truth_trajectory)
    
    # Compute RPE
    rpe_trans_errors, rpe_rot_errors, rpe_metrics = compute_rpe(
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
        "estimator_metadata": slam_result.get('metadata', {})
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
    
    return metrics_file