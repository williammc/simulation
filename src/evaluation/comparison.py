"""
Comparison tools for evaluating multiple SLAM estimators.
"""

import time
import tracemalloc
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import pandas as pd
from scipy import stats

from src.estimation.ekf_slam import EKFSlam
from src.estimation.swba_slam import SlidingWindowBA
from src.estimation.srif_slam import SRIFSlam
from src.common.config import EKFConfig, SWBAConfig, SRIFConfig
from src.estimation.base_estimator import (
    BaseEstimator, EstimatorType, EstimatorResult
)
from src.common.data_structures import (
    Trajectory, Map, IMUMeasurement, CameraFrame,
    CameraCalibration, IMUCalibration
)
from src.evaluation.metrics import (
    compute_ate, compute_rpe, compute_nees,
    TrajectoryMetrics, ConsistencyMetrics
)
from src.common.json_io import SimulationData, load_simulation_data


@dataclass
class EstimatorPerformance:
    """
    Performance metrics for an estimator run.
    
    Attributes:
        estimator_type: Type of estimator
        runtime_ms: Total runtime in milliseconds
        peak_memory_mb: Peak memory usage in megabytes
        trajectory_metrics: Trajectory error metrics (ATE, RPE)
        consistency_metrics: Consistency metrics (NEES)
        num_iterations: Number of iterations (for optimization methods)
        converged: Whether the estimator converged
        metadata: Additional estimator-specific metadata
    """
    estimator_type: EstimatorType
    runtime_ms: float
    peak_memory_mb: float
    trajectory_metrics: TrajectoryMetrics
    consistency_metrics: Optional[ConsistencyMetrics] = None
    num_iterations: int = 0
    converged: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "estimator_type": self.estimator_type.value,
            "runtime_ms": self.runtime_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "trajectory_metrics": self.trajectory_metrics.to_dict(),
            "consistency_metrics": self.consistency_metrics.to_dict() if self.consistency_metrics else None,
            "num_iterations": self.num_iterations,
            "converged": self.converged,
            "metadata": self.metadata
        }


@dataclass
class ComparisonResult:
    """
    Results from comparing multiple estimators.
    
    Attributes:
        performances: Performance results for each estimator
        statistical_tests: Results of statistical significance tests
        best_estimator: Best performing estimator by ATE
        simulation_metadata: Information about the simulation data
    """
    performances: Dict[str, EstimatorPerformance]
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    best_estimator: Optional[str] = None
    simulation_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "performances": {k: v.to_dict() for k, v in self.performances.items()},
            "statistical_tests": self.statistical_tests,
            "best_estimator": self.best_estimator,
            "simulation_metadata": self.simulation_metadata
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for easy analysis."""
        data = []
        for name, perf in self.performances.items():
            row = {
                "estimator": name,
                "runtime_ms": perf.runtime_ms,
                "memory_mb": perf.peak_memory_mb,
                "ate_rmse": perf.trajectory_metrics.ate_rmse,
                "ate_mean": perf.trajectory_metrics.ate_mean,
                "rpe_trans_rmse": perf.trajectory_metrics.rpe_trans_rmse,
                "rpe_rot_rmse": perf.trajectory_metrics.rpe_rot_rmse,
                "converged": perf.converged
            }
            if perf.consistency_metrics:
                row["nees_mean"] = perf.consistency_metrics.nees_mean
                row["is_consistent"] = perf.consistency_metrics.is_consistent
            data.append(row)
        
        return pd.DataFrame(data)


class EstimatorRunner:
    """
    Runner for evaluating SLAM estimators on simulation data.
    """
    
    def __init__(
        self,
        camera_calibration: CameraCalibration,
        imu_calibration: Optional[IMUCalibration] = None,
        enable_profiling: bool = True
    ):
        """
        Initialize estimator runner.
        
        Args:
            camera_calibration: Camera calibration
            imu_calibration: Optional IMU calibration
            enable_profiling: Whether to profile runtime and memory
        """
        self.camera_calib = camera_calibration
        self.imu_calib = imu_calibration
        self.enable_profiling = enable_profiling
        
        # Available estimators
        self.estimator_configs = {
            "EKF": EKFConfig(),
            "SWBA": SWBAConfig(),
            "SRIF": SRIFConfig()
        }
    
    def run_estimator(
        self,
        estimator_type: str,
        imu_measurements: List[List[IMUMeasurement]],
        camera_frames: List[CameraFrame],
        ground_truth: Trajectory,
        landmarks: Map,
        initial_pose: 'Pose'
    ) -> EstimatorPerformance:
        """
        Run a single estimator and evaluate its performance.
        
        Args:
            estimator_type: Type of estimator ("EKF", "SWBA", "SRIF")
            imu_measurements: IMU measurements grouped by prediction step
            camera_frames: Camera measurements
            ground_truth: Ground truth trajectory
            landmarks: Known landmarks
            initial_pose: Initial pose
        
        Returns:
            Performance metrics for the estimator
        """
        # Create estimator
        estimator = self._create_estimator(estimator_type)
        
        # Start profiling
        if self.enable_profiling:
            tracemalloc.start()
            start_time = time.perf_counter()
        
        # Initialize estimator
        estimator.initialize(initial_pose)
        
        # Process measurements
        states = []
        cam_idx = 0
        
        for imu_batch in imu_measurements:
            if imu_batch:
                # IMU prediction
                dt = imu_batch[-1].timestamp - imu_batch[0].timestamp
                if dt > 0:
                    estimator.predict(imu_batch, dt)
            
            # Check for camera update
            while cam_idx < len(camera_frames):
                frame = camera_frames[cam_idx]
                if imu_batch and frame.timestamp <= imu_batch[-1].timestamp:
                    estimator.update(frame, landmarks)
                    states.append(estimator.get_state())
                    cam_idx += 1
                else:
                    break
        
        # Get final result
        result = estimator.get_result()
        
        # Stop profiling
        if self.enable_profiling:
            end_time = time.perf_counter()
            runtime_ms = (end_time - start_time) * 1000
            
            current, peak = tracemalloc.get_traced_memory()
            peak_memory_mb = peak / 1024 / 1024
            tracemalloc.stop()
        else:
            runtime_ms = 0
            peak_memory_mb = 0
        
        # Compute metrics
        _, traj_metrics = compute_ate(result.trajectory, ground_truth, align=True)
        
        # Add RPE metrics
        _, _, rpe_metrics = compute_rpe(result.trajectory, ground_truth, delta=1)
        traj_metrics.rpe_trans_rmse = rpe_metrics.rpe_trans_rmse
        traj_metrics.rpe_trans_mean = rpe_metrics.rpe_trans_mean
        traj_metrics.rpe_rot_rmse = rpe_metrics.rpe_rot_rmse
        traj_metrics.rpe_rot_mean = rpe_metrics.rpe_rot_mean
        
        # Compute consistency metrics if covariance available
        consistency_metrics = None
        if states and states[0].covariance_matrix is not None:
            _, consistency_metrics = compute_nees(states, ground_truth)
        
        # Build performance result
        return EstimatorPerformance(
            estimator_type=EstimatorType[estimator_type],
            runtime_ms=runtime_ms,
            peak_memory_mb=peak_memory_mb,
            trajectory_metrics=traj_metrics,
            consistency_metrics=consistency_metrics,
            num_iterations=result.iterations if hasattr(result, 'iterations') else 0,
            converged=result.converged if hasattr(result, 'converged') else True,
            metadata=result.metadata if hasattr(result, 'metadata') else {}
        )
    
    def run_all_estimators(
        self,
        simulation_data: SimulationData,
        estimators: Optional[List[str]] = None
    ) -> ComparisonResult:
        """
        Run all estimators on the same simulation data.
        
        Args:
            simulation_data: Simulation data to process
            estimators: List of estimator names to run (default: all)
        
        Returns:
            Comparison results
        """
        if estimators is None:
            estimators = list(self.estimator_configs.keys())
        
        # Prepare data
        imu_batches = self._prepare_imu_batches(simulation_data.imu_measurements)
        initial_pose = simulation_data.ground_truth_trajectory.states[0].pose
        
        # Run each estimator
        performances = {}
        for est_name in estimators:
            print(f"Running {est_name}...")
            try:
                perf = self.run_estimator(
                    est_name,
                    imu_batches,
                    simulation_data.camera_measurements,
                    simulation_data.ground_truth_trajectory,
                    simulation_data.landmarks,
                    initial_pose
                )
                performances[est_name] = perf
            except Exception as e:
                print(f"Error running {est_name}: {e}")
                continue
        
        # Determine best estimator
        best_estimator = None
        best_ate = float('inf')
        for name, perf in performances.items():
            if perf.trajectory_metrics.ate_rmse < best_ate:
                best_ate = perf.trajectory_metrics.ate_rmse
                best_estimator = name
        
        # Statistical tests
        statistical_tests = self._perform_statistical_tests(performances)
        
        # Build comparison result
        return ComparisonResult(
            performances=performances,
            statistical_tests=statistical_tests,
            best_estimator=best_estimator,
            simulation_metadata={
                "num_poses": len(simulation_data.ground_truth_trajectory.states),
                "num_landmarks": len(simulation_data.landmarks.landmarks) if simulation_data.landmarks else 0,
                "trajectory_length": self._compute_trajectory_length(simulation_data.ground_truth_trajectory)
            }
        )
    
    def _create_estimator(self, estimator_type: str) -> BaseEstimator:
        """Create estimator instance."""
        config = self.estimator_configs[estimator_type]
        
        if estimator_type == "EKF":
            return EKFSlam(config, self.camera_calib, self.imu_calib)
        elif estimator_type == "SWBA":
            return SlidingWindowBA(config, self.camera_calib, self.imu_calib)
        elif estimator_type == "SRIF":
            return SRIFSlam(config, self.camera_calib, self.imu_calib)
        else:
            raise ValueError(f"Unknown estimator type: {estimator_type}")
    
    def _prepare_imu_batches(
        self,
        imu_measurements: List[IMUMeasurement]
    ) -> List[List[IMUMeasurement]]:
        """
        Group IMU measurements into batches for prediction steps.
        
        Simple batching by fixed time intervals.
        """
        if not imu_measurements:
            return []
        
        batches = []
        batch_dt = 0.1  # 100ms batches
        
        current_batch = []
        batch_start = imu_measurements[0].timestamp
        
        for meas in imu_measurements:
            if meas.timestamp - batch_start < batch_dt:
                current_batch.append(meas)
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [meas]
                batch_start = meas.timestamp
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _perform_statistical_tests(
        self,
        performances: Dict[str, EstimatorPerformance]
    ) -> Dict[str, Any]:
        """
        Perform statistical significance tests between estimators.
        
        Args:
            performances: Performance results for each estimator
        
        Returns:
            Dictionary of statistical test results
        """
        tests = {}
        
        # Extract ATE values for testing
        ate_values = {name: perf.trajectory_metrics.ate_rmse 
                     for name, perf in performances.items()}
        
        # Pairwise comparisons
        estimator_names = list(performances.keys())
        for i in range(len(estimator_names)):
            for j in range(i + 1, len(estimator_names)):
                name1, name2 = estimator_names[i], estimator_names[j]
                
                # Simple comparison of RMSE values
                diff = ate_values[name1] - ate_values[name2]
                percent_diff = (diff / ate_values[name2]) * 100 if ate_values[name2] > 0 else 0
                
                tests[f"{name1}_vs_{name2}"] = {
                    "ate_difference": diff,
                    "percent_difference": percent_diff,
                    "better": name1 if diff < 0 else name2
                }
        
        # Ranking
        ranking = sorted(ate_values.items(), key=lambda x: x[1])
        tests["ranking"] = [name for name, _ in ranking]
        
        return tests
    
    def _compute_trajectory_length(self, trajectory: Trajectory) -> float:
        """Compute total trajectory length."""
        length = 0.0
        for i in range(1, len(trajectory.states)):
            prev_pos = trajectory.states[i-1].pose.position
            curr_pos = trajectory.states[i].pose.position
            length += np.linalg.norm(curr_pos - prev_pos)
        return length


def compare_estimators(
    simulation_path: str,
    estimators: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> ComparisonResult:
    """
    Compare multiple estimators on simulation data.
    
    Args:
        simulation_path: Path to simulation data JSON file
        estimators: List of estimator names to compare
        output_path: Optional path to save comparison results
    
    Returns:
        Comparison results
    """
    # Load simulation data
    sim_data = load_simulation_data(simulation_path)
    
    # Extract calibrations
    camera_calib = None
    imu_calib = None
    
    if sim_data.camera_calibrations:
        camera_calib = list(sim_data.camera_calibrations.values())[0]
    if sim_data.imu_calibrations:
        imu_calib = list(sim_data.imu_calibrations.values())[0]
    
    if camera_calib is None:
        raise ValueError("No camera calibration found in simulation data")
    
    # Create runner and run comparison
    runner = EstimatorRunner(camera_calib, imu_calib, enable_profiling=True)
    results = runner.run_all_estimators(sim_data, estimators)
    
    # Save results if requested
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        
        # Also save as CSV for easy analysis
        csv_path = Path(output_path).with_suffix('.csv')
        results.to_dataframe().to_csv(csv_path, index=False)
    
    return results


def generate_comparison_table(results: ComparisonResult) -> str:
    """
    Generate a formatted comparison table.
    
    Args:
        results: Comparison results
    
    Returns:
        Formatted table as string
    """
    df = results.to_dataframe()
    
    # Format numeric columns
    format_dict = {
        'runtime_ms': '{:.1f}',
        'memory_mb': '{:.1f}',
        'ate_rmse': '{:.4f}',
        'ate_mean': '{:.4f}',
        'rpe_trans_rmse': '{:.4f}',
        'rpe_rot_rmse': '{:.4f}',
        'nees_mean': '{:.2f}'
    }
    
    # Apply formatting
    for col, fmt in format_dict.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else '')
    
    # Convert to string table
    return df.to_string(index=False)


def load_comparison_results(path: str) -> ComparisonResult:
    """
    Load comparison results from file.
    
    Args:
        path: Path to results JSON file
    
    Returns:
        Comparison results
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Reconstruct objects
    performances = {}
    for name, perf_dict in data['performances'].items():
        traj_metrics = TrajectoryMetrics(**perf_dict['trajectory_metrics']['ate'])
        
        cons_metrics = None
        if perf_dict.get('consistency_metrics'):
            cons_metrics = ConsistencyMetrics(**perf_dict['consistency_metrics'])
        
        perf = EstimatorPerformance(
            estimator_type=EstimatorType[perf_dict['estimator_type'].upper()],
            runtime_ms=perf_dict['runtime_ms'],
            peak_memory_mb=perf_dict['peak_memory_mb'],
            trajectory_metrics=traj_metrics,
            consistency_metrics=cons_metrics,
            num_iterations=perf_dict.get('num_iterations', 0),
            converged=perf_dict.get('converged', True),
            metadata=perf_dict.get('metadata', {})
        )
        performances[name] = perf
    
    return ComparisonResult(
        performances=performances,
        statistical_tests=data.get('statistical_tests', {}),
        best_estimator=data.get('best_estimator'),
        simulation_metadata=data.get('simulation_metadata', {})
    )