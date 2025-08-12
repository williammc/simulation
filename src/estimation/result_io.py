"""
Result storage and I/O for SLAM estimators.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from src.estimation.base_estimator import EstimatorResult, EstimatorConfig, EstimatorType
from src.evaluation.metrics import TrajectoryMetrics, ConsistencyMetrics
from src.common.data_structures import Trajectory, Map, Pose, Landmark
from src.common.json_io import NumpyJSONEncoder


class EstimatorResultStorage:
    """
    Handle storage and retrieval of estimator results.
    
    Provides standardized format for saving SLAM estimation results
    that can be used for evaluation and comparison.
    """
    
    @staticmethod
    def save_result(
        result: EstimatorResult,
        config: EstimatorConfig,
        output_path: Path,
        trajectory_metrics: Optional[TrajectoryMetrics] = None,
        consistency_metrics: Optional[ConsistencyMetrics] = None,
        simulation_metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save estimator result to JSON file.
        
        Args:
            result: Estimation result
            config: Estimator configuration
            output_path: Output directory or file path
            trajectory_metrics: Computed trajectory error metrics
            consistency_metrics: Computed consistency metrics
            simulation_metadata: Original simulation metadata
        
        Returns:
            Path to saved file
        """
        # Generate filename if directory provided
        if output_path.is_dir():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = str(uuid.uuid4())[:8]
            filename = f"slam_{config.estimator_type.value}_{timestamp}_{run_id}.json"
            output_file = output_path / filename
        else:
            output_file = output_path
            run_id = str(uuid.uuid4())[:8]
        
        # Build result dictionary
        result_dict = {
            # Metadata
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "algorithm": config.estimator_type.value,
            
            # Configuration
            "configuration": {
                "estimator_type": config.estimator_type.value,
                "max_landmarks": config.max_landmarks,
                "max_iterations": config.max_iterations,
                "convergence_threshold": config.convergence_threshold,
                "outlier_threshold": config.outlier_threshold,
                "enable_marginalization": config.enable_marginalization,
                "marginalization_window": config.marginalization_window,
                "process_noise": {
                    "position": config.process_noise_position,
                    "orientation": config.process_noise_orientation,
                    "velocity": config.process_noise_velocity,
                    "bias": config.process_noise_bias
                },
                "measurement_noise": {
                    "camera": config.measurement_noise_camera,
                    "imu_accel": config.measurement_noise_imu_accel,
                    "imu_gyro": config.measurement_noise_imu_gyro
                }
            },
            
            # Results
            "results": {
                "runtime_ms": result.runtime_ms,
                "iterations": result.iterations,
                "converged": result.converged,
                "final_cost": result.final_cost,
                "num_poses": len(result.trajectory.states),
                "num_landmarks": len(result.landmarks.landmarks),
                "metadata": result.metadata
            },
            
            # Metrics (if computed)
            "metrics": {}
        }
        
        # Add trajectory metrics if available
        if trajectory_metrics:
            result_dict["metrics"]["trajectory_error"] = trajectory_metrics.to_dict()
        
        # Add consistency metrics if available
        if consistency_metrics:
            result_dict["metrics"]["consistency"] = consistency_metrics.to_dict()
        
        # Add simulation metadata if available
        if simulation_metadata:
            result_dict["simulation"] = simulation_metadata
        
        # Add trajectory data
        result_dict["estimated_trajectory"] = EstimatorResultStorage._trajectory_to_dict(result.trajectory)
        
        # Add landmark data
        result_dict["estimated_landmarks"] = EstimatorResultStorage._landmarks_to_dict(result.landmarks)
        
        # Add state history (compact format)
        result_dict["state_history"] = EstimatorResultStorage._states_to_dict(result.states)
        
        # Save to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2, cls=NumpyJSONEncoder)
        
        return output_file
    
    @staticmethod
    def load_result(filepath: Path) -> Dict[str, Any]:
        """
        Load estimator result from JSON file.
        
        Args:
            filepath: Path to result JSON file
        
        Returns:
            Dictionary containing result data
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert trajectory from dict
        if "estimated_trajectory" in data:
            data["trajectory"] = EstimatorResultStorage._dict_to_trajectory(
                data["estimated_trajectory"]
            )
        
        # Convert landmarks from dict
        if "estimated_landmarks" in data:
            data["landmarks"] = EstimatorResultStorage._dict_to_landmarks(
                data["estimated_landmarks"]
            )
        
        return data
    
    @staticmethod
    def _trajectory_to_dict(trajectory: Trajectory) -> Dict[str, Any]:
        """Convert trajectory to dictionary format."""
        return {
            "frame_id": trajectory.frame_id,
            "poses": [
                {
                    "timestamp": state.pose.timestamp,
                    "position": state.pose.position.tolist(),
                    "quaternion": state.pose.quaternion.tolist(),
                    "velocity": state.velocity.tolist() if state.velocity is not None else None
                }
                for state in trajectory.states
            ]
        }
    
    @staticmethod
    def _dict_to_trajectory(data: Dict[str, Any]) -> Trajectory:
        """Convert dictionary to trajectory."""
        trajectory = Trajectory(frame_id=data.get("frame_id", "world"))
        
        for pose_dict in data.get("poses", []):
            pose = Pose(
                timestamp=pose_dict["timestamp"],
                position=np.array(pose_dict["position"]),
                quaternion=np.array(pose_dict["quaternion"])
            )
            
            from src.common.data_structures import TrajectoryState
            state = TrajectoryState(
                pose=pose,
                velocity=np.array(pose_dict["velocity"]) if pose_dict.get("velocity") else None
            )
            trajectory.add_state(state)
        
        return trajectory
    
    @staticmethod
    def _landmarks_to_dict(landmarks: Map) -> Dict[str, Any]:
        """Convert landmarks to dictionary format."""
        return {
            "frame_id": landmarks.frame_id,
            "landmarks": [
                {
                    "id": lmk.id,
                    "position": lmk.position.tolist(),
                    "descriptor": lmk.descriptor.tolist() if lmk.descriptor is not None else None,
                    "covariance": lmk.covariance.tolist() if lmk.covariance is not None else None
                }
                for lmk in landmarks.landmarks.values()
            ]
        }
    
    @staticmethod
    def _dict_to_landmarks(data: Dict[str, Any]) -> Map:
        """Convert dictionary to landmarks."""
        landmarks = Map(frame_id=data.get("frame_id", "world"))
        
        for lmk_dict in data.get("landmarks", []):
            landmark = Landmark(
                id=lmk_dict["id"],
                position=np.array(lmk_dict["position"]),
                descriptor=np.array(lmk_dict["descriptor"]) if lmk_dict.get("descriptor") else None,
                covariance=np.array(lmk_dict["covariance"]) if lmk_dict.get("covariance") else None
            )
            landmarks.add_landmark(landmark)
        
        return landmarks
    
    @staticmethod
    def _states_to_dict(states: list) -> list:
        """Convert state history to compact dictionary format."""
        state_list = []
        
        for state in states:
            state_dict = {
                "t": state.timestamp,
                "p": state.robot_pose.position.tolist(),
                "q": state.robot_pose.quaternion.tolist()
            }
            
            # Add optional fields only if present
            if state.robot_velocity is not None:
                state_dict["v"] = state.robot_velocity.tolist()
            
            # Add covariance diagonal for space efficiency
            if state.robot_covariance is not None:
                state_dict["cov_diag"] = np.diag(state.robot_covariance).tolist()
            
            state_list.append(state_dict)
        
        return state_list
    
    @staticmethod
    def create_kpi_summary(
        result_file: Path,
        ground_truth_file: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Create KPI summary from result file.
        
        Args:
            result_file: Path to estimator result JSON
            ground_truth_file: Optional path to ground truth data
        
        Returns:
            KPI summary dictionary
        """
        # Load result
        result_data = EstimatorResultStorage.load_result(result_file)
        
        kpi = {
            "run_id": result_data.get("run_id"),
            "timestamp": result_data.get("timestamp"),
            "algorithm": result_data.get("algorithm"),
            "configuration": {
                "trajectory_type": result_data.get("simulation", {}).get("trajectory_type", "unknown")
            },
            "metrics": result_data.get("metrics", {})
        }
        
        # Add computational metrics
        kpi["metrics"]["computational"] = {
            "total_time": result_data.get("results", {}).get("runtime_ms", 0) / 1000.0,
            "iterations": result_data.get("results", {}).get("iterations", 0),
            "poses_per_second": (
                result_data.get("results", {}).get("num_poses", 0) / 
                (result_data.get("results", {}).get("runtime_ms", 1) / 1000.0)
            )
        }
        
        # Add convergence info
        kpi["metrics"]["convergence"] = {
            "converged": result_data.get("results", {}).get("converged", False),
            "final_cost": result_data.get("results", {}).get("final_cost", 0),
            "iterations": result_data.get("results", {}).get("iterations", 0)
        }
        
        return kpi


def compare_results(
    result_files: list[Path],
    output_file: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Compare multiple estimation results.
    
    Args:
        result_files: List of result JSON files
        output_file: Optional path to save comparison
    
    Returns:
        Comparison summary
    """
    comparison = {
        "num_results": len(result_files),
        "results": [],
        "best": {}
    }
    
    best_ate = float('inf')
    best_time = float('inf')
    best_ate_alg = None
    best_time_alg = None
    
    for result_file in result_files:
        kpi = EstimatorResultStorage.create_kpi_summary(result_file)
        comparison["results"].append(kpi)
        
        # Track best performers
        ate = kpi.get("metrics", {}).get("trajectory_error", {}).get("ate", {}).get("rmse", float('inf'))
        time = kpi.get("metrics", {}).get("computational", {}).get("total_time", float('inf'))
        
        if ate < best_ate:
            best_ate = ate
            best_ate_alg = kpi.get("algorithm")
        
        if time < best_time:
            best_time = time
            best_time_alg = kpi.get("algorithm")
    
    comparison["best"] = {
        "accuracy": {"algorithm": best_ate_alg, "ate_rmse": best_ate},
        "speed": {"algorithm": best_time_alg, "time_seconds": best_time}
    }
    
    # Save comparison if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)
    
    return comparison