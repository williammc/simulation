"""
Unit tests for EstimatorResultStorage.
"""

import json
import numpy as np
from pathlib import Path
import tempfile
import pytest
from datetime import datetime

from src.estimation.result_io import EstimatorResultStorage, compare_results
from src.estimation.base_estimator import (
    EstimatorResult, EstimatorConfig, EstimatorType
)
from src.common.data_structures import (
    Trajectory, TrajectoryState, Pose, Map, Landmark
)
from src.evaluation.metrics import TrajectoryMetrics, ConsistencyMetrics


class TestEstimatorResultStorage:
    """Test EstimatorResultStorage functionality."""
    
    @pytest.fixture
    def sample_result(self):
        """Create sample estimation result."""
        # Create trajectory
        trajectory = Trajectory(frame_id="world")
        for i in range(10):
            pose = Pose(
                timestamp=i * 0.1,
                position=np.array([i, i*0.5, 0]),
                rotation_matrix=np.eye(3)
            )
            state = TrajectoryState(pose=pose, velocity=np.array([1, 0.5, 0]))
            trajectory.add_state(state)
        
        # Create landmarks
        landmarks = Map(frame_id="world")
        for i in range(5):
            landmark = Landmark(
                id=i,
                position=np.array([i*2, i*3, 1]),
                descriptor=np.random.randn(128),
                covariance=np.eye(3) * 0.01
            )
            landmarks.add_landmark(landmark)
        
        # Create result
        result = EstimatorResult(
            trajectory=trajectory,
            landmarks=landmarks,
            runtime_ms=1234.5,
            iterations=42,
            converged=True,
            final_cost=0.001,
            states=[],  # Empty for simplicity
            metadata={"test_key": "test_value"}
        )
        
        return result
    
    @pytest.fixture
    def sample_config(self):
        """Create sample estimator configuration."""
        config = EstimatorConfig()
        config.estimator_type = EstimatorType.EKF
        config.max_landmarks = 100
        config.max_iterations = 50
        config.convergence_threshold = 1e-6
        return config
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample trajectory metrics."""
        metrics = TrajectoryMetrics(
            ate_rmse=0.123,
            ate_mean=0.100,
            ate_median=0.095,
            ate_std=0.045,
            ate_min=0.010,
            ate_max=0.250,
            trajectory_length=10.5
        )
        return metrics
    
    def test_save_and_load_basic(self, sample_result, sample_config):
        """Test basic save and load functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            
            # Save result
            saved_file = EstimatorResultStorage.save_result(
                result=sample_result,
                config=sample_config,
                output_path=output_path
            )
            
            assert saved_file.exists()
            assert saved_file.suffix == ".json"
            
            # Load result
            loaded_data = EstimatorResultStorage.load_result(saved_file)
            
            # Check basic fields
            assert loaded_data["algorithm"] == "ekf"
            assert loaded_data["results"]["runtime_ms"] == 1234.5
            assert loaded_data["results"]["iterations"] == 42
            assert loaded_data["results"]["converged"] is True
            assert loaded_data["results"]["final_cost"] == 0.001
            assert loaded_data["results"]["metadata"]["test_key"] == "test_value"
            
            # Check trajectory was loaded
            assert "trajectory" in loaded_data
            trajectory = loaded_data["trajectory"]
            assert len(trajectory.states) == 10
            assert trajectory.frame_id == "world"
            
            # Check first state
            first_state = trajectory.states[0]
            assert first_state.pose.timestamp == 0.0
            np.testing.assert_array_almost_equal(
                first_state.pose.position, np.array([0, 0, 0])
            )
            
            # Check landmarks were loaded
            assert "landmarks" in loaded_data
            landmarks = loaded_data["landmarks"]
            assert len(landmarks.landmarks) == 5
            assert landmarks.frame_id == "world"
            
            # Check first landmark
            first_landmark = landmarks.landmarks[0]
            assert first_landmark.id == 0
            np.testing.assert_array_almost_equal(
                first_landmark.position, np.array([0, 0, 1])
            )
    
    def test_save_with_metrics(self, sample_result, sample_config, sample_metrics):
        """Test saving with trajectory metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            
            # Create consistency metrics
            consistency_metrics = ConsistencyMetrics(
                nees_mean=2.5,
                nees_std=0.8,
                nees_chi2_percentage=85.0
            )
            
            # Save with metrics
            saved_file = EstimatorResultStorage.save_result(
                result=sample_result,
                config=sample_config,
                output_path=output_path,
                trajectory_metrics=sample_metrics,
                consistency_metrics=consistency_metrics
            )
            
            # Load and check metrics
            loaded_data = EstimatorResultStorage.load_result(saved_file)
            
            assert "metrics" in loaded_data
            assert "trajectory_error" in loaded_data["metrics"]
            assert loaded_data["metrics"]["trajectory_error"]["ate"]["rmse"] == 0.123
            assert loaded_data["metrics"]["trajectory_error"]["ate"]["mean"] == 0.100
            
            assert "consistency" in loaded_data["metrics"]
            assert loaded_data["metrics"]["consistency"]["nees_mean"] == 2.5
            assert loaded_data["metrics"]["consistency"]["nees_std"] == 0.8
            assert loaded_data["metrics"]["consistency"]["chi2_percentage"] == 85.0
    
    def test_save_with_simulation_metadata(self, sample_result, sample_config):
        """Test saving with simulation metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            
            sim_metadata = {
                "input_file": "test_trajectory.json",
                "trajectory_type": "circle",
                "duration": 10.0,
                "noise_level": 0.1
            }
            
            # Save with metadata
            saved_file = EstimatorResultStorage.save_result(
                result=sample_result,
                config=sample_config,
                output_path=output_path,
                simulation_metadata=sim_metadata
            )
            
            # Load and check metadata
            loaded_data = EstimatorResultStorage.load_result(saved_file)
            
            assert "simulation" in loaded_data
            assert loaded_data["simulation"]["input_file"] == "test_trajectory.json"
            assert loaded_data["simulation"]["trajectory_type"] == "circle"
            assert loaded_data["simulation"]["duration"] == 10.0
    
    def test_trajectory_conversion(self, sample_result):
        """Test trajectory to dict and back conversion."""
        trajectory = sample_result.trajectory
        
        # Convert to dict
        traj_dict = EstimatorResultStorage._trajectory_to_dict(trajectory)
        
        assert traj_dict["frame_id"] == "world"
        assert len(traj_dict["poses"]) == 10
        assert traj_dict["poses"][0]["timestamp"] == 0.0
        assert traj_dict["poses"][0]["position"] == [0, 0, 0]
        assert len(traj_dict["poses"][0]["quaternion"]) == 4  # Quaternion has 4 elements
        
        # Convert back to trajectory
        recovered_traj = EstimatorResultStorage._dict_to_trajectory(traj_dict)
        
        assert recovered_traj.frame_id == "world"
        assert len(recovered_traj.states) == 10
        np.testing.assert_array_almost_equal(
            recovered_traj.states[0].pose.position,
            trajectory.states[0].pose.position
        )
    
    def test_landmarks_conversion(self, sample_result):
        """Test landmarks to dict and back conversion."""
        landmarks = sample_result.landmarks
        
        # Convert to dict
        landmarks_dict = EstimatorResultStorage._landmarks_to_dict(landmarks)
        
        assert landmarks_dict["frame_id"] == "world"
        assert len(landmarks_dict["landmarks"]) == 5
        assert landmarks_dict["landmarks"][0]["id"] == 0
        assert landmarks_dict["landmarks"][0]["position"] == [0, 0, 1]
        
        # Convert back to landmarks
        recovered_landmarks = EstimatorResultStorage._dict_to_landmarks(landmarks_dict)
        
        assert recovered_landmarks.frame_id == "world"
        assert len(recovered_landmarks.landmarks) == 5
        np.testing.assert_array_almost_equal(
            recovered_landmarks.landmarks[0].position,
            landmarks.landmarks[0].position
        )
    
    def test_create_kpi_summary(self, sample_result, sample_config):
        """Test KPI summary creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            
            # Save result
            saved_file = EstimatorResultStorage.save_result(
                result=sample_result,
                config=sample_config,
                output_path=output_path
            )
            
            # Create KPI summary
            kpi = EstimatorResultStorage.create_kpi_summary(saved_file)
            
            assert kpi["algorithm"] == "ekf"
            assert "run_id" in kpi
            assert "timestamp" in kpi
            assert "metrics" in kpi
            assert "computational" in kpi["metrics"]
            assert kpi["metrics"]["computational"]["iterations"] == 42
            assert "convergence" in kpi["metrics"]
            assert kpi["metrics"]["convergence"]["converged"] is True
    
    def test_filename_generation(self, sample_result, sample_config):
        """Test automatic filename generation when directory is provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Save to directory (should generate filename)
            saved_file = EstimatorResultStorage.save_result(
                result=sample_result,
                config=sample_config,
                output_path=output_dir
            )
            
            assert saved_file.parent == output_dir
            assert saved_file.name.startswith("slam_ekf_")
            assert saved_file.suffix == ".json"
    
    def test_specific_filename(self, sample_result, sample_config):
        """Test saving with specific filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "my_custom_result.json"
            
            # Save to specific file
            saved_file = EstimatorResultStorage.save_result(
                result=sample_result,
                config=sample_config,
                output_path=output_file
            )
            
            assert saved_file == output_file
            assert saved_file.exists()
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            EstimatorResultStorage.load_result(Path("nonexistent.json"))
    
    def test_load_invalid_json(self):
        """Test loading invalid JSON raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("not valid json {")
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(json.JSONDecodeError):
                EstimatorResultStorage.load_result(temp_path)
        finally:
            temp_path.unlink()


class TestCompareResults:
    """Test result comparison functionality."""
    
    def create_test_result(self, algorithm: str, ate_rmse: float, runtime_ms: float) -> Path:
        """Helper to create a test result file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            result_data = {
                "algorithm": algorithm,
                "run_id": f"test_{algorithm}",
                "timestamp": datetime.now().isoformat(),
                "results": {
                    "runtime_ms": runtime_ms,
                    "iterations": 10,
                    "converged": True
                },
                "metrics": {
                    "trajectory_error": {
                        "ate": {"rmse": ate_rmse}
                    },
                    "computational": {
                        "total_time": runtime_ms / 1000.0
                    }
                }
            }
            json.dump(result_data, f)
            return Path(f.name)
    
    def test_compare_multiple_results(self):
        """Test comparing multiple estimation results."""
        # Create test result files
        result1 = self.create_test_result("ekf", ate_rmse=0.15, runtime_ms=1000)
        result2 = self.create_test_result("swba", ate_rmse=0.10, runtime_ms=2000)
        result3 = self.create_test_result("srif", ate_rmse=0.12, runtime_ms=1500)
        
        try:
            # Compare results
            comparison = compare_results([result1, result2, result3])
            
            assert comparison["num_results"] == 3
            assert len(comparison["results"]) == 3
            
            # Check best accuracy (lowest ATE)
            assert comparison["best"]["accuracy"]["algorithm"] == "swba"
            assert comparison["best"]["accuracy"]["ate_rmse"] == 0.10
            
            # Check best speed (lowest runtime)
            assert comparison["best"]["speed"]["algorithm"] == "ekf"
            assert comparison["best"]["speed"]["time_seconds"] == 1.0
            
        finally:
            # Clean up
            result1.unlink()
            result2.unlink()
            result3.unlink()
    
    def test_compare_with_output_file(self):
        """Test saving comparison to file."""
        result1 = self.create_test_result("ekf", ate_rmse=0.15, runtime_ms=1000)
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_file = Path(tmpdir) / "comparison.json"
                
                # Compare and save
                compare_results([result1], output_file=output_file)
                
                assert output_file.exists()
                
                # Load and verify
                with open(output_file, 'r') as f:
                    loaded = json.load(f)
                
                assert loaded["num_results"] == 1
                assert loaded["results"][0]["algorithm"] == "ekf"
                
        finally:
            result1.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])