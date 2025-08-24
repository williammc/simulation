#!/usr/bin/env python3
"""
Integration tests for the actual C++ mock_estimator binary.
This test requires the C++ code to be built first.
"""

import sys
import os
import json
import tempfile
import shutil
import pytest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.estimation.cpp_binary_estimator import (
    CppBinaryEstimator, CppBinaryTimeoutError, CppBinaryExecutionError
)
from src.estimation.result_io import EstimatorResultStorage
from src.common.data_structures import Trajectory, Landmark, TrajectoryState, Pose
import scipy.spatial.transform as spt


def create_test_data(num_poses=20, num_landmarks=10):
    """Create simple test data for testing."""
    # Create trajectory
    timestamps = np.linspace(0, 10, num_poses)
    positions = np.column_stack([
        np.sin(timestamps),
        np.cos(timestamps),
        timestamps * 0.1
    ])
    
    trajectory = Trajectory()
    for i, t in enumerate(timestamps):
        pose = Pose(
            timestamp=t,
            position=positions[i],
            rotation_matrix=spt.Rotation.identity().as_matrix()
        )
        state = TrajectoryState(pose=pose)
        trajectory.add_state(state)
    
    # Create landmarks
    landmarks = [
        Landmark(id=i, position=np.array([i*0.5, i*0.5, i*0.2]))
        for i in range(num_landmarks)
    ]
    
    return trajectory, landmarks


@pytest.fixture
def mock_estimator_path():
    """Get path to the compiled mock_estimator binary."""
    binary_path = Path(__file__).parent.parent / "cpp_estimation" / "build" / "examples" / "mock_estimator"
    
    if not binary_path.exists():
        pytest.skip(f"C++ mock_estimator not found at {binary_path}. "
                   f"Build it first: cd cpp_estimation && mkdir build && cd build && cmake .. && make")
    
    return binary_path


class TestCppMockEstimator:
    """Test suite for the C++ mock_estimator binary."""
    
    def test_help_output(self, mock_estimator_path):
        """Test that the binary provides help output."""
        import subprocess
        
        result = subprocess.run(
            [str(mock_estimator_path), "--help"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Mock C++ SLAM Estimator" in result.stdout
        assert "--input" in result.stdout
        assert "--output" in result.stdout
        assert "--noise" in result.stdout
    
    def test_successful_execution(self, mock_estimator_path):
        """Test successful execution of the C++ mock_estimator."""
        trajectory, landmarks = create_test_data(10, 5)
        
        # Create config for the C++ binary
        config = {
            "parameters": {
                "executable": str(mock_estimator_path),
                "args": ["--noise", "0.05", "--delay", "10"],
                "timeout": 30,
                "input_file": "test_input.json",
                "output_file": "test_output.json"
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config["parameters"]["working_dir"] = tmpdir
            
            # Create estimator and run
            estimator = CppBinaryEstimator(config)
            result = estimator.run(
                trajectory_gt=trajectory,
                landmarks=landmarks
            )
            
            # Validate result
            assert result is not None
            assert 'trajectory' in result
            assert 'landmarks' in result
            
            # Check trajectory
            traj = result['trajectory']
            assert len(traj.states) == len(trajectory.states)
            
            # Check landmarks
            lm_map = result['landmarks']
            assert len(lm_map.landmarks) == len(landmarks)
            
            # Verify noise was added (positions should be different)
            orig_pos = trajectory.states[0].pose.position
            est_pos = traj.states[0].pose.position
            diff = np.linalg.norm(orig_pos - est_pos)
            assert diff > 1e-6, "Positions should be different due to noise"
    
    def test_different_noise_levels(self, mock_estimator_path):
        """Test that different noise levels produce different results."""
        trajectory, landmarks = create_test_data(5, 3)
        
        results = []
        noise_levels = [0.01, 0.05, 0.1]
        
        for noise in noise_levels:
            config = {
                "parameters": {
                    "executable": str(mock_estimator_path),
                    "args": ["--noise", str(noise)],
                    "timeout": 30,
                    "input_file": "test_input.json",
                    "output_file": "test_output.json"
                }
            }
            
            with tempfile.TemporaryDirectory() as tmpdir:
                config["parameters"]["working_dir"] = tmpdir
                
                estimator = CppBinaryEstimator(config)
                result = estimator.run(
                    trajectory_gt=trajectory,
                    landmarks=landmarks
                )
                results.append(result)
        
        # Compare errors - higher noise should generally produce larger errors
        errors = []
        for result in results:
            traj = result['trajectory']
            error = 0
            for i in range(len(trajectory.states)):
                orig_pos = trajectory.states[i].pose.position
                est_pos = traj.states[i].pose.position
                error += np.linalg.norm(orig_pos - est_pos)
            errors.append(error / len(trajectory.states))
        
        # Generally, higher noise leads to higher error (with some randomness)
        # We just check that they're different, not strictly increasing
        assert len(set(errors)) > 1, "Different noise levels should produce different errors"
    
    def test_large_dataset(self, mock_estimator_path):
        """Test with a larger dataset to verify scalability."""
        trajectory, landmarks = create_test_data(100, 50)
        
        config = {
            "parameters": {
                "executable": str(mock_estimator_path),
                "args": ["--noise", "0.02", "--delay", "5"],
                "timeout": 60,
                "input_file": "test_input.json",
                "output_file": "test_output.json"
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config["parameters"]["working_dir"] = tmpdir
            
            estimator = CppBinaryEstimator(config)
            result = estimator.run(
                trajectory_gt=trajectory,
                landmarks=landmarks
            )
            
            # Check all data was processed
            assert len(result['trajectory'].states) == 100
            assert len(result['landmarks'].landmarks) == 50
    
    def test_empty_landmarks(self, mock_estimator_path):
        """Test with trajectory but no landmarks."""
        trajectory, _ = create_test_data(10, 0)
        landmarks = []
        
        config = {
            "parameters": {
                "executable": str(mock_estimator_path),
                "args": ["--noise", "0.01"],
                "timeout": 30,
                "input_file": "test_input.json",
                "output_file": "test_output.json"
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config["parameters"]["working_dir"] = tmpdir
            
            estimator = CppBinaryEstimator(config)
            result = estimator.run(
                trajectory_gt=trajectory,
                landmarks=landmarks
            )
            
            # Should handle empty landmarks gracefully
            assert len(result['trajectory'].states) == 10
            assert len(result['landmarks'].landmarks) == 0
    
    def test_output_format(self, mock_estimator_path):
        """Test that output format matches expected structure."""
        trajectory, landmarks = create_test_data(5, 5)
        
        config = {
            "parameters": {
                "executable": str(mock_estimator_path),
                "args": ["--noise", "0.03"],
                "timeout": 30,
                "input_file": "test_input.json",
                "output_file": "test_output.json"
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config["parameters"]["working_dir"] = tmpdir
            
            estimator = CppBinaryEstimator(config)
            
            # Write input manually to check output format
            from src.estimation.cpp_binary_estimator import CppBinaryEstimator as Estimator
            input_data = estimator._prepare_simulation_data(trajectory, landmarks, None, None)
            input_path = Path(tmpdir) / "test_input.json"
            with open(input_path, 'w') as f:
                json.dump(input_data, f)
            
            # Run the binary directly
            import subprocess
            result = subprocess.run(
                [str(mock_estimator_path), "--input", str(input_path), 
                 "--output", str(Path(tmpdir) / "test_output.json")],
                cwd=tmpdir,
                capture_output=True,
                timeout=30
            )
            
            assert result.returncode == 0
            
            # Load and check output format
            output_path = Path(tmpdir) / "test_output.json"
            with open(output_path) as f:
                output_data = json.load(f)
            
            # Check required fields
            assert "metadata" in output_data
            assert "estimated_trajectory" in output_data
            assert "estimated_landmarks" in output_data
            assert "runtime_ms" in output_data
            assert "iterations" in output_data
            assert "converged" in output_data
            assert "final_cost" in output_data
            
            # Check metadata
            assert output_data["metadata"]["estimator"] == "mock_cpp"
            assert "version" in output_data["metadata"]
            assert "noise_level" in output_data["metadata"]
    
    def test_performance_metrics(self, mock_estimator_path):
        """Test that performance metrics are reported correctly."""
        trajectory, landmarks = create_test_data(10, 10)
        
        delays = [10, 50, 100]  # milliseconds
        
        for delay in delays:
            config = {
                "parameters": {
                    "executable": str(mock_estimator_path),
                    "args": ["--delay", str(delay)],
                    "timeout": 30,
                    "input_file": "test_input.json",
                    "output_file": "test_output.json"
                }
            }
            
            with tempfile.TemporaryDirectory() as tmpdir:
                config["parameters"]["working_dir"] = tmpdir
                
                estimator = CppBinaryEstimator(config)
                
                import time
                start_time = time.time()
                result = estimator.run(
                    trajectory_gt=trajectory,
                    landmarks=landmarks
                )
                elapsed = (time.time() - start_time) * 1000  # ms
                
                # Check that delay is reflected in runtime
                # (allowing some overhead for file I/O)
                assert elapsed >= delay, f"Runtime {elapsed}ms should be >= delay {delay}ms"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])