#!/usr/bin/env python3
"""
Integration tests for C++ binary estimator.
"""

import sys
import os
import json
import tempfile
import shutil
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


def create_test_data():
    """Create simple test data for testing."""
    # Create trajectory
    timestamps = np.linspace(0, 10, 100)
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
        Landmark(id=i, position=np.array([i, i, i]))
        for i in range(10)
    ]
    
    return trajectory, landmarks


def test_successful_execution():
    """Test successful execution of mock binary."""
    print("Testing successful execution...")
    
    # Create test data
    trajectory, landmarks = create_test_data()
    
    # Create config for mock binary
    mock_script = Path(__file__).parent / "mock_cpp_estimator.py"
    config = {
        "parameters": {
            "executable": sys.executable,  # Use current Python interpreter
            "args": [str(mock_script)],
            "timeout": 5,
            "input_file": "test_input.json",
            "output_file": "test_output.json"
        }
    }
    
    # Create estimator
    with tempfile.TemporaryDirectory() as tmpdir:
        config["parameters"]["working_dir"] = tmpdir
        
        try:
            estimator = CppBinaryEstimator(config)
            
            # Run estimation
            result = estimator.run(
                trajectory_gt=trajectory,
                landmarks=landmarks
            )
            
            # Check result - it's a dictionary with 'trajectory' and 'landmarks'
            assert result is not None
            assert 'trajectory' in result
            assert 'landmarks' in result
            assert len(result['trajectory'].states) == len(trajectory.states)
            assert len(result['landmarks'].landmarks) == len(landmarks)
            
            print("✓ Successful execution test passed")
            
        except Exception as e:
            print(f"✗ Successful execution test failed: {e}")
            raise


def test_binary_timeout():
    """Test timeout handling."""
    print("\nTesting timeout handling...")
    
    trajectory, landmarks = create_test_data()
    
    # Config with timeout flag
    config = {
        "parameters": {
            "executable": sys.executable,
            "args": [str(Path(__file__).parent / "mock_cpp_estimator.py"), "--timeout"],
            "timeout": 1,  # 1 second timeout
            "input_file": "test_input.json",
            "output_file": "test_output.json"
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config["parameters"]["working_dir"] = tmpdir
        
        try:
            estimator = CppBinaryEstimator(config)
            
            # This should timeout
            try:
                result = estimator.run(
                    trajectory_gt=trajectory,
                    landmarks=landmarks
                )
                print("✗ Timeout test failed: Should have timed out")
                return False
            except CppBinaryTimeoutError:
                print("✓ Timeout test passed")
                return True
                
        except Exception as e:
            print(f"✗ Timeout test failed with unexpected error: {e}")
            return False


def test_binary_failure():
    """Test failure handling."""
    print("\nTesting failure handling...")
    
    trajectory, landmarks = create_test_data()
    
    # Config with failure flag
    config = {
        "parameters": {
            "executable": sys.executable,
            "args": [str(Path(__file__).parent / "mock_cpp_estimator.py"), "--fail"],
            "timeout": 5,
            "input_file": "test_input.json",
            "output_file": "test_output.json"
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config["parameters"]["working_dir"] = tmpdir
        
        try:
            estimator = CppBinaryEstimator(config)
            
            # This should fail
            try:
                result = estimator.run(
                    trajectory_gt=trajectory,
                    landmarks=landmarks
                )
                print("✗ Failure test failed: Should have raised error")
                return False
            except CppBinaryExecutionError:
                print("✓ Failure test passed")
                return True
                
        except Exception as e:
            print(f"✗ Failure test failed with unexpected error: {e}")
            return False


def test_data_serialization():
    """Test data serialization and deserialization."""
    print("\nTesting data serialization...")
    
    trajectory, landmarks = create_test_data()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock result and save it
        from src.estimation.base_estimator import EstimatorResult
        
        from src.common.data_structures import Map
        
        # Create Map from landmarks
        landmark_map = Map()
        for lm in landmarks:
            landmark_map.add_landmark(lm)
        
        mock_result = EstimatorResult(
            trajectory=trajectory,
            landmarks=landmark_map,
            states=[],
            runtime_ms=100.0,
            iterations=10,
            converged=True,
            final_cost=0.1,
            metadata={"test": True}
        )
        
        # Save using EstimatorResultStorage
        output_file = Path(tmpdir) / "output.json"
        from src.common.config import EstimatorConfig, EstimatorType
        
        test_config = EstimatorConfig(type=EstimatorType.EKF)
        
        EstimatorResultStorage.save_result(
            result=mock_result,
            output_path=output_file,
            config=test_config,
            simulation_metadata={"test": True}
        )
        
        # Check file exists
        assert output_file.exists()
        
        # Load it back
        result = EstimatorResultStorage.load_result(output_file)
        
        assert 'trajectory' in result
        assert 'landmarks' in result
        assert len(result['trajectory'].states) == len(trajectory.states)
        assert len(result['landmarks'].landmarks) == len(landmarks)
        
        print("✓ Data serialization test passed")
        return True


def test_retry_mechanism():
    """Test retry mechanism on failure."""
    print("\nTesting retry mechanism...")
    
    trajectory, landmarks = create_test_data()
    
    # Config with retry enabled
    config = {
        "parameters": {
            "executable": sys.executable,
            "args": [str(Path(__file__).parent / "mock_cpp_estimator.py"), "--fail"],
            "timeout": 5,
            "retry_on_failure": True,
            "max_retries": 2,
            "input_file": "test_input.json",
            "output_file": "test_output.json"
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config["parameters"]["working_dir"] = tmpdir
        
        try:
            estimator = CppBinaryEstimator(config)
            
            # This should fail after retries
            try:
                result = estimator.run(
                    trajectory_gt=trajectory,
                    landmarks=landmarks
                )
                print("✗ Retry test failed: Should have failed after retries")
                return False
            except CppBinaryExecutionError as e:
                # Check that it actually retried
                if "attempt 3/3" in str(e) or "Failed after" in str(e):
                    print("✓ Retry mechanism test passed")
                    return True
                else:
                    print(f"✗ Retry test failed: {e}")
                    return False
                    
        except Exception as e:
            print(f"✗ Retry test failed with unexpected error: {e}")
            return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("C++ Binary Estimator Integration Tests")
    print("=" * 60)
    
    tests = [
        test_successful_execution,
        test_binary_timeout,
        test_binary_failure,
        test_data_serialization,
        test_retry_mechanism
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"Test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())