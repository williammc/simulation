"""
Test simulation pipeline with preintegration support.
"""

import tempfile
from pathlib import Path
import json
import pytest

from tools.simulate import run_simulation


def test_simulation_with_preintegration():
    """Test that simulation can generate data with preintegration enabled."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Run simulation with preintegration enabled
        exit_code = run_simulation(
            trajectory="circle",
            config=None,
            duration=2.0,  # Short duration for test
            output=output_dir,
            seed=42,
            noise_config=None,
            add_noise=False,
            enable_preintegration=True,
            keyframe_interval=5
        )
        
        assert exit_code == 0, "Simulation should succeed"
        
        # Check that output file was created
        output_files = list(output_dir.glob("simulation_circle_*.json"))
        assert len(output_files) == 1, "Should create one output file"
        
        # Load the data and verify preintegration was included
        with open(output_files[0], 'r') as f:
            data = json.load(f)
        
        # Check metadata
        assert data["metadata"]["preintegration_enabled"] == True
        assert data["metadata"]["keyframe_interval"] == 5
        assert data["metadata"]["num_keyframes"] is not None
        assert data["metadata"]["num_keyframes"] > 0
        
        # Check that preintegrated IMU data exists
        assert "measurements" in data
        assert "preintegrated_imu" in data["measurements"]
        assert isinstance(data["measurements"]["preintegrated_imu"], list)
        
        if data["measurements"]["preintegrated_imu"]:
            # Verify structure of preintegrated data
            first_preint = data["measurements"]["preintegrated_imu"][0]
            assert "from_keyframe_id" in first_preint
            assert "to_keyframe_id" in first_preint
            assert "delta_position" in first_preint
            assert "delta_velocity" in first_preint
            assert "delta_rotation" in first_preint
            assert "covariance" in first_preint
            assert "dt" in first_preint
            assert "num_measurements" in first_preint
        
        # Check that camera frames have keyframe markers
        camera_frames = data["measurements"]["camera_frames"]
        keyframes = [f for f in camera_frames if f.get("is_keyframe", False)]
        assert len(keyframes) > 0, "Should have keyframes marked"
        
        # Verify keyframe IDs are present
        for kf in keyframes:
            assert kf["keyframe_id"] is not None


def test_simulation_without_preintegration():
    """Test that simulation works without preintegration (backward compatibility)."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Run simulation without preintegration
        exit_code = run_simulation(
            trajectory="line",
            config=None,
            duration=1.0,
            output=output_dir,
            seed=123,
            noise_config=None,
            add_noise=False,
            enable_preintegration=False,
            keyframe_interval=10
        )
        
        assert exit_code == 0, "Simulation should succeed"
        
        # Check that output file was created
        output_files = list(output_dir.glob("simulation_line_*.json"))
        assert len(output_files) == 1, "Should create one output file"
        
        # Load the data and verify preintegration was NOT included
        with open(output_files[0], 'r') as f:
            data = json.load(f)
        
        # Check metadata
        assert data["metadata"]["preintegration_enabled"] == False
        assert data["metadata"]["num_keyframes"] is None
        
        # Preintegrated IMU should be empty or not present
        if "preintegrated_imu" in data["measurements"]:
            assert data["measurements"]["preintegrated_imu"] == []


def test_preintegration_consistency():
    """Test that preintegration produces consistent results."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Run twice with same seed
        for i in range(2):
            exit_code = run_simulation(
                trajectory="figure8",
                config=None,
                duration=1.5,
                output=output_dir / f"run_{i}",
                seed=999,
                noise_config=None,
                add_noise=False,
                enable_preintegration=True,
                keyframe_interval=3
            )
            assert exit_code == 0
        
        # Load both outputs
        file1 = list((output_dir / "run_0").glob("*.json"))[0]
        file2 = list((output_dir / "run_1").glob("*.json"))[0]
        
        with open(file1, 'r') as f:
            data1 = json.load(f)
        with open(file2, 'r') as f:
            data2 = json.load(f)
        
        # Should have same number of keyframes
        assert data1["metadata"]["num_keyframes"] == data2["metadata"]["num_keyframes"]
        
        # Should have same amount of preintegrated data
        preint1 = data1["measurements"]["preintegrated_imu"]
        preint2 = data2["measurements"]["preintegrated_imu"]
        assert len(preint1) == len(preint2)
        
        # Verify preintegrated values are consistent (with same seed)
        if preint1 and preint2:
            for p1, p2 in zip(preint1, preint2):
                assert p1["from_keyframe_id"] == p2["from_keyframe_id"]
                assert p1["to_keyframe_id"] == p2["to_keyframe_id"]
                assert abs(p1["dt"] - p2["dt"]) < 1e-6
                assert p1["num_measurements"] == p2["num_measurements"]