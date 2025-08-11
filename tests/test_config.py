"""
Unit tests for configuration models and validation.
"""

import pytest
import yaml
from pathlib import Path
import tempfile

from src.common.config import (
    SimulationConfig,
    EstimatorConfig,
    CameraIntrinsics,
    CameraExtrinsics,
    IMUConfig,
    TrajectoryConfig,
    TrajectoryType,
    EstimatorType,
    NoiseModel,
    load_simulation_config,
    load_estimator_config,
    save_config
)


class TestCameraIntrinsics:
    """Test camera intrinsics validation."""
    
    def test_valid_intrinsics(self):
        """Test creating valid camera intrinsics."""
        intrinsics = CameraIntrinsics(
            fx=458.654,
            fy=457.296,
            cx=367.215,
            cy=248.375,
            width=640,
            height=480
        )
        assert intrinsics.fx > 0
        assert intrinsics.fy > 0
        assert len(intrinsics.distortion) == 5
    
    def test_invalid_focal_length(self):
        """Test that negative focal length raises error."""
        with pytest.raises(ValueError):
            CameraIntrinsics(
                fx=-458.654,  # Invalid negative
                fy=457.296,
                cx=367.215,
                cy=248.375
            )
    
    def test_distortion_padding(self):
        """Test that 4-element distortion is padded to 5."""
        intrinsics = CameraIntrinsics(
            fx=458.654,
            fy=457.296,
            cx=367.215,
            cy=248.375,
            distortion=[0.1, 0.2, 0.3, 0.4]  # Only 4 elements
        )
        assert len(intrinsics.distortion) == 5
        assert intrinsics.distortion[4] == 0.0


class TestCameraExtrinsics:
    """Test camera extrinsics validation."""
    
    def test_quaternion_normalization(self):
        """Test that quaternions are normalized."""
        extrinsics = CameraExtrinsics(
            translation=[0.1, 0.2, 0.3],
            quaternion=[2.0, 0.0, 0.0, 0.0]  # Not normalized
        )
        # Check quaternion is normalized
        q = extrinsics.quaternion
        norm = sum(x**2 for x in q) ** 0.5
        assert abs(norm - 1.0) < 1e-6
    
    def test_invalid_translation_size(self):
        """Test that wrong translation size raises error."""
        with pytest.raises(ValueError):
            CameraExtrinsics(
                translation=[0.1, 0.2],  # Only 2 elements
                quaternion=[1.0, 0.0, 0.0, 0.0]
            )
    
    def test_invalid_quaternion_size(self):
        """Test that wrong quaternion size raises error."""
        with pytest.raises(ValueError):
            CameraExtrinsics(
                translation=[0.1, 0.2, 0.3],
                quaternion=[1.0, 0.0, 0.0]  # Only 3 elements
            )


class TestTrajectoryConfig:
    """Test trajectory configuration."""
    
    def test_default_params(self):
        """Test that default parameters are set based on trajectory type."""
        config = TrajectoryConfig(type=TrajectoryType.CIRCLE)
        assert "radius" in config.params
        assert "height" in config.params
        assert "angular_velocity" in config.params
    
    def test_custom_params(self):
        """Test that custom parameters override defaults."""
        config = TrajectoryConfig(
            type=TrajectoryType.CIRCLE,
            params={"radius": 5.0, "custom": 123}
        )
        assert config.params["radius"] == 5.0
        assert config.params["custom"] == 123
    
    def test_all_trajectory_types(self):
        """Test that all trajectory types have default params."""
        for traj_type in TrajectoryType:
            config = TrajectoryConfig(type=traj_type)
            assert len(config.params) > 0


class TestIMUConfig:
    """Test IMU configuration."""
    
    def test_noise_model_application(self):
        """Test that noise models modify parameters correctly."""
        # Test low noise model
        config_low = IMUConfig(noise_model=NoiseModel.LOW_NOISE)
        assert config_low.noise_params.accelerometer_noise_density == 0.00009
        
        # Test aggressive noise model
        config_aggressive = IMUConfig(noise_model=NoiseModel.AGGRESSIVE)
        assert config_aggressive.noise_params.accelerometer_noise_density == 0.00036
        
        # Test standard keeps defaults
        config_standard = IMUConfig(noise_model=NoiseModel.STANDARD)
        assert config_standard.noise_params.accelerometer_noise_density == 0.00018
    
    def test_rate_validation(self):
        """Test IMU rate validation."""
        # Valid rate
        config = IMUConfig(rate=200.0)
        assert config.rate == 200.0
        
        # Test bounds
        with pytest.raises(ValueError):
            IMUConfig(rate=10.0)  # Too low
        
        with pytest.raises(ValueError):
            IMUConfig(rate=2000.0)  # Too high


class TestSimulationConfig:
    """Test complete simulation configuration."""
    
    def test_minimal_config(self):
        """Test that minimal config with at least one sensor works."""
        # Should fail with no sensors
        with pytest.raises(ValueError):
            SimulationConfig()
        
        # Should work with one IMU
        config = SimulationConfig(
            imus=[IMUConfig()]
        )
        assert len(config.imus) == 1
    
    def test_seed_reproducibility(self):
        """Test that seed can be set for reproducibility."""
        config = SimulationConfig(
            imus=[IMUConfig()],
            seed=42
        )
        assert config.seed == 42
    
    def test_environment_validation(self):
        """Test environment configuration validation."""
        config = SimulationConfig(
            imus=[IMUConfig()]
        )
        assert config.environment.num_landmarks == 1000
        assert len(config.environment.landmark_range) == 3


class TestEstimatorConfig:
    """Test estimator configuration."""
    
    def test_auto_config_creation(self):
        """Test that appropriate config is created based on type."""
        # EKF should create EKF config
        config = EstimatorConfig(type=EstimatorType.EKF)
        assert config.ekf is not None
        assert config.swba is None
        assert config.srif is None
        
        # SWBA should create SWBA config
        config = EstimatorConfig(type=EstimatorType.SWBA)
        assert config.swba is not None
        assert config.ekf is None
        
        # SRIF should create SRIF config
        config = EstimatorConfig(type=EstimatorType.SRIF)
        assert config.srif is not None
        assert config.ekf is None
    
    def test_swba_specific_params(self):
        """Test SWBA-specific parameters."""
        config = EstimatorConfig(type=EstimatorType.SWBA)
        assert config.swba.window_size == 10  # Default
        assert config.swba.robust_kernel == "huber"
    
    def test_ekf_specific_params(self):
        """Test EKF-specific parameters."""
        config = EstimatorConfig(type=EstimatorType.EKF)
        assert config.ekf.chi2_threshold == 5.991
        assert config.ekf.initial_position_std == 0.1


class TestConfigIO:
    """Test configuration loading and saving."""
    
    def test_save_and_load_simulation_config(self):
        """Test saving and loading simulation configuration."""
        # Create config
        original = SimulationConfig(
            imus=[IMUConfig(id="test_imu", rate=100.0)],
            seed=12345
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            save_config(original, f.name)
            temp_path = f.name
        
        try:
            # Load and compare
            loaded = load_simulation_config(temp_path)
            assert loaded.seed == 12345
            assert len(loaded.imus) == 1
            assert loaded.imus[0].id == "test_imu"
            assert loaded.imus[0].rate == 100.0
        finally:
            Path(temp_path).unlink()
    
    def test_save_and_load_estimator_config(self):
        """Test saving and loading estimator configuration."""
        # Create config
        original = EstimatorConfig(
            type=EstimatorType.SWBA,
            output_rate=50.0
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            save_config(original, f.name)
            temp_path = f.name
        
        try:
            # Load and compare
            loaded = load_estimator_config(temp_path)
            assert loaded.type == EstimatorType.SWBA
            assert loaded.output_rate == 50.0
            assert loaded.swba is not None
        finally:
            Path(temp_path).unlink()
    
    def test_load_sample_configs(self):
        """Test loading the sample configuration files."""
        config_dir = Path("config")
        
        if config_dir.exists():
            # Test simulation config
            sim_config_path = config_dir / "simulation_circle.yaml"
            if sim_config_path.exists():
                config = load_simulation_config(sim_config_path)
                assert config.trajectory.type == TrajectoryType.CIRCLE
                assert len(config.cameras) > 0 or len(config.imus) > 0
            
            # Test estimator configs
            for estimator_file in ["ekf.yaml", "swba.yaml", "srif.yaml"]:
                est_config_path = config_dir / estimator_file
                if est_config_path.exists():
                    config = load_estimator_config(est_config_path)
                    assert config.type in [EstimatorType.EKF, EstimatorType.SWBA, EstimatorType.SRIF]