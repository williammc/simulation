"""
Unit tests for estimation base classes and metrics.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from src.estimation.base_estimator import (
    EstimatorState, EstimatorConfig, EstimatorResult, EstimatorType
)
from src.estimation.result_io import EstimatorResultStorage
from src.evaluation.metrics import (
    compute_ate, compute_rpe, compute_nees, align_trajectories,
    TrajectoryMetrics, ConsistencyMetrics
)
from src.common.data_structures import (
    Trajectory, TrajectoryState, Pose, Map, Landmark
)


class TestEstimatorState:
    """Test EstimatorState class."""
    
    def test_state_creation(self):
        """Test creating estimator state."""
        pose = Pose(
            timestamp=1.0,
            position=np.array([1, 2, 3]),
            rotation_matrix=np.eye(3)
        )
        
        state = EstimatorState(
            timestamp=1.0,
            robot_pose=pose,
            robot_velocity=np.array([0.1, 0.2, 0.3])
        )
        
        assert state.timestamp == 1.0
        assert np.allclose(state.robot_pose.position, [1, 2, 3])
        assert np.allclose(state.robot_velocity, [0.1, 0.2, 0.3])
    
    def test_get_trajectory_point(self):
        """Test getting trajectory point from state."""
        pose = Pose(
            timestamp=1.0, 
            position=np.array([1, 2, 3]),
            rotation_matrix=np.eye(3)
        )
        state = EstimatorState(timestamp=1.0, robot_pose=pose)
        
        traj_point = state.get_trajectory_point()
        assert traj_point.timestamp == 1.0
        assert np.allclose(traj_point.position, [1, 2, 3])


class TestEstimatorConfig:
    """Test EstimatorConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EstimatorConfig()
        
        assert config.estimator_type == EstimatorType.UNKNOWN
        assert config.max_landmarks == 1000
        assert config.max_iterations == 100
        assert config.convergence_threshold == 1e-6
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = EstimatorConfig(
            estimator_type=EstimatorType.EKF,
            max_landmarks=500,
            verbose=True
        )
        
        assert config.estimator_type == EstimatorType.EKF
        assert config.max_landmarks == 500
        assert config.verbose is True


class TestEstimatorResult:
    """Test EstimatorResult class."""
    
    def test_result_creation(self):
        """Test creating estimation result."""
        trajectory = Trajectory()
        landmarks = Map()
        states = []
        
        result = EstimatorResult(
            trajectory=trajectory,
            landmarks=landmarks,
            states=states,
            runtime_ms=100.0,
            iterations=10,
            converged=True,
            final_cost=0.1
        )
        
        assert result.runtime_ms == 100.0
        assert result.iterations == 10
        assert result.converged is True
        assert result.final_cost == 0.1
    
    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = EstimatorResult(
            trajectory=Trajectory(),
            landmarks=Map(),
            states=[],
            runtime_ms=100.0,
            iterations=10,
            converged=True,
            final_cost=0.1
        )
        
        result_dict = result.to_dict()
        assert "runtime_ms" in result_dict
        assert "iterations" in result_dict
        assert "converged" in result_dict


class TestTrajectoryMetrics:
    """Test trajectory error metrics."""
    
    @pytest.fixture
    def sample_trajectories(self):
        """Create sample trajectories for testing."""
        # Ground truth trajectory (straight line)
        gt_traj = Trajectory()
        for i in range(10):
            pose = Pose(
                timestamp=i * 0.1,
                position=np.array([i * 0.1, 0, 0]),
                rotation_matrix=np.eye(3)
            )
            state = TrajectoryState(pose=pose)
            gt_traj.add_state(state)
        
        # Estimated trajectory (with small error)
        est_traj = Trajectory()
        for i in range(10):
            pose = Pose(
                timestamp=i * 0.1,
                position=np.array([i * 0.1, 0.01, 0]),  # 1cm error in y
                rotation_matrix=np.eye(3)
            )
            state = TrajectoryState(pose=pose)
            est_traj.add_state(state)
        
        return gt_traj, est_traj
    
    def test_compute_ate(self, sample_trajectories):
        """Test ATE computation."""
        gt_traj, est_traj = sample_trajectories
        
        errors, metrics = compute_ate(est_traj, gt_traj, align=False)
        
        assert len(errors) == 10
        assert np.allclose(errors, 0.01)  # All errors should be 1cm
        assert np.isclose(metrics.ate_rmse, 0.01)
        assert np.isclose(metrics.ate_mean, 0.01)
    
    def test_compute_rpe(self, sample_trajectories):
        """Test RPE computation."""
        gt_traj, est_traj = sample_trajectories
        
        trans_errors, rot_errors, metrics = compute_rpe(est_traj, gt_traj, delta=1)
        
        assert len(trans_errors) == 9  # n-1 relative poses
        assert metrics.rpe_trans_mean >= 0
        assert metrics.rpe_rot_mean >= 0
    
    def test_align_trajectories(self, sample_trajectories):
        """Test trajectory alignment."""
        gt_traj, est_traj = sample_trajectories
        
        # Add offset to estimated trajectory
        for state in est_traj.states:
            state.pose.position += np.array([1, 1, 1])
        
        aligned, T = align_trajectories(est_traj, gt_traj)
        
        # After alignment, should be close to ground truth
        errors, metrics = compute_ate(aligned, gt_traj, align=False)
        assert metrics.ate_rmse < 0.1  # Should be much smaller after alignment


class TestConsistencyMetrics:
    """Test consistency metrics."""
    
    def test_compute_nees(self):
        """Test NEES computation."""
        # Create estimated states with covariance
        states = []
        for i in range(10):
            pose = Pose(
                timestamp=i * 0.1,
                position=np.array([i * 0.1, 0.01, 0]),
                rotation_matrix=np.eye(3)
            )
            state = EstimatorState(
                timestamp=i * 0.1,
                robot_pose=pose,
                robot_covariance=np.eye(6) * 0.01  # Small covariance
            )
            states.append(state)
        
        # Create ground truth
        gt_traj = Trajectory()
        for i in range(10):
            pose = Pose(
                timestamp=i * 0.1,
                position=np.array([i * 0.1, 0, 0]),
                rotation_matrix=np.eye(3)
            )
            state = TrajectoryState(pose=pose)
            gt_traj.add_state(state)
        
        nees_values, metrics = compute_nees(states, gt_traj)
        
        assert len(nees_values) == 10
        assert metrics.nees_mean > 0
        assert 0 <= metrics.nees_chi2_percentage <= 100


class TestResultStorage:
    """Test result storage and I/O."""
    
    def test_save_and_load_result(self):
        """Test saving and loading estimation results."""
        # Create sample result
        trajectory = Trajectory()
        pose = Pose(
            timestamp=0.0,
            position=np.array([1, 2, 3]),
            rotation_matrix=np.eye(3)
        )
        state = TrajectoryState(pose=pose)
        trajectory.add_state(state)
        
        landmarks = Map()
        landmark = Landmark(id=1, position=np.array([4, 5, 6]))
        landmarks.add_landmark(landmark)
        
        result = EstimatorResult(
            trajectory=trajectory,
            landmarks=landmarks,
            states=[],
            runtime_ms=100.0,
            iterations=10,
            converged=True,
            final_cost=0.1
        )
        
        config = EstimatorConfig(estimator_type=EstimatorType.EKF)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            
            # Save result
            saved_file = EstimatorResultStorage.save_result(
                result, config, output_path
            )
            
            assert saved_file.exists()
            
            # Load result
            loaded_data = EstimatorResultStorage.load_result(saved_file)
            
            assert loaded_data["algorithm"] == "ekf"
            assert loaded_data["results"]["runtime_ms"] == 100.0
            assert loaded_data["results"]["converged"] is True
    
    def test_create_kpi_summary(self):
        """Test KPI summary creation."""
        # Create and save a result
        result = EstimatorResult(
            trajectory=Trajectory(),
            landmarks=Map(),
            states=[],
            runtime_ms=100.0,
            iterations=10,
            converged=True,
            final_cost=0.1
        )
        
        config = EstimatorConfig(estimator_type=EstimatorType.EKF)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            saved_file = EstimatorResultStorage.save_result(
                result, config, output_path
            )
            
            # Create KPI summary
            kpi = EstimatorResultStorage.create_kpi_summary(saved_file)
            
            assert kpi["algorithm"] == "ekf"
            assert "metrics" in kpi
            assert "computational" in kpi["metrics"]