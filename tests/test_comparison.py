"""
Unit tests for estimator comparison tools.
"""

import pytest
import numpy as np
from pathlib import Path
import json
import tempfile

from src.evaluation.comparison import (
    EstimatorPerformance, ComparisonResult, EstimatorRunner,
    compare_estimators, generate_comparison_table
)
from src.evaluation.metrics import TrajectoryMetrics, ConsistencyMetrics
from src.estimation.base_estimator import EstimatorType
from src.common.data_structures import (
    Pose, Trajectory, TrajectoryState, Map, Landmark,
    IMUMeasurement, CameraFrame, CameraObservation, ImagePoint,
    CameraCalibration, CameraIntrinsics, CameraExtrinsics, CameraModel
)
from src.common.json_io import SimulationData


class TestEstimatorPerformance:
    """Test EstimatorPerformance class."""
    
    def test_performance_creation(self):
        """Test creating performance metrics."""
        traj_metrics = TrajectoryMetrics(
            ate_rmse=0.5,
            ate_mean=0.4,
            rpe_trans_rmse=0.1,
            rpe_rot_rmse=0.05
        )
        
        perf = EstimatorPerformance(
            estimator_type=EstimatorType.EKF,
            runtime_ms=100.0,
            peak_memory_mb=50.0,
            trajectory_metrics=traj_metrics
        )
        
        assert perf.estimator_type == EstimatorType.EKF
        assert perf.runtime_ms == 100.0
        assert perf.peak_memory_mb == 50.0
        assert perf.trajectory_metrics.ate_rmse == 0.5
    
    def test_performance_to_dict(self):
        """Test serialization to dictionary."""
        traj_metrics = TrajectoryMetrics(ate_rmse=0.5)
        cons_metrics = ConsistencyMetrics(nees_mean=3.0, is_consistent=True)
        
        perf = EstimatorPerformance(
            estimator_type=EstimatorType.SWBA,
            runtime_ms=200.0,
            peak_memory_mb=100.0,
            trajectory_metrics=traj_metrics,
            consistency_metrics=cons_metrics
        )
        
        data = perf.to_dict()
        assert data["estimator_type"] == "swba"
        assert data["runtime_ms"] == 200.0
        assert data["consistency_metrics"]["nees_mean"] == 3.0


class TestComparisonResult:
    """Test ComparisonResult class."""
    
    def test_comparison_result_creation(self):
        """Test creating comparison results."""
        performances = {
            "EKF": EstimatorPerformance(
                estimator_type=EstimatorType.EKF,
                runtime_ms=100.0,
                peak_memory_mb=50.0,
                trajectory_metrics=TrajectoryMetrics(ate_rmse=0.5)
            ),
            "SWBA": EstimatorPerformance(
                estimator_type=EstimatorType.SWBA,
                runtime_ms=200.0,
                peak_memory_mb=100.0,
                trajectory_metrics=TrajectoryMetrics(ate_rmse=0.3)
            )
        }
        
        result = ComparisonResult(
            performances=performances,
            best_estimator="SWBA"
        )
        
        assert len(result.performances) == 2
        assert result.best_estimator == "SWBA"
    
    def test_comparison_to_dataframe(self):
        """Test conversion to pandas DataFrame."""
        performances = {
            "EKF": EstimatorPerformance(
                estimator_type=EstimatorType.EKF,
                runtime_ms=100.0,
                peak_memory_mb=50.0,
                trajectory_metrics=TrajectoryMetrics(
                    ate_rmse=0.5,
                    rpe_trans_rmse=0.1
                )
            )
        }
        
        result = ComparisonResult(performances=performances)
        df = result.to_dataframe()
        
        assert len(df) == 1
        assert df.iloc[0]["estimator"] == "EKF"
        assert df.iloc[0]["ate_rmse"] == 0.5
        assert df.iloc[0]["runtime_ms"] == 100.0


class TestEstimatorRunner:
    """Test EstimatorRunner class."""
    
    @pytest.fixture
    def camera_calibration(self):
        """Create camera calibration."""
        intrinsics = CameraIntrinsics(
            model=CameraModel.PINHOLE,
            width=640,
            height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            distortion=np.zeros(4)
        )
        
        extrinsics = CameraExtrinsics(B_T_C=np.eye(4))
        
        return CameraCalibration(
            camera_id="cam0",
            intrinsics=intrinsics,
            extrinsics=extrinsics
        )
    
    @pytest.fixture
    def simple_simulation_data(self, camera_calibration):
        """Create simple simulation data for testing."""
        # Ground truth trajectory
        gt_trajectory = Trajectory()
        for i in range(10):
            pose = Pose(
                timestamp=i * 0.1,
                position=np.array([i * 0.1, 0, 0]),
                quaternion=np.array([1, 0, 0, 0])
            )
            state = TrajectoryState(pose=pose, velocity=np.array([1, 0, 0]))
            gt_trajectory.add_state(state)
        
        # Landmarks
        landmarks = Map()
        for i in range(3):
            landmark = Landmark(id=i, position=np.array([5 + i, 0, 2]))
            landmarks.add_landmark(landmark)
        
        # IMU measurements
        imu_measurements = []
        for i in range(10):
            meas = IMUMeasurement(
                timestamp=i * 0.1,
                accelerometer=np.array([0, 0, 0]),
                gyroscope=np.array([0, 0, 0])
            )
            imu_measurements.append(meas)
        
        # Camera measurements
        camera_measurements = []
        for i in range(5):
            obs = CameraObservation(
                landmark_id=0,
                pixel=ImagePoint(u=320, v=240)
            )
            frame = CameraFrame(
                timestamp=i * 0.2,
                camera_id="cam0",
                observations=[obs]
            )
            camera_measurements.append(frame)
        
        # Create simulation data
        sim_data = SimulationData()
        sim_data.ground_truth_trajectory = gt_trajectory
        sim_data.landmarks = landmarks
        sim_data.imu_measurements = imu_measurements
        sim_data.camera_measurements = camera_measurements
        sim_data.camera_calibrations = {"cam0": camera_calibration}
        
        return sim_data
    
    def test_estimator_runner_initialization(self, camera_calibration):
        """Test EstimatorRunner initialization."""
        runner = EstimatorRunner(camera_calibration, enable_profiling=False)
        
        assert runner.camera_calib == camera_calibration
        assert "EKF" in runner.estimator_configs
        assert "SWBA" in runner.estimator_configs
        assert "SRIF" in runner.estimator_configs
    
    def test_run_single_estimator(self, camera_calibration, simple_simulation_data):
        """Test running a single estimator."""
        runner = EstimatorRunner(camera_calibration, enable_profiling=False)
        
        # Prepare data
        imu_batches = runner._prepare_imu_batches(simple_simulation_data.imu_measurements)
        initial_pose = simple_simulation_data.ground_truth_trajectory.states[0].pose
        
        # Run EKF
        perf = runner.run_estimator(
            "EKF",
            imu_batches,
            simple_simulation_data.camera_measurements,
            simple_simulation_data.ground_truth_trajectory,
            simple_simulation_data.landmarks,
            initial_pose
        )
        
        assert perf.estimator_type == EstimatorType.EKF
        assert perf.trajectory_metrics.ate_rmse >= 0
        assert perf.converged
    
    def test_prepare_imu_batches(self, camera_calibration):
        """Test IMU batching."""
        runner = EstimatorRunner(camera_calibration)
        
        # Create IMU measurements
        measurements = []
        for i in range(20):
            meas = IMUMeasurement(
                timestamp=i * 0.05,  # 50ms intervals
                accelerometer=np.zeros(3),
                gyroscope=np.zeros(3)
            )
            measurements.append(meas)
        
        batches = runner._prepare_imu_batches(measurements)
        
        assert len(batches) > 0
        assert all(isinstance(batch, list) for batch in batches)
        assert all(isinstance(m, IMUMeasurement) for batch in batches for m in batch)
    
    def test_statistical_tests(self, camera_calibration):
        """Test statistical significance tests."""
        runner = EstimatorRunner(camera_calibration)
        
        performances = {
            "EKF": EstimatorPerformance(
                estimator_type=EstimatorType.EKF,
                runtime_ms=100.0,
                peak_memory_mb=50.0,
                trajectory_metrics=TrajectoryMetrics(ate_rmse=0.5)
            ),
            "SWBA": EstimatorPerformance(
                estimator_type=EstimatorType.SWBA,
                runtime_ms=200.0,
                peak_memory_mb=100.0,
                trajectory_metrics=TrajectoryMetrics(ate_rmse=0.3)
            )
        }
        
        tests = runner._perform_statistical_tests(performances)
        
        assert "EKF_vs_SWBA" in tests
        assert "ranking" in tests
        assert tests["ranking"][0] == "SWBA"  # Better ATE


class TestComparisonTable:
    """Test comparison table generation."""
    
    def test_generate_comparison_table(self):
        """Test generating formatted comparison table."""
        performances = {
            "EKF": EstimatorPerformance(
                estimator_type=EstimatorType.EKF,
                runtime_ms=100.5,
                peak_memory_mb=50.3,
                trajectory_metrics=TrajectoryMetrics(
                    ate_rmse=0.5432,
                    ate_mean=0.4321
                )
            )
        }
        
        result = ComparisonResult(performances=performances)
        table = generate_comparison_table(result)
        
        assert "EKF" in table
        assert "100.5" in table or "100" in table  # Runtime
        assert "0.5432" in table or "0.54" in table  # ATE RMSE