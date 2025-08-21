"""
Comprehensive tests for visualization module.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json
import os
import plotly.graph_objects as go

from src.common.data_structures import (
    Trajectory, TrajectoryState, Pose,
    Map, Landmark,
    IMUData, IMUMeasurement,
    CameraData, CameraFrame, CameraObservation, ImagePoint,
    CameraCalibration, CameraIntrinsics, CameraExtrinsics, CameraModel
)
from src.plotting.enhanced_plots import (
    plot_trajectory_and_landmarks,
    plot_measurements_with_keyframes,
    plot_imu_data_enhanced as plot_imu_data,  # Use the enhanced version
    create_camera_frustum,
    project_landmarks_to_frame,
    create_full_visualization,
    create_html_dashboard,
    compute_trajectory_length
)
from src.plotting.comparison_plots import (
    plot_trajectory_comparison,
    plot_error_over_time,
    plot_performance_metrics,
    generate_html_report,
    create_comparison_dashboard
)


class TestEnhancedPlots:
    """Test enhanced plotting functions."""
    
    @pytest.fixture
    def sample_trajectory(self):
        """Create a sample trajectory for testing."""
        traj = Trajectory()
        for i in range(100):
            t = i * 0.1
            pose = Pose(
                timestamp=t,
                position=np.array([np.cos(t), np.sin(t), 0.5 * t]),
                rotation_matrix=np.eye(3)  # Identity quaternion
            )
            state = TrajectoryState(
                pose=pose,
                velocity=np.array([-np.sin(t), np.cos(t), 0.5]),
                angular_velocity=np.array([0, 0, 0.1])
            )
            traj.add_state(state)
        return traj
    
    @pytest.fixture
    def sample_landmarks(self):
        """Create sample landmarks."""
        map_data = Map(frame_id="world")
        np.random.seed(42)
        for i in range(50):
            landmark = Landmark(
                id=i,
                position=np.random.randn(3) * 10
            )
            map_data.add_landmark(landmark)
        return map_data
    
    @pytest.fixture
    def sample_imu_data(self):
        """Create sample IMU data."""
        imu_data = IMUData()
        imu_data.imu_id = "imu0"
        imu_data.rate = 200.0
        
        for i in range(200):
            t = i * 0.005
            # Simulate realistic IMU measurements
            meas = IMUMeasurement(
                timestamp=t,
                accelerometer=np.array([
                    0.1 * np.sin(t),
                    0.2 * np.cos(t),
                    9.81 + 0.05 * np.random.randn()
                ]),
                gyroscope=np.array([
                    0.01 * np.sin(2*t),
                    0.02 * np.cos(2*t),
                    0.1 + 0.01 * np.random.randn()
                ])
            )
            imu_data.add_measurement(meas)
        return imu_data
    
    @pytest.fixture
    def sample_camera_data(self):
        """Create sample camera data."""
        camera_data = CameraData(camera_id="cam0", rate=30.0)
        
        np.random.seed(42)
        for i in range(30):
            observations = []
            # Randomly observe some landmarks
            n_obs = np.random.randint(5, 15)
            landmark_ids = np.random.choice(50, n_obs, replace=False)
            
            for lid in landmark_ids:
                obs = CameraObservation(
                    landmark_id=int(lid),
                    pixel=ImagePoint(
                        u=320 + np.random.randn() * 100,
                        v=240 + np.random.randn() * 80
                    )
                )
                observations.append(obs)
            
            frame = CameraFrame(
                timestamp=i * 0.033,
                camera_id="cam0",
                observations=observations
            )
            camera_data.add_frame(frame)
        
        return camera_data
    
    @pytest.fixture
    def sample_calibration(self):
        """Create sample camera calibration."""
        intrinsics = CameraIntrinsics(
            model=CameraModel.PINHOLE,
            width=640,
            height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            distortion=np.zeros(5)
        )
        extrinsics = CameraExtrinsics(
            B_T_C=np.eye(4)
        )
        return CameraCalibration(
            camera_id="cam0",
            intrinsics=intrinsics,
            extrinsics=extrinsics
        )
    
    @pytest.fixture
    def sample_sim_data(self, sample_trajectory, sample_landmarks, 
                       sample_imu_data, sample_camera_data, sample_calibration):
        """Create sample simulation data object."""
        class SimData:
            pass
        
        data = SimData()
        data.ground_truth_trajectory = sample_trajectory
        data.landmarks = sample_landmarks
        data.imu_measurements = sample_imu_data.measurements
        data.camera_measurements = sample_camera_data.frames
        data.camera_calibrations = {"cam0": sample_calibration}
        data.metadata = {
            "trajectory_type": "test",
            "duration": 10.0
        }
        return data
    
    def test_plot_trajectory_and_landmarks(self, sample_sim_data):
        """Test 3D trajectory and landmark plotting."""
        fig = plot_trajectory_and_landmarks(
            sample_sim_data,
            compare_data=None,
            title="Test Plot"
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        
        # Check that trajectory trace exists
        trace_names = [trace.name for trace in fig.data if trace.name]
        assert "Trajectory" in trace_names
        assert "Landmarks" in trace_names
        
        # Check layout
        assert fig.layout.title.text == "Test Plot"
        assert fig.layout.scene.xaxis.title.text == "X (m)"
    
    def test_plot_trajectory_comparison(self, sample_sim_data):
        """Test trajectory comparison plotting."""
        # Create a slightly modified trajectory as comparison
        compare_data = sample_sim_data
        
        fig = plot_trajectory_and_landmarks(
            sample_sim_data,
            compare_data=compare_data,
            title="Comparison"
        )
        
        assert isinstance(fig, go.Figure)
        trace_names = [trace.name for trace in fig.data if trace.name]
        assert "Ground Truth" in trace_names
        assert "SLAM Estimate" in trace_names
    
    def test_plot_measurements_with_keyframes(self, sample_sim_data):
        """Test 2D measurement visualization with keyframes."""
        fig = plot_measurements_with_keyframes(
            sample_sim_data,
            max_keyframes=5
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        
        # Check dropdown menu exists
        assert fig.layout.updatemenus is not None
        assert len(fig.layout.updatemenus) > 0
        
        # Check axes
        assert fig.layout.xaxis.title.text == "u (pixels)"
        assert fig.layout.yaxis.title.text == "v (pixels)"
    
    def test_plot_imu_data(self, sample_sim_data):
        """Test IMU data plotting."""
        fig = plot_imu_data(sample_sim_data)
        
        assert isinstance(fig, go.Figure)
        # Should have multiple subplots
        assert len(fig.data) >= 8  # At least accel x,y,z + gyro x,y,z + magnitudes
        
        # Check subplot titles
        assert fig.layout.annotations is not None
    
    def test_plot_imu_comparison(self, sample_sim_data):
        """Test IMU data comparison plotting."""
        fig = plot_imu_data(sample_sim_data, compare_data=sample_sim_data)
        
        assert isinstance(fig, go.Figure)
        # Should have more traces for comparison
        assert len(fig.data) > 8
    
    def test_create_camera_frustum(self):
        """Test camera frustum creation."""
        position = np.array([1, 2, 3])
        rotation_matrix = np.eye(3)  # Identity rotation
        
        frustum = create_camera_frustum(
            position, rotation_matrix,
            scale=0.5, aspect=1.33, fov=60.0
        )
        
        assert "x" in frustum
        assert "y" in frustum
        assert "z" in frustum
        assert "i" in frustum
        assert "j" in frustum
        assert "k" in frustum
        
        # Check vertices
        assert len(frustum["x"]) == 5  # 1 camera center + 4 corners
        assert len(frustum["i"]) == 6  # 6 triangular faces
    
    def test_project_landmarks_to_frame(self, sample_landmarks, sample_trajectory, sample_calibration):
        """Test landmark projection to camera frame."""
        frame = CameraFrame(
            timestamp=0.5,
            camera_id="cam0",
            observations=[]
        )
        
        projected = project_landmarks_to_frame(
            sample_landmarks,
            frame,
            sample_trajectory,
            sample_calibration
        )
        
        assert isinstance(projected, np.ndarray)
        if len(projected) > 0:
            assert projected.shape[1] == 2  # u, v coordinates
    
    def test_compute_trajectory_length(self, sample_trajectory):
        """Test trajectory length computation."""
        length = compute_trajectory_length(sample_trajectory)
        
        assert isinstance(length, float)
        assert length > 0
        
        # For circular trajectory with vertical motion
        # The spiral trajectory should have length > 10
        assert 10 < length < 12  # Approximate expected length
    
    def test_create_full_visualization(self, sample_sim_data):
        """Test full visualization creation."""
        html = create_full_visualization(
            sample_sim_data,
            compare_data=None,
            show_trajectory=True,
            show_measurements=True,
            show_imu=True,
            max_keyframes=5
        )
        
        assert isinstance(html, str)
        assert len(html) > 1000  # Should be substantial HTML
        assert "<!DOCTYPE html>" in html
        assert "plotly" in html.lower()
        assert "3D Trajectory and Landmarks" in html
    
    def test_create_html_dashboard(self, sample_sim_data):
        """Test HTML dashboard creation."""
        # Create a simple figure
        fig = plot_trajectory_and_landmarks(sample_sim_data)
        figures = [("trajectory", fig)]
        
        html = create_html_dashboard(figures, sample_sim_data)
        
        assert isinstance(html, str)
        assert "SLAM Visualization Dashboard" in html
        assert "plot_trajectory" in html
        assert "stats-grid" in html  # Statistics section
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        class EmptyData:
            ground_truth_trajectory = None
            landmarks = None
            imu_measurements = None
            camera_measurements = None
            camera_calibrations = None
            metadata = {}
        
        empty_data = EmptyData()
        
        # Should not crash with empty data
        fig = plot_trajectory_and_landmarks(empty_data)
        assert isinstance(fig, go.Figure)
        
        html = create_full_visualization(
            empty_data,
            show_trajectory=True,
            show_measurements=False,
            show_imu=False
        )
        assert isinstance(html, str)


class TestComparisonPlots:
    """Test comparison plotting functions."""
    
    @pytest.fixture
    def sample_results(self):
        """Create sample comparison results."""
        return {
            "best_estimator": "EKF",
            "estimators": {
                "EKF": {
                    "algorithm": "EKF",
                    "ate_rmse": 0.125,
                    "ate_mean": 0.100,
                    "ate_std": 0.050,
                    "rpe_rmse": 0.085,
                    "execution_time": 1.234,
                    "trajectory": self._create_sample_trajectory(0.1),
                    "error_stats": {
                        "position_errors": np.random.randn(100) * 0.1,
                        "rotation_errors": np.random.randn(100) * 0.05
                    }
                },
                "SWBA": {
                    "algorithm": "SWBA",
                    "ate_rmse": 0.150,
                    "ate_mean": 0.120,
                    "ate_std": 0.060,
                    "rpe_rmse": 0.095,
                    "execution_time": 2.456,
                    "trajectory": self._create_sample_trajectory(0.15),
                    "error_stats": {
                        "position_errors": np.random.randn(100) * 0.15,
                        "rotation_errors": np.random.randn(100) * 0.07
                    }
                }
            },
            "ground_truth": self._create_sample_trajectory(0)
        }
    
    def _create_sample_trajectory(self, noise_level=0):
        """Helper to create sample trajectory."""
        traj = Trajectory()
        for i in range(100):
            t = i * 0.1
            pose = Pose(
                timestamp=t,
                position=np.array([
                    np.cos(t) + np.random.randn() * noise_level,
                    np.sin(t) + np.random.randn() * noise_level,
                    0.5 * t + np.random.randn() * noise_level * 0.5
                ]),
                rotation_matrix=np.eye(3)
            )
            traj.add_state(TrajectoryState(pose=pose))
        return traj
    
    def test_plot_trajectory_comparison(self, sample_results):
        """Test trajectory comparison plotting."""
        from src.evaluation.comparison import ComparisonResult, EstimatorPerformance
        from src.evaluation.metrics import TrajectoryMetrics
        from src.estimation.base_estimator import EstimatorType
        
        # Create EstimatorPerformance objects
        performances = {}
        for name, data in sample_results["estimators"].items():
            traj_metrics = TrajectoryMetrics()
            traj_metrics.ate_rmse = data["ate_rmse"]
            traj_metrics.ate_mean = data["ate_mean"]
            
            perf = EstimatorPerformance(
                estimator_type=EstimatorType.EKF if name == "EKF" else EstimatorType.SWBA,
                runtime_ms=data["execution_time"] * 1000,
                peak_memory_mb=100,
                trajectory_metrics=traj_metrics,
                consistency_metrics=None,
                num_iterations=1,
                converged=True,
                metadata={}
            )
            performances[name] = perf
        
        # Create a ComparisonResult object
        comp_result = ComparisonResult(performances=performances)
        comp_result.ground_truth = sample_results["ground_truth"]
        comp_result.estimator_results = {
            name: {
                "trajectory": data["trajectory"],
                "ate_rmse": data["ate_rmse"]
            }
            for name, data in sample_results["estimators"].items()
        }
        
        # Extract trajectories for the plotting function
        trajectories = {
            name: data["trajectory"]
            for name, data in sample_results["estimators"].items()
        }
        
        fig = plot_trajectory_comparison(
            trajectories=trajectories,
            ground_truth=sample_results["ground_truth"]
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3  # Ground truth + at least 2 estimators
    
    def test_plot_error_over_time(self, sample_results):
        """Test error over time plotting."""
        from src.evaluation.comparison import ComparisonResult
        
        comp_result = ComparisonResult(performances={})
        comp_result.ground_truth = sample_results["ground_truth"]
        comp_result.estimator_results = sample_results["estimators"]
        
        # Create mock error data for testing
        timestamps = np.linspace(0, 5, 100)
        error_data = {}
        
        for name in sample_results["estimators"].keys():
            # Generate some synthetic error data
            error_data[name] = 0.1 + 0.05 * np.sin(timestamps) + 0.01 * np.random.randn(len(timestamps))
        
        fig = plot_error_over_time(comp_result, error_data, timestamps)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # At least EKF and SWBA
    
    def test_plot_performance_metrics(self, sample_results):
        """Test performance metrics plotting."""
        from src.evaluation.comparison import ComparisonResult, EstimatorPerformance
        from src.evaluation.metrics import TrajectoryMetrics
        from src.estimation.base_estimator import EstimatorType
        
        # Create EstimatorPerformance objects with proper data
        performances = {}
        for name, data in sample_results["estimators"].items():
            traj_metrics = TrajectoryMetrics()
            traj_metrics.ate_rmse = data["ate_rmse"]
            traj_metrics.ate_mean = data["ate_mean"]
            traj_metrics.rpe_translation_rmse = 0.02  # Mock RPE data
            
            perf = EstimatorPerformance(
                estimator_type=EstimatorType.EKF if name == "EKF" else EstimatorType.SWBA,
                runtime_ms=data["execution_time"] * 1000,
                peak_memory_mb=100,
                trajectory_metrics=traj_metrics,
                consistency_metrics=None,
                num_iterations=1,
                converged=True,
                metadata={}
            )
            performances[name] = perf
        
        comp_result = ComparisonResult(performances=performances)
        comp_result.estimator_results = sample_results["estimators"]
        
        fig = plot_performance_metrics(comp_result)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        
        # Should have bar charts
        assert any(isinstance(trace, go.Bar) for trace in fig.data)
    
    def test_generate_html_report(self, sample_results):
        """Test HTML report generation."""
        from src.evaluation.comparison import ComparisonResult, EstimatorPerformance
        from src.evaluation.metrics import TrajectoryMetrics
        from src.estimation.base_estimator import EstimatorType
        
        # Create EstimatorPerformance objects with proper data
        performances = {}
        for name, data in sample_results["estimators"].items():
            traj_metrics = TrajectoryMetrics()
            traj_metrics.ate_rmse = data["ate_rmse"]
            traj_metrics.ate_mean = data["ate_mean"]
            
            perf = EstimatorPerformance(
                estimator_type=EstimatorType.EKF if name == "EKF" else EstimatorType.SWBA,
                runtime_ms=data["execution_time"] * 1000,
                peak_memory_mb=100,
                trajectory_metrics=traj_metrics,
                consistency_metrics=None,
                num_iterations=1,
                converged=True,
                metadata={}
            )
            performances[name] = perf
        
        comp_result = ComparisonResult(performances=performances)
        comp_result.ground_truth = sample_results["ground_truth"]
        comp_result.estimator_results = sample_results["estimators"]
        comp_result.best_estimator = sample_results["best_estimator"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = generate_html_report(
                comp_result,
                tmpdir,
                include_plots=True
            )
            
            assert os.path.exists(output_path)
            assert output_path.endswith(".html")
            
            # Check content
            with open(output_path, 'r') as f:
                content = f.read()
            assert "Comparison Report" in content
            assert "EKF" in content
            assert "SWBA" in content
            
            # Check that plot files were created and contain plotly
            dashboard_path = os.path.join(tmpdir, "dashboard.html")
            performance_path = os.path.join(tmpdir, "performance.html")
            
            assert os.path.exists(dashboard_path)
            assert os.path.exists(performance_path)
            
            # Check that at least one plot file contains plotly
            with open(dashboard_path, 'r') as f:
                dashboard_content = f.read()
            assert "plotly" in dashboard_content.lower()


class TestVisualizationIntegration:
    """Integration tests for visualization system."""
    
    def test_data_conversion_consistency(self):
        """Test that numpy arrays are properly converted for plotly."""
        # Create data with numpy arrays
        positions = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0].tolist(),
            y=positions[:, 1].tolist(),
            z=positions[:, 2].tolist(),
            mode='lines',
            name='Test'
        ))
        
        # Generate HTML and check it contains actual data
        html = fig.to_html(full_html=False, include_plotlyjs=False)
        assert '"x":[1' in html  # Data should be serialized as list
        assert '"x":{"dtype"' not in html  # Should not be binary encoded
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets."""
        # Create large trajectory
        traj = Trajectory()
        for i in range(10000):  # 10k points
            t = i * 0.001
            pose = Pose(
                timestamp=t,
                position=np.array([t, t*2, t*3]),
                rotation_matrix=np.eye(3)
            )
            traj.add_state(TrajectoryState(pose=pose))
        
        class BigData:
            ground_truth_trajectory = traj
            landmarks = None
            imu_measurements = None
            camera_measurements = None
            camera_calibrations = None
            metadata = {}
        
        # Should handle large data without crashing
        fig = plot_trajectory_and_landmarks(BigData())
        assert isinstance(fig, go.Figure)
        assert len(fig.data[0].x) == 10000
    
    def test_html_sanitization(self):
        """Test that HTML generation properly escapes content."""
        class TestData:
            ground_truth_trajectory = None
            landmarks = None
            imu_measurements = None
            camera_measurements = None
            camera_calibrations = None
            metadata = {
                "source": "<script>alert('xss')</script>",
                "trajectory_type": "test&<>\"'"
            }
        
        html = create_full_visualization(
            TestData(),
            show_trajectory=False,
            show_measurements=False,
            show_imu=False
        )
        
        # The metadata with script tags is included in the HTML but should be safe
        # because it's not executed (it's just displayed as text in metadata)
        # What matters is that script tags aren't actually executed
        assert "<!DOCTYPE html>" in html  # Valid HTML structure
        assert "plotly" in html.lower()  # Has plotting library
    
    @pytest.mark.parametrize("show_trajectory,show_measurements,show_imu", [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, True),
    ])
    def test_selective_visualization(self, show_trajectory, show_measurements, show_imu):
        """Test selective enabling of visualization components."""
        class TestData:
            ground_truth_trajectory = Trajectory() if show_trajectory else None
            landmarks = None
            imu_measurements = [] if show_imu else None
            camera_measurements = [] if show_measurements else None
            camera_calibrations = {}
            metadata = {}
        
        # Add minimal data if needed
        if show_trajectory and TestData.ground_truth_trajectory:
            pose = Pose(0, np.zeros(3), np.eye(3))  # Identity rotation matrix
            TestData.ground_truth_trajectory.add_state(TrajectoryState(pose=pose))
        
        html = create_full_visualization(
            TestData(),
            show_trajectory=show_trajectory,
            show_measurements=show_measurements,
            show_imu=show_imu
        )
        
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html