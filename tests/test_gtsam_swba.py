"""
Unit tests for GTSAM-based Sliding Window Bundle Adjustment estimator.

Tests window management, marginalization, and optimization within sliding window.
"""

import numpy as np
import pytest
import time

try:
    import gtsam
    GTSAM_AVAILABLE = True
except ImportError:
    GTSAM_AVAILABLE = False
    pytest.skip("GTSAM not installed", allow_module_level=True)

from src.estimation.gtsam_swba_estimator import GtsamSWBAEstimator
from src.estimation.base_estimator import EstimatorConfig, EstimatorResult
from src.common.data_structures import (
    Pose, Map, Landmark,
    PreintegratedIMUData, CameraFrame, CameraObservation, ImagePoint
)


def create_preintegrated_imu(
    dt: float,
    delta_rotation: np.ndarray = None,
    delta_velocity: np.ndarray = None,
    delta_position: np.ndarray = None,
    covariance: np.ndarray = None,
    from_id: int = 0,
    to_id: int = 1
) -> PreintegratedIMUData:
    """Helper to create PreintegratedIMUData with defaults."""
    if delta_rotation is None:
        delta_rotation = np.eye(3)
    if delta_velocity is None:
        delta_velocity = np.zeros(3)
    if delta_position is None:
        delta_position = np.zeros(3)
    if covariance is None:
        covariance = np.eye(15) * 0.01
    
    return PreintegratedIMUData(
        dt=dt,
        delta_rotation=delta_rotation,
        delta_velocity=delta_velocity,
        delta_position=delta_position,
        covariance=covariance,
        from_keyframe_id=from_id,
        to_keyframe_id=to_id,
        num_measurements=10
    )


class TestGtsamSWBAInitialization:
    """Test GTSAM SWBA estimator initialization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = EstimatorConfig(
            estimator_type='gtsam_swba',
            max_landmarks=1000,
            max_iterations=100,
            convergence_threshold=1e-6,
            outlier_threshold=5.0,
            enable_marginalization=True,
            marginalization_window=5,
            verbose=False,
            save_intermediate=False,
            seed=42
        )
        
        # Add SWBA specific configuration
        self.config.swba = {
            'window_size': 5,
            'optimization_iterations': 3,
            'relative_error_tol': 1e-5,
            'absolute_error_tol': 1e-5
        }
    
    def test_swba_initialization(self):
        """Test SWBA can be initialized."""
        estimator = GtsamSWBAEstimator(self.config)
        
        # Check window management structures
        assert estimator.window_size == 5
        assert estimator.optimization_iterations == 3
        assert len(estimator.window_poses) == 0
        assert not estimator.initialized
        assert estimator.num_optimizations == 0
    
    def test_swba_initial_pose(self):
        """Test initializing SWBA with initial pose."""
        estimator = GtsamSWBAEstimator(self.config)
        
        initial_pose = Pose(
            timestamp=0.0,
            position=np.array([1.0, 2.0, 3.0]),
            rotation_matrix=np.eye(3)
        )
        
        estimator.initialize(initial_pose)
        
        # Check initialization
        assert estimator.initialized
        assert len(estimator.window_poses) == 1
        assert 0 in estimator.window_poses
        assert len(estimator.pose_timestamps) == 1
        
        # Check values exist
        assert estimator.window_values.exists(estimator.X(0))
        assert estimator.all_values.exists(estimator.X(0))


class TestGtsamSWBAWindowManagement:
    """Test SWBA window management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = EstimatorConfig(
            estimator_type='gtsam_swba',
            max_landmarks=1000,
            max_iterations=100,
            convergence_threshold=1e-6,
            outlier_threshold=5.0,
            enable_marginalization=True,
            marginalization_window=3,  # Small window for testing
            verbose=False,
            save_intermediate=False,
            seed=42
        )
        
        self.config.swba = {
            'window_size': 3,  # Small window for testing
            'optimization_iterations': 2
        }
        
        self.estimator = GtsamSWBAEstimator(self.config)
        
        # Initialize
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            rotation_matrix=np.eye(3)
        )
        self.estimator.initialize(initial_pose)
    
    def test_window_size_maintained(self):
        """Test that window size is maintained."""
        # Add poses until we exceed window size
        for i in range(5):  # More than window size
            preintegrated = create_preintegrated_imu(
                dt=0.1,
                delta_position=np.array([0.1, 0, 0]),
                from_id=i,
                to_id=i+1
            )
            self.estimator.predict(preintegrated)
        
        # Window should be at max size
        assert len(self.estimator.window_poses) == 3
        
        # Latest poses should be in window
        assert 3 in self.estimator.window_poses
        assert 4 in self.estimator.window_poses
        assert 5 in self.estimator.window_poses
        
        # Oldest poses should not be in window
        assert 0 not in self.estimator.window_poses
        assert 1 not in self.estimator.window_poses
        assert 2 not in self.estimator.window_poses
    
    def test_marginalization_preserves_information(self):
        """Test that marginalized poses are preserved in all_values."""
        # Add poses to fill and exceed window
        for i in range(4):
            preintegrated = create_preintegrated_imu(
                dt=0.1,
                delta_position=np.array([0.1 * (i+1), 0, 0]),
                from_id=i,
                to_id=i+1
            )
            self.estimator.predict(preintegrated)
        
        # Check that all poses exist in all_values
        for i in range(5):  # 0 (initial) + 4 predictions
            assert self.estimator.all_values.exists(self.estimator.X(i))
        
        # But only last 3 in window
        assert len(self.estimator.window_poses) == 3
    
    def test_window_optimization(self):
        """Test that window optimization runs."""
        initial_optimizations = self.estimator.num_optimizations
        
        # Add a pose
        preintegrated = create_preintegrated_imu(
            dt=0.1,
            delta_position=np.array([0.1, 0, 0]),
            from_id=0,
            to_id=1
        )
        self.estimator.predict(preintegrated)
        
        # Trigger optimization via update
        frame = CameraFrame(
            timestamp=0.1,
            camera_id="cam0",
            observations=[]
        )
        self.estimator.update(frame, Map())
        
        # Check optimization counter increased
        assert self.estimator.num_optimizations > initial_optimizations


class TestGtsamSWBATrajectory:
    """Test SWBA trajectory estimation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = EstimatorConfig(
            estimator_type='gtsam_swba',
            max_landmarks=1000,
            max_iterations=100,
            convergence_threshold=1e-6,
            outlier_threshold=5.0,
            enable_marginalization=True,
            marginalization_window=5,
            verbose=False,
            save_intermediate=False,
            seed=42
        )
        
        self.config.swba = {
            'window_size': 5,
            'optimization_iterations': 3
        }
    
    def test_linear_trajectory(self):
        """Test SWBA on linear trajectory."""
        estimator = GtsamSWBAEstimator(self.config)
        
        # Initialize
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            rotation_matrix=np.eye(3)
        )
        estimator.initialize(initial_pose)
        
        # Create linear trajectory
        n_steps = 10
        for i in range(n_steps):
            preintegrated = create_preintegrated_imu(
                dt=0.1,
                delta_velocity=np.array([1.0, 0, 0]) * 0.1,
                delta_position=np.array([0.05, 0, 0]),
                from_id=i,
                to_id=i+1
            )
            estimator.predict(preintegrated)
            
            # Periodically optimize
            if i % 2 == 0:
                frame = CameraFrame(
                    timestamp=(i+1) * 0.1,
                    camera_id="cam0",
                    observations=[]
                )
                estimator.update(frame, Map())
        
        # Get result
        result = estimator.get_result()
        
        # Check trajectory
        assert len(result.trajectory.states) == n_steps + 1
        
        # Check poses are progressively forward
        positions = [state.pose.position for state in result.trajectory.states]
        for i in range(1, len(positions)):
            assert positions[i][0] > positions[i-1][0]  # Moving forward in x
    
    def test_get_result(self):
        """Test get_result returns proper format."""
        estimator = GtsamSWBAEstimator(self.config)
        
        # Initialize
        initial_pose = Pose(
            timestamp=0.0,
            position=np.array([1.0, 2.0, 3.0]),
            rotation_matrix=np.eye(3)
        )
        estimator.initialize(initial_pose)
        
        # Add some predictions
        for i in range(3):
            preintegrated = create_preintegrated_imu(
                dt=0.1,
                delta_position=np.array([0.1, 0, 0]),
                from_id=i,
                to_id=i+1
            )
            estimator.predict(preintegrated)
        
        # Get result
        result = estimator.get_result()
        
        # Check result structure
        assert isinstance(result, EstimatorResult)
        assert result.trajectory is not None
        assert result.landmarks is not None
        assert len(result.trajectory.states) == 4  # Initial + 3 predictions
        assert len(result.landmarks.landmarks) == 0  # No landmarks in simplified SWBA
        
        # Check metadata
        assert result.metadata['window_size'] == len(estimator.window_poses)
        assert result.metadata['estimator_type'] == 'gtsam_swba_imu'
        assert result.metadata['mode'] == 'simplified_imu_only'
    
    def test_reset(self):
        """Test reset functionality."""
        estimator = GtsamSWBAEstimator(self.config)
        
        # Initialize and add poses
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            rotation_matrix=np.eye(3)
        )
        estimator.initialize(initial_pose)
        
        for i in range(3):
            preintegrated = create_preintegrated_imu(
                dt=0.1,
                delta_position=np.array([0.1, 0, 0]),
                from_id=i,
                to_id=i+1
            )
            estimator.predict(preintegrated)
        
        # Reset
        estimator.reset()
        
        # Check state is reset
        assert not estimator.initialized
        assert len(estimator.window_poses) == 0
        assert estimator.pose_count == 0
        assert estimator.num_optimizations == 0


class TestGtsamSWBAPerformance:
    """Test SWBA performance characteristics."""
    
    def test_window_size_limits_computation(self):
        """Test that computation is bounded by window size."""
        config = EstimatorConfig(
            estimator_type='gtsam_swba',
            max_landmarks=1000,
            max_iterations=100,
            convergence_threshold=1e-6,
            outlier_threshold=5.0,
            enable_marginalization=True,
            marginalization_window=5,
            verbose=False,
            save_intermediate=False,
            seed=42
        )
        
        config.swba = {
            'window_size': 5,
            'optimization_iterations': 2
        }
        
        estimator = GtsamSWBAEstimator(config)
        
        # Initialize
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            rotation_matrix=np.eye(3)
        )
        estimator.initialize(initial_pose)
        
        # Add many poses
        optimization_times = []
        for i in range(20):
            preintegrated = create_preintegrated_imu(
                dt=0.1,
                delta_position=np.array([0.1, 0, 0]),
                from_id=i,
                to_id=i+1
            )
            estimator.predict(preintegrated)
            
            # Measure optimization time
            start = time.perf_counter()
            frame = CameraFrame(
                timestamp=(i+1) * 0.1,
                camera_id="cam0",
                observations=[]
            )
            estimator.update(frame, Map())
            end = time.perf_counter()
            
            if i >= 5:  # After window is full
                optimization_times.append(end - start)
        
        # Check that optimization times are relatively constant
        # (bounded by window size, not total trajectory length)
        avg_time = np.mean(optimization_times)
        max_time = np.max(optimization_times)
        
        # Max should not be much larger than average
        assert max_time < avg_time * 3.0  # Allow some variation
    
    def test_marginalization_efficiency(self):
        """Test that marginalization maintains efficiency."""
        config = EstimatorConfig(
            estimator_type='gtsam_swba',
            max_landmarks=1000,
            max_iterations=100,
            convergence_threshold=1e-6,
            outlier_threshold=5.0,
            enable_marginalization=True,
            marginalization_window=3,  # Small window
            verbose=False,
            save_intermediate=False,
            seed=42
        )
        
        config.swba = {
            'window_size': 3,
            'optimization_iterations': 2
        }
        
        estimator = GtsamSWBAEstimator(config)
        
        # Initialize
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            rotation_matrix=np.eye(3)
        )
        estimator.initialize(initial_pose)
        
        # Add many poses
        for i in range(10):
            preintegrated = create_preintegrated_imu(
                dt=0.1,
                delta_position=np.array([0.1, 0, 0]),
                from_id=i,
                to_id=i+1
            )
            estimator.predict(preintegrated)
        
        # Window should still be at max size
        assert len(estimator.window_poses) == 3
        
        # But all poses should be preserved
        result = estimator.get_result()
        assert len(result.trajectory.states) == 11  # Initial + 10 predictions