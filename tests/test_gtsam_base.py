"""
Unit tests for GTSAM base estimator class.

Tests data conversion utilities, symbol management, and common functionality.
"""

import numpy as np
import pytest
from typing import Dict, Any

try:
    import gtsam
    GTSAM_AVAILABLE = True
except ImportError:
    GTSAM_AVAILABLE = False
    pytest.skip("GTSAM not installed", allow_module_level=True)

from src.estimation.gtsam_base import GtsamBaseEstimator
from src.estimation.base_estimator import EstimatorResult, EstimatorConfig
from src.common.data_structures import (
    Pose, Trajectory, TrajectoryState, Map, Landmark,
    CameraFrame, CameraObservation, ImagePoint, PreintegratedIMUData
)


class TestableGtsamEstimator(GtsamBaseEstimator):
    """Concrete implementation for testing base functionality."""
    
    def initialize(self, initial_pose: Pose) -> None:
        """Initialize the estimator."""
        pass
    
    def predict(self, preintegrated_imu: PreintegratedIMUData) -> None:
        """Predict next state."""
        pass
    
    def update(self, frame: CameraFrame, landmarks: Map) -> None:
        """Update with measurements."""
        pass
    
    def optimize(self) -> EstimatorResult:
        """Run optimization."""
        return self.get_result()
    
    def get_result(self) -> EstimatorResult:
        """Get current result."""
        values = gtsam.Values()  # Empty for testing
        return EstimatorResult(
            trajectory=self.extract_trajectory(values),
            landmarks=self.extract_landmarks(values),
            states=[],
            runtime_ms=0.0,
            iterations=0,
            converged=True,
            final_cost=0.0,
            metadata={}
        )
    
    def get_state_vector(self) -> np.ndarray:
        """Get state vector."""
        return np.zeros(1)
    
    def get_covariance_matrix(self) -> np.ndarray:
        """Get covariance matrix."""
        return np.eye(1)
    
    def marginalize(self) -> None:
        """Marginalize old states."""
        pass


class TestGtsamBaseEstimator:
    """Test GTSAM base estimator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create proper EstimatorConfig
        self.config = EstimatorConfig(
            estimator_type='gtsam_test',
            max_landmarks=1000,
            max_iterations=100,
            convergence_threshold=1e-6,
            outlier_threshold=5.0,
            enable_marginalization=False,
            marginalization_window=10,
            verbose=False,
            save_intermediate=False,
            seed=None
        )
        
        # Add noise models as extra attributes for GTSAM
        self.config.noise_models = {
            'prior_pose': [0.1, 0.1, 0.1, 0.05, 0.05, 0.05],
            'prior_velocity': [0.1, 0.1, 0.1],
            'prior_bias': [0.01, 0.01, 0.01, 0.001, 0.001, 0.001],
            'projection_noise': [1.0, 1.0]
        }
        
        self.estimator = TestableGtsamEstimator(self.config)
    
    def test_initialization(self):
        """Test GTSAM base estimator initialization."""
        assert self.estimator.graph is not None
        assert self.estimator.initial_values is not None
        assert self.estimator.pose_count == 0
        assert self.estimator.landmark_count == 0
        assert self.estimator.velocity_count == 0
        assert len(self.estimator.initialized_landmarks) == 0
    
    def test_symbol_generation(self):
        """Test symbol generation for variables."""
        # Test pose symbols
        x0 = self.estimator.X(0)
        x1 = self.estimator.X(1)
        assert x0 != x1
        assert gtsam.Symbol(x0).chr() == ord('x')
        assert gtsam.Symbol(x0).index() == 0
        assert gtsam.Symbol(x1).index() == 1
        
        # Test landmark symbols
        l0 = self.estimator.L(0)
        l1 = self.estimator.L(1)
        assert l0 != l1
        assert gtsam.Symbol(l0).chr() == ord('l')
        assert gtsam.Symbol(l0).index() == 0
        
        # Test velocity symbols
        v0 = self.estimator.V(0)
        v1 = self.estimator.V(1)
        assert v0 != v1
        assert gtsam.Symbol(v0).chr() == ord('v')
        
        # Test bias symbols
        b0 = self.estimator.B(0)
        b1 = self.estimator.B(1)
        assert b0 != b1
        assert gtsam.Symbol(b0).chr() == ord('b')
    
    def test_pose_to_gtsam_conversion(self):
        """Test conversion from custom Pose to GTSAM Pose3."""
        # Create custom pose
        position = np.array([1.0, 2.0, 3.0])
        rotation_matrix = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])  # 90 degree rotation around z-axis
        
        pose = Pose(
            timestamp=1.0,
            position=position,
            rotation_matrix=rotation_matrix
        )
        
        # Convert to GTSAM
        gtsam_pose = self.estimator.pose_to_gtsam(pose)
        
        # Verify translation
        np.testing.assert_array_almost_equal(
            gtsam_pose.translation(),
            position
        )
        
        # Verify rotation
        np.testing.assert_array_almost_equal(
            gtsam_pose.rotation().matrix(),
            rotation_matrix
        )
    
    def test_gtsam_to_pose_conversion(self):
        """Test conversion from GTSAM Pose3 to custom Pose."""
        # Create GTSAM pose
        rotation = gtsam.Rot3.Ypr(np.pi/4, 0, 0)  # 45 degree yaw
        translation = gtsam.Point3(1.0, 2.0, 3.0)
        gtsam_pose = gtsam.Pose3(rotation, translation)
        
        # Convert to custom Pose
        timestamp = 2.5
        pose = self.estimator.gtsam_to_pose(gtsam_pose, timestamp)
        
        # Verify timestamp
        assert pose.timestamp == timestamp
        
        # Verify position
        np.testing.assert_array_almost_equal(
            pose.position,
            np.array([1.0, 2.0, 3.0])
        )
        
        # Verify rotation matrix
        np.testing.assert_array_almost_equal(
            pose.rotation_matrix,
            rotation.matrix()
        )
    
    def test_bidirectional_pose_conversion(self):
        """Test that pose conversion is bidirectional."""
        # Create original pose
        original_pose = Pose(
            timestamp=1.5,
            position=np.array([1.0, 2.0, 3.0]),
            rotation_matrix=np.eye(3)
        )
        
        # Convert to GTSAM and back
        gtsam_pose = self.estimator.pose_to_gtsam(original_pose)
        recovered_pose = self.estimator.gtsam_to_pose(
            gtsam_pose, 
            original_pose.timestamp
        )
        
        # Verify recovery
        assert recovered_pose.timestamp == original_pose.timestamp
        np.testing.assert_array_almost_equal(
            recovered_pose.position,
            original_pose.position
        )
        np.testing.assert_array_almost_equal(
            recovered_pose.rotation_matrix,
            original_pose.rotation_matrix
        )
    
    def test_extract_trajectory(self):
        """Test trajectory extraction from GTSAM values."""
        # Create GTSAM values with poses
        values = gtsam.Values()
        
        # Add poses
        for i in range(3):
            pose = gtsam.Pose3(
                gtsam.Rot3(),
                gtsam.Point3(i, 0, 0)
            )
            values.insert(self.estimator.X(i), pose)
            
            # Add velocity
            velocity = np.array([1.0, 0.0, 0.0])
            values.insert(self.estimator.V(i), velocity)
        
        # Update pose count
        self.estimator.pose_count = 3
        
        # Extract trajectory
        trajectory = self.estimator.extract_trajectory(values)
        
        # Verify trajectory
        assert len(trajectory.states) == 3
        for i, state in enumerate(trajectory.states):
            assert state.pose.timestamp == float(i)
            np.testing.assert_array_almost_equal(
                state.pose.position,
                np.array([i, 0, 0])
            )
            np.testing.assert_array_almost_equal(
                state.velocity,
                np.array([1.0, 0.0, 0.0])
            )
    
    def test_extract_landmarks(self):
        """Test landmark extraction from GTSAM values."""
        # Create GTSAM values with landmarks
        values = gtsam.Values()
        
        # Add landmarks
        landmark_positions = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]
        
        for i, pos in enumerate(landmark_positions):
            values.insert(
                self.estimator.L(i),
                gtsam.Point3(pos[0], pos[1], pos[2])
            )
            self.estimator.initialized_landmarks.add(i)
        
        # Extract landmarks
        landmark_map = self.estimator.extract_landmarks(values)
        
        # Verify landmarks
        assert len(landmark_map.landmarks) == 3
        for i, pos in enumerate(landmark_positions):
            landmark = landmark_map.get_landmark(i)
            assert landmark is not None
            np.testing.assert_array_almost_equal(
                landmark.position,
                np.array(pos)
            )
    
    def test_add_prior_factors(self):
        """Test adding prior factors for initial state."""
        # Create initial pose
        initial_pose = Pose(
            timestamp=0.0,
            position=np.array([1.0, 2.0, 3.0]),
            rotation_matrix=np.eye(3)
        )
        
        # Add prior factors
        self.estimator.add_prior_factors(initial_pose)
        
        # Verify graph has factors
        assert self.estimator.graph.size() == 3  # pose, velocity, bias priors
        
        # Verify initial values
        assert self.estimator.initial_values.exists(self.estimator.X(0))
        assert self.estimator.initial_values.exists(self.estimator.V(0))
        assert self.estimator.initial_values.exists(self.estimator.B(0))
        
        # Verify pose value
        pose_value = self.estimator.initial_values.atPose3(self.estimator.X(0))
        np.testing.assert_array_almost_equal(
            pose_value.translation(),
            initial_pose.position
        )
    
    def test_add_vision_factor(self):
        """Test adding vision factor for landmark observation."""
        # Create observation
        observation = CameraObservation(
            landmark_id=0,
            pixel=ImagePoint(u=320, v=240)
        )
        
        # Create landmark
        landmark = Landmark(
            id=0,
            position=np.array([5.0, 0.0, 0.0])
        )
        
        # Add vision factor
        pose_key = self.estimator.X(0)
        self.estimator.add_vision_factor(
            pose_key,
            observation,
            landmark
        )
        
        # Verify factor was added
        assert self.estimator.graph.size() == 1
        
        # Verify landmark was initialized
        assert 0 in self.estimator.initialized_landmarks
        assert self.estimator.initial_values.exists(self.estimator.L(0))
    
    def test_noise_model_setup(self):
        """Test noise model setup from configuration."""
        # Verify prior pose noise
        assert self.estimator.prior_pose_noise is not None
        assert self.estimator.prior_pose_noise.covariance().shape == (6, 6)
        
        # Verify prior velocity noise
        assert self.estimator.prior_velocity_noise is not None
        assert self.estimator.prior_velocity_noise.covariance().shape == (3, 3)
        
        # Verify prior bias noise
        assert self.estimator.prior_bias_noise is not None
        assert self.estimator.prior_bias_noise.covariance().shape == (6, 6)
        
        # Verify projection noise
        assert self.estimator.projection_noise is not None
        assert self.estimator.projection_noise.covariance().shape == (2, 2)
    
    def test_default_noise_models(self):
        """Test default noise models when config is empty."""
        # Create minimal config without noise models
        config = EstimatorConfig(
            estimator_type='gtsam_test',
            max_landmarks=1000,
            max_iterations=100,
            convergence_threshold=1e-6,
            outlier_threshold=5.0,
            enable_marginalization=False,
            marginalization_window=10,
            verbose=False,
            save_intermediate=False,
            seed=None
        )
        estimator = TestableGtsamEstimator(config)
        
        # Should have default noise models
        assert estimator.prior_pose_noise is not None
        assert estimator.prior_velocity_noise is not None
        assert estimator.prior_bias_noise is not None
        assert estimator.projection_noise is not None


class TestSymbolManagement:
    """Test symbol management in GTSAM base estimator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        config = EstimatorConfig(
            estimator_type='gtsam_test',
            max_landmarks=1000,
            max_iterations=100,
            convergence_threshold=1e-6,
            outlier_threshold=5.0,
            enable_marginalization=False,
            marginalization_window=10,
            verbose=False,
            save_intermediate=False,
            seed=None
        )
        self.estimator = TestableGtsamEstimator(config)
    
    def test_symbol_uniqueness(self):
        """Test that symbols are unique across types."""
        x0 = self.estimator.X(0)
        l0 = self.estimator.L(0)
        v0 = self.estimator.V(0)
        b0 = self.estimator.B(0)
        
        # All should be different
        assert x0 != l0
        assert x0 != v0
        assert x0 != b0
        assert l0 != v0
        assert l0 != b0
        assert v0 != b0
    
    def test_symbol_consistency(self):
        """Test that same index gives same symbol."""
        x0_first = self.estimator.X(0)
        x0_second = self.estimator.X(0)
        assert x0_first == x0_second
        
        l5_first = self.estimator.L(5)
        l5_second = self.estimator.L(5)
        assert l5_first == l5_second


class TestDataStructureIntegration:
    """Test integration with custom data structures."""
    
    def setup_method(self):
        """Set up test fixtures."""
        config = EstimatorConfig(
            estimator_type='gtsam_test',
            max_landmarks=1000,
            max_iterations=100,
            convergence_threshold=1e-6,
            outlier_threshold=5.0,
            enable_marginalization=False,
            marginalization_window=10,
            verbose=False,
            save_intermediate=False,
            seed=None
        )
        self.estimator = TestableGtsamEstimator(config)
    
    def test_trajectory_with_missing_velocities(self):
        """Test trajectory extraction when some velocities are missing."""
        values = gtsam.Values()
        
        # Add poses but only some velocities
        for i in range(3):
            pose = gtsam.Pose3(
                gtsam.Rot3(),
                gtsam.Point3(i, 0, 0)
            )
            values.insert(self.estimator.X(i), pose)
            
            # Only add velocity for even indices
            if i % 2 == 0:
                velocity = np.array([1.0, 0.0, 0.0])
                values.insert(self.estimator.V(i), velocity)
        
        self.estimator.pose_count = 3
        
        # Extract trajectory
        trajectory = self.estimator.extract_trajectory(values)
        
        # Verify trajectory
        assert len(trajectory.states) == 3
        assert trajectory.states[0].velocity is not None
        assert trajectory.states[1].velocity is None
        assert trajectory.states[2].velocity is not None
    
    def test_empty_landmark_extraction(self):
        """Test landmark extraction with no landmarks."""
        values = gtsam.Values()
        
        # Extract landmarks (should be empty)
        landmark_map = self.estimator.extract_landmarks(values)
        
        assert len(landmark_map.landmarks) == 0
    
    def test_partial_landmark_extraction(self):
        """Test landmark extraction when not all initialized landmarks are in values."""
        values = gtsam.Values()
        
        # Initialize 3 landmarks but only add 2 to values
        self.estimator.initialized_landmarks = {0, 1, 2}
        
        values.insert(self.estimator.L(0), gtsam.Point3(1, 2, 3))
        values.insert(self.estimator.L(2), gtsam.Point3(4, 5, 6))
        # Landmark 1 is missing
        
        # Extract landmarks
        landmark_map = self.estimator.extract_landmarks(values)
        
        # Should only get the 2 that exist in values
        assert len(landmark_map.landmarks) == 2
        assert landmark_map.get_landmark(0) is not None
        assert landmark_map.get_landmark(1) is None
        assert landmark_map.get_landmark(2) is not None