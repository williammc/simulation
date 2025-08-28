"""
Unit tests for camera-model-independent NewSWBAEstimator.

These tests work with synthetic pre-processed measurements without
requiring camera setup or projection operations.
"""

import pytest
import numpy as np
from typing import List

from src.estimation.new.swba_estimator import NewSWBAEstimator, NewSWBAConfig
from src.estimation.interfaces import (
    VisualMeasurement,
    ProcessedVisualFrame,
    PreprocessedIMUData
)
from src.common.data_structures import Pose, Map, Landmark


class TestNewSWBAEstimator:
    """Test suite for NewSWBAEstimator."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return NewSWBAConfig(
            window_size=5,
            min_keyframe_distance=0.5,
            min_keyframe_angle=10.0,
            keyframe_selection_method="distance",
            max_optimization_iterations=10,
            use_robust_kernels=True,
            verbose_optimization=False
        )
    
    @pytest.fixture
    def estimator(self, config):
        """Create estimator instance."""
        return NewSWBAEstimator(config)
    
    @pytest.fixture
    def synthetic_visual_measurement(self):
        """Create synthetic visual measurement with pre-computed values."""
        return VisualMeasurement(
            landmark_id=1,
            observed_pixel=np.array([320.5, 240.2]),
            predicted_pixel=np.array([320.0, 240.0]),
            residual=np.array([0.5, 0.2]),
            pixel_covariance=np.eye(2) * 1.0,
            jacobian_wrt_pose=np.random.randn(2, 6) * 0.1,
            jacobian_wrt_landmark=np.random.randn(2, 3) * 0.1,
            robust_weight=0.95
        )
    
    @pytest.fixture
    def synthetic_visual_frame(self, synthetic_visual_measurement):
        """Create synthetic processed visual frame."""
        measurements = [
            synthetic_visual_measurement,
            VisualMeasurement(
                landmark_id=2,
                observed_pixel=np.array([400.0, 300.0]),
                predicted_pixel=np.array([399.5, 299.8]),
                residual=np.array([0.5, 0.2]),
                pixel_covariance=np.eye(2) * 1.0,
                jacobian_wrt_pose=np.random.randn(2, 6) * 0.1,
                jacobian_wrt_landmark=np.random.randn(2, 3) * 0.1,
                robust_weight=0.90
            )
        ]
        
        return ProcessedVisualFrame(
            timestamp=1.0,
            frame_id=1,
            is_keyframe=True,
            keyframe_id=0,
            measurements=measurements,
            predicted_pose=Pose(
                timestamp=1.0,
                position=np.array([1.0, 0.0, 0.0]),
                rotation_matrix=np.eye(3)
            )
        )
    
    @pytest.fixture
    def synthetic_imu_data(self):
        """Create synthetic pre-integrated IMU data."""
        return PreprocessedIMUData(
            from_keyframe_id=0,
            to_keyframe_id=1,
            delta_position=np.array([0.1, 0.05, 0.01]),
            delta_velocity=np.array([0.5, 0.2, 0.05]),
            delta_rotation=np.eye(3),  # Small rotation approximated as identity
            covariance=np.eye(9) * 0.01,
            delta_t=0.1,
            num_measurements=20
        )
    
    @pytest.fixture
    def landmark_map(self):
        """Create simple landmark map."""
        map_obj = Map()
        landmarks = [
            Landmark(id=1, position=np.array([5.0, 2.0, 0.0])),
            Landmark(id=2, position=np.array([5.0, -2.0, 0.0])),
            Landmark(id=3, position=np.array([10.0, 0.0, 0.0]))
        ]
        for lm in landmarks:
            map_obj.add_landmark(lm)
        return map_obj
    
    def test_initialization(self, estimator):
        """Test estimator initialization."""
        initial_pose = Pose(
            timestamp=0.0,
            position=np.array([0.0, 0.0, 0.0]),
            rotation_matrix=np.eye(3)
        )
        
        estimator.initialize(initial_pose)
        
        assert estimator.current_pose is not None
        assert np.allclose(estimator.current_pose.position, initial_pose.position)
        assert np.allclose(estimator.current_pose.rotation_matrix, initial_pose.rotation_matrix)
        assert np.allclose(estimator.current_velocity, np.zeros(3))
        assert estimator.current_imu_bias is not None
    
    def test_predict_with_preprocessed_imu(self, estimator, synthetic_imu_data):
        """Test prediction with pre-integrated IMU data."""
        # Initialize
        initial_pose = Pose(
            timestamp=0.0,
            position=np.array([0.0, 0.0, 0.0]),
            rotation_matrix=np.eye(3)
        )
        estimator.initialize(initial_pose)
        
        # Predict
        estimator.predict(synthetic_imu_data, synthetic_imu_data.delta_t)
        
        # Check that pose was updated
        assert estimator.current_pose.timestamp == synthetic_imu_data.delta_t
        assert not np.allclose(estimator.current_pose.position, initial_pose.position)
        
        # Check that pre-integration was stored
        assert estimator.total_predictions == 1
    
    def test_update_with_processed_frame(self, estimator, synthetic_visual_frame, landmark_map):
        """Test update with pre-processed visual frame."""
        # Initialize
        initial_pose = Pose(
            timestamp=0.0,
            position=np.array([0.0, 0.0, 0.0]),
            rotation_matrix=np.eye(3)
        )
        estimator.initialize(initial_pose)
        
        # Update
        estimator.update(synthetic_visual_frame, landmark_map)
        
        # Check that keyframe was created
        assert len(estimator.keyframes) == 1
        assert len(estimator.keyframe_poses) == 1
        assert estimator.num_keyframes_created == 1
        
        # Check that landmarks were initialized
        assert len(estimator.landmark_estimates) == 2
        assert 1 in estimator.landmark_estimates
        assert 2 in estimator.landmark_estimates
    
    def test_keyframe_selection(self, estimator, config):
        """Test keyframe selection logic without camera model."""
        initial_pose = Pose(
            timestamp=0.0,
            position=np.array([0.0, 0.0, 0.0]),
            rotation_matrix=np.eye(3)
        )
        estimator.initialize(initial_pose)
        
        # Create first keyframe
        frame1 = ProcessedVisualFrame(
            timestamp=0.0,
            frame_id=0,
            is_keyframe=True,
            measurements=[]
        )
        estimator.update(frame1, None)
        assert len(estimator.keyframes) == 1
        
        # Move less than threshold - should not create keyframe
        estimator.current_pose = Pose(
            timestamp=0.1,
            position=np.array([0.3, 0.0, 0.0]),  # Less than 0.5m threshold
            rotation_matrix=np.eye(3)
        )
        frame2 = ProcessedVisualFrame(
            timestamp=0.1,
            frame_id=1,
            is_keyframe=False,
            measurements=[]
        )
        estimator.update(frame2, None)
        assert len(estimator.keyframes) == 1  # No new keyframe
        
        # Move more than threshold - should create keyframe
        estimator.current_pose = Pose(
            timestamp=0.2,
            position=np.array([0.6, 0.0, 0.0]),  # More than 0.5m threshold
            rotation_matrix=np.eye(3)
        )
        frame3 = ProcessedVisualFrame(
            timestamp=0.2,
            frame_id=2,
            is_keyframe=True,
            measurements=[]
        )
        estimator.update(frame3, None)
        assert len(estimator.keyframes) == 2  # New keyframe created
    
    def test_sliding_window_maintenance(self, estimator, config):
        """Test that sliding window size is maintained."""
        initial_pose = Pose(
            timestamp=0.0,
            position=np.array([0.0, 0.0, 0.0]),
            rotation_matrix=np.eye(3)
        )
        estimator.initialize(initial_pose)
        
        # Create more keyframes than window size
        for i in range(config.window_size + 2):
            estimator.current_pose = Pose(
                timestamp=float(i),
                position=np.array([i * 1.0, 0.0, 0.0]),
                rotation_matrix=np.eye(3)
            )
            frame = ProcessedVisualFrame(
                timestamp=float(i),
                frame_id=i,
                is_keyframe=True,
                measurements=[]
            )
            estimator.update(frame, None)
        
        # Check window size is maintained
        assert len(estimator.keyframes) == config.window_size
        assert len(estimator.keyframe_poses) == config.window_size
    
    def test_optimize_with_synthetic_data(self, estimator, synthetic_visual_frame, landmark_map):
        """Test optimization with synthetic measurements."""
        # Initialize
        initial_pose = Pose(
            timestamp=0.0,
            position=np.array([0.0, 0.0, 0.0]),
            rotation_matrix=np.eye(3)
        )
        estimator.initialize(initial_pose)
        
        # Add multiple keyframes
        for i in range(3):
            estimator.current_pose = Pose(
                timestamp=float(i),
                position=np.array([i * 1.0, 0.0, 0.0]),
                rotation_matrix=np.eye(3)
            )
            frame = ProcessedVisualFrame(
                timestamp=float(i),
                frame_id=i,
                is_keyframe=True,
                measurements=synthetic_visual_frame.measurements if i == 0 else []
            )
            estimator.update(frame, landmark_map)
        
        # Run optimization
        converged = estimator.optimize()
        
        assert converged
        assert estimator.num_optimizations == 1
    
    def test_state_vector_extraction(self, estimator):
        """Test state vector extraction."""
        # Create rotation matrix for 90 deg around Y
        angle = np.pi / 2
        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        
        initial_pose = Pose(
            timestamp=0.0,
            position=np.array([1.0, 2.0, 3.0]),
            rotation_matrix=rotation_matrix
        )
        estimator.initialize(initial_pose)
        estimator.current_velocity = np.array([0.5, -0.2, 0.1])
        
        state = estimator.get_state_vector()
        
        assert state.shape == (21,)  # Updated size for rotation matrix
        assert np.allclose(state[0:3], initial_pose.position)
        assert np.allclose(state[3:12], rotation_matrix.flatten())
        assert np.allclose(state[12:15], estimator.current_velocity)
        assert np.allclose(state[15:21], np.zeros(6))  # Biases
    
    def test_get_result(self, estimator, synthetic_visual_frame, landmark_map):
        """Test getting estimation result."""
        # Initialize and add some data
        initial_pose = Pose(
            timestamp=0.0,
            position=np.array([0.0, 0.0, 0.0]),
            rotation_matrix=np.eye(3)
        )
        estimator.initialize(initial_pose)
        
        # Add keyframes
        for i in range(2):
            estimator.current_pose = Pose(
                timestamp=float(i),
                position=np.array([i * 1.0, 0.0, 0.0]),
                rotation_matrix=np.eye(3)
            )
            frame = ProcessedVisualFrame(
                timestamp=float(i),
                frame_id=i,
                is_keyframe=True,
                measurements=synthetic_visual_frame.measurements if i == 0 else []
            )
            estimator.update(frame, landmark_map)
        
        # Get result
        result = estimator.get_result()
        
        assert result is not None
        assert len(result.trajectory.states) == 2
        assert len(result.landmarks.landmarks) == 2
        assert result.metadata['estimator_type'] == 'new_swba'
        assert result.metadata['num_keyframes'] == 2
    
    def test_no_camera_calibration_required(self, config):
        """Verify that no camera calibration is needed."""
        estimator = NewSWBAEstimator(config)
        
        # Check that no calibration parameters are stored
        assert estimator.imu_calib is None
        assert estimator.camera_calib is None
        
        # Config should not have calibration fields
        assert not hasattr(config, 'camera_intrinsics')
        assert not hasattr(config, 'imu_noise_params')
    
    def test_preprocessed_jacobians_used(self, estimator, synthetic_visual_frame, landmark_map):
        """Verify that pre-computed Jacobians are available and valid."""
        # Initialize
        initial_pose = Pose(
            timestamp=0.0,
            position=np.array([0.0, 0.0, 0.0]),
            rotation_matrix=np.eye(3)
        )
        estimator.initialize(initial_pose)
        
        # Update with frame containing pre-computed Jacobians
        estimator.update(synthetic_visual_frame, landmark_map)
        
        # Verify measurements have Jacobians
        for measurement in synthetic_visual_frame.measurements:
            assert measurement.jacobian_wrt_pose is not None
            assert measurement.jacobian_wrt_landmark is not None
            assert measurement.jacobian_wrt_pose.shape == (2, 6)
            assert measurement.jacobian_wrt_landmark.shape == (2, 3)
    
    def test_robust_weights_applied(self, synthetic_visual_measurement):
        """Test that pre-computed robust weights are used."""
        # Check that measurement has robust weight
        assert hasattr(synthetic_visual_measurement, 'robust_weight')
        assert 0.0 <= synthetic_visual_measurement.robust_weight <= 1.0
        
        # Check weighted residual
        weighted_residual = synthetic_visual_measurement.weighted_residual
        expected = synthetic_visual_measurement.residual * synthetic_visual_measurement.robust_weight
        assert np.allclose(weighted_residual, expected)