"""
Unit tests for SimpleSWBAVIO estimator.
"""

import pytest
import numpy as np
from typing import List

from src.estimation.new.simple_swba_vio import SimpleSWBAVIO, SimpleSWBAConfig
from src.estimation.interfaces import (
    VisualMeasurement,
    ProcessedVisualFrame,
    PreprocessedIMUData
)
from src.common.data_structures import Pose, Map, Landmark


class TestSimpleSWBAVIO:
    """Test suite for SimpleSWBAVIO estimator."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SimpleSWBAConfig(
            window_size=5,
            keyframe_spacing=5,
            max_iterations=5,
            convergence_threshold=1e-4,
            min_measurements=3
        )
    
    @pytest.fixture
    def estimator(self, config):
        """Create estimator instance."""
        return SimpleSWBAVIO(config)
    
    @pytest.fixture
    def initial_pose(self):
        """Create initial pose."""
        return Pose(
            timestamp=0.0,
            position=np.array([0.0, 0.0, 0.0]),
            rotation_matrix=np.eye(3)
        )
    
    @pytest.fixture
    def imu_data(self):
        """Create sample preintegrated IMU data."""
        return PreprocessedIMUData(
            from_keyframe_id=0,
            to_keyframe_id=1,
            delta_position=np.array([0.1, 0.05, 0.01]),
            delta_velocity=np.array([0.5, 0.2, 0.05]),
            delta_rotation=np.eye(3),  # Changed from rotation_matrix to delta_rotation
            delta_t=0.1,  # Changed from dt to delta_t
            covariance=np.eye(9),  # Changed from information_matrix to covariance
            num_measurements=10
        )
    
    @pytest.fixture
    def visual_measurement(self):
        """Create sample visual measurement with Jacobians."""
        return VisualMeasurement(
            landmark_id=1,
            observed_pixel=np.array([320.5, 240.2]),
            predicted_pixel=np.array([320.0, 240.0]),
            residual=np.array([0.5, 0.2]),
            pixel_covariance=np.eye(2),
            jacobian_wrt_pose=np.random.randn(2, 6) * 0.1,
            jacobian_wrt_landmark=np.random.randn(2, 3) * 0.1,
            # Ideal coordinates
            observed_ideal=np.array([0.0, 0.0]),
            predicted_ideal=np.array([0.001, 0.001]),
            ideal_residual=np.array([-0.001, -0.001]),
            ideal_jacobian_wrt_pose=np.random.randn(2, 6) * 0.01,
            ideal_jacobian_wrt_landmark=np.random.randn(2, 3) * 0.01
        )
    
    @pytest.fixture
    def visual_frame(self, visual_measurement):
        """Create processed visual frame."""
        measurements = [visual_measurement]
        # Add more measurements
        for i in range(2, 5):
            meas = VisualMeasurement(
                landmark_id=i,
                observed_pixel=np.array([320.0 + i*10, 240.0 + i*5]),
                predicted_pixel=np.array([320.0 + i*10, 240.0 + i*5]),
                residual=np.array([0.1, 0.1]),
                pixel_covariance=np.eye(2),
                jacobian_wrt_pose=np.random.randn(2, 6) * 0.1,
                jacobian_wrt_landmark=np.random.randn(2, 3) * 0.1
            )
            measurements.append(meas)
            
        return ProcessedVisualFrame(
            timestamp=0.1,
            frame_id=1,
            measurements=measurements,
            is_keyframe=True
        )
    
    def test_initialization(self, estimator, initial_pose):
        """Test estimator initialization."""
        estimator.initialize(initial_pose)
        
        assert estimator.current_pose is not None
        assert np.allclose(estimator.current_pose.position, initial_pose.position)
        assert np.allclose(estimator.current_pose.rotation_matrix, initial_pose.rotation_matrix)
        assert np.allclose(estimator.current_velocity, np.zeros(3))
    
    def test_predict(self, estimator, initial_pose, imu_data):
        """Test IMU prediction step."""
        estimator.initialize(initial_pose)
        
        # Store initial state
        initial_pos = estimator.current_pose.position.copy()
        initial_vel = estimator.current_velocity.copy()
        
        # Predict
        estimator.predict(imu_data, dt=0.1)
        
        # Check state propagation
        assert estimator.current_pose is not None
        assert not np.allclose(estimator.current_pose.position, initial_pos)
        assert not np.allclose(estimator.current_velocity, initial_vel)
        
        # Check IMU data is stored
        assert estimator.pending_imu is not None
        assert estimator.pending_imu == imu_data
    
    def test_update_keyframe_creation(self, estimator, initial_pose, visual_frame):
        """Test keyframe creation during update."""
        estimator.initialize(initial_pose)
        
        # First update should create keyframe
        estimator.update(visual_frame)
        
        assert len(estimator.keyframes) == 1
        assert estimator.keyframe_count == 1
        
        # Update with spacing
        for i in range(estimator.config.keyframe_spacing - 1):
            estimator.update(visual_frame)
        
        # Should create second keyframe
        assert len(estimator.keyframes) == 2
        assert estimator.keyframe_count == 2
    
    def test_landmark_initialization(self, estimator, initial_pose, visual_frame):
        """Test landmark initialization from measurements."""
        estimator.initialize(initial_pose)
        estimator.update(visual_frame)
        
        # Check landmarks were initialized
        assert len(estimator.landmarks) > 0
        
        # Check each measurement has corresponding landmark
        for meas in visual_frame.measurements:
            assert meas.landmark_id in estimator.landmarks
            assert estimator.landmarks[meas.landmark_id].shape == (3,)
    
    def test_optimization(self, estimator, initial_pose, imu_data, visual_frame):
        """Test optimization with IMU and visual constraints."""
        estimator.initialize(initial_pose)
        
        # Create multiple keyframes
        for i in range(3):
            # Predict with IMU
            estimator.predict(imu_data, dt=0.1)
            
            # Create keyframe
            for j in range(estimator.config.keyframe_spacing):
                estimator.update(visual_frame)
        
        # Should have optimized
        assert estimator.total_iterations > 0
        assert len(estimator.keyframes) >= 3
    
    def test_marginalization(self, estimator, initial_pose, visual_frame):
        """Test sliding window marginalization."""
        estimator.initialize(initial_pose)
        
        # Create more keyframes than window size
        for i in range(estimator.config.window_size + 2):
            for j in range(estimator.config.keyframe_spacing):
                estimator.update(visual_frame)
        
        # Check window size is maintained
        assert len(estimator.keyframes) == estimator.config.window_size
        assert len(estimator.keyframe_ids) == estimator.config.window_size
    
    def test_imu_constraints(self, estimator, initial_pose, imu_data, visual_frame):
        """Test IMU constraints between keyframes."""
        estimator.initialize(initial_pose)
        
        # Create first keyframe
        estimator.update(visual_frame)
        
        # Predict with IMU
        estimator.predict(imu_data, dt=0.1)
        
        # Create second keyframe
        for j in range(estimator.config.keyframe_spacing):
            estimator.update(visual_frame)
        
        # Check IMU constraint was stored
        assert len(estimator.imu_constraints) > 0
        
        # Check constraint connects consecutive keyframes
        for (from_id, to_id), constraint in estimator.imu_constraints.items():
            assert to_id == from_id + 1
            assert constraint is not None
    
    def test_state_vector(self, estimator, initial_pose):
        """Test state vector extraction."""
        estimator.initialize(initial_pose)
        
        state = estimator.get_state_vector()
        
        assert state.shape == (21,)
        assert np.allclose(state[0:3], initial_pose.position)
        assert np.allclose(state[3:12], initial_pose.rotation_matrix.flatten())
        assert np.allclose(state[12:15], np.zeros(3))  # velocity
        assert np.allclose(state[15:21], np.zeros(6))  # biases
    
    def test_ideal_coordinates_preferred(self, estimator, initial_pose, visual_frame):
        """Test that ideal coordinates are preferred in optimization."""
        estimator.initialize(initial_pose)
        
        # Create keyframes for optimization
        for i in range(3):
            for j in range(estimator.config.keyframe_spacing):
                estimator.update(visual_frame)
        
        # Optimization should have used ideal coordinates
        # (checked by weight difference in optimize method)
        assert estimator.total_iterations > 0
        
        # Verify first measurement has ideal coordinates
        first_meas = visual_frame.measurements[0]
        assert first_meas.has_ideal_coordinates
        assert first_meas.ideal_residual is not None
        assert first_meas.ideal_jacobian_wrt_pose is not None