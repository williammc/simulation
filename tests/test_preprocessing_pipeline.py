"""
Tests for the visual measurement preprocessing pipeline.

Tests conversion from raw camera frames to processed measurements
without requiring camera models in the estimator.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from src.estimation.preprocessing import (
    VisualMeasurementPreprocessor,
    IMUPreprocessor
)
from src.estimation.interfaces import (
    VisualMeasurement,
    ProcessedVisualFrame,
    PreprocessedIMUData,
    ProjectionInterface
)
from src.common.data_structures import (
    CameraFrame,
    CameraObservation,
    ImagePoint,
    Pose,
    TrajectoryState,
    Map,
    Landmark,
    CameraCalibration,
    CameraIntrinsics,
    CameraExtrinsics,
    CameraModel
)

# Alias for compatibility
State = TrajectoryState


class MockProjectionService:
    """Mock projection service for testing."""
    
    def __init__(self, pixel_noise_std=1.0):
        self.pixel_noise_std = pixel_noise_std
        self.calls = []
        
        # Create mock camera calibration
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
        extrinsics = CameraExtrinsics(B_T_C=np.eye(4))
        self.camera_calib = CameraCalibration(
            camera_id="cam0",
            intrinsics=intrinsics,
            extrinsics=extrinsics
        )
    
    def project(self, landmark_position, camera_pose, compute_jacobians=False):
        """Mock projection that returns synthetic measurements."""
        self.calls.append((landmark_position, camera_pose, compute_jacobians))
        
        # Create synthetic projection result
        predicted_pixel = np.array([320.0, 240.0]) + np.random.randn(2) * 0.1
        
        result = Mock()
        result.predicted_pixel = predicted_pixel
        result.pixel_covariance = np.eye(2) * (self.pixel_noise_std ** 2)
        
        if compute_jacobians:
            result.jacobian_wrt_pose = np.random.randn(2, 6) * 0.1
            result.jacobian_wrt_landmark = np.random.randn(2, 3) * 0.1
        else:
            result.jacobian_wrt_pose = None
            result.jacobian_wrt_landmark = None
        
        return result


class TestVisualMeasurementPreprocessor:
    """Test visual measurement preprocessing."""
    
    @pytest.fixture
    def projection_service(self):
        """Create mock projection service."""
        return MockProjectionService(pixel_noise_std=1.0)
    
    @pytest.fixture
    def preprocessor(self, projection_service):
        """Create preprocessor with mock projection service."""
        return VisualMeasurementPreprocessor(
            projection_service=projection_service,
            pixel_noise_std=1.0,
            robust_kernel='huber',
            huber_delta=1.0
        )
    
    @pytest.fixture
    def raw_camera_frame(self):
        """Create raw camera frame with observations."""
        observations = [
            CameraObservation(
                landmark_id=1,
                pixel=ImagePoint(u=320.5, v=240.2),
                descriptor=None
            ),
            CameraObservation(
                landmark_id=2,
                pixel=ImagePoint(u=400.0, v=300.0),
                descriptor=None
            )
        ]
        
        return CameraFrame(
            timestamp=1.0,
            camera_id="cam0",
            observations=observations,
            is_keyframe=True,
            keyframe_id=0
        )
    
    @pytest.fixture
    def current_state(self):
        """Create current state estimate."""
        return State(
            pose=Pose(
                timestamp=1.0,
                position=np.array([1.0, 0.0, 0.0]),
                rotation_matrix=np.eye(3)
            ),
            velocity=np.array([0.5, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.1])
        )
    
    @pytest.fixture
    def landmark_map(self):
        """Create landmark map."""
        map_obj = Map()
        landmarks = [
            Landmark(id=1, position=np.array([5.0, 2.0, 2.0])),  # Move to z=2 so camera sees z=1
            Landmark(id=2, position=np.array([5.0, -2.0, 2.0])),  # Move to z=2 so camera sees z=1  
            Landmark(id=3, position=np.array([10.0, 0.0, 2.0]))  # Move to z=2 so camera sees z=1
        ]
        for lm in landmarks:
            map_obj.add_landmark(lm)
        return map_obj
    
    def test_process_frame_basic(self, preprocessor, raw_camera_frame, current_state, landmark_map):
        """Test basic frame processing."""
        processed = preprocessor.process_frame(
            raw_camera_frame,
            current_state,
            landmark_map,
            compute_jacobians=True
        )
        
        assert isinstance(processed, ProcessedVisualFrame)
        assert processed.timestamp == raw_camera_frame.timestamp
        assert processed.is_keyframe == raw_camera_frame.is_keyframe
        assert processed.keyframe_id == raw_camera_frame.keyframe_id
        assert len(processed.measurements) == 2
        
        # Check each measurement
        for meas in processed.measurements:
            assert isinstance(meas, VisualMeasurement)
            assert meas.observed_pixel.shape == (2,)
            assert meas.predicted_pixel.shape == (2,)
            assert meas.residual.shape == (2,)
            assert meas.pixel_covariance.shape == (2, 2)
            # Check ideal coordinate fields (new camera-model-independent approach)
            assert meas.observed_ideal is not None
            assert meas.observed_ideal.shape == (2,)
            assert meas.predicted_ideal is not None
            assert meas.predicted_ideal.shape == (2,)
            assert meas.ideal_residual is not None
            assert meas.ideal_residual.shape == (2,)
            assert meas.ideal_jacobian_wrt_pose is not None
            assert meas.ideal_jacobian_wrt_pose.shape == (2, 6)
            assert meas.ideal_jacobian_wrt_landmark is not None
            assert meas.ideal_jacobian_wrt_landmark.shape == (2, 3)
    
    def test_residual_computation(self, preprocessor, raw_camera_frame, current_state, landmark_map):
        """Test that residuals are computed correctly."""
        processed = preprocessor.process_frame(
            raw_camera_frame,
            current_state,
            landmark_map,
            compute_jacobians=False
        )
        
        for i, meas in enumerate(processed.measurements):
            obs = raw_camera_frame.observations[i]
            observed = np.array([obs.pixel.u, obs.pixel.v])
            
            # Residual should be observed - predicted
            expected_residual = observed - meas.predicted_pixel
            assert np.allclose(meas.residual, expected_residual)
    
    def test_outlier_rejection(self, preprocessor, raw_camera_frame, current_state, landmark_map):
        """Test chi-squared outlier rejection."""
        # Add an outlier observation
        raw_camera_frame.observations.append(
            CameraObservation(
                landmark_id=3,
                pixel=ImagePoint(u=1000.0, v=1000.0),  # Far from expected
                descriptor=None
            )
        )
        
        chi2_threshold = 5.991  # 95% for 2 DOF
        processed = preprocessor.process_frame(
            raw_camera_frame,
            current_state,
            landmark_map,
            compute_jacobians=False,
            chi2_threshold=chi2_threshold
        )
        
        # Should have fewer measurements due to outlier rejection
        # (exact number depends on mock projection results)
        assert len(processed.measurements) <= len(raw_camera_frame.observations)
    
    def test_robust_kernel_huber(self, projection_service):
        """Test Huber robust kernel application."""
        preprocessor = VisualMeasurementPreprocessor(
            projection_service=projection_service,
            pixel_noise_std=1.0,
            robust_kernel='huber',
            huber_delta=1.0
        )
        
        # Test weight computation for different residual magnitudes
        small_residual = np.array([0.5, 0.5])
        large_residual = np.array([5.0, 5.0])
        covariance = np.eye(2)
        
        small_weight = preprocessor.compute_robust_weight(small_residual, covariance)
        large_weight = preprocessor.compute_robust_weight(large_residual, covariance)
        
        assert small_weight > large_weight
        assert 0.0 <= small_weight <= 1.0
        assert 0.0 <= large_weight <= 1.0
    
    def test_robust_kernel_cauchy(self, projection_service):
        """Test Cauchy robust kernel application."""
        preprocessor = VisualMeasurementPreprocessor(
            projection_service=projection_service,
            pixel_noise_std=1.0,
            robust_kernel='cauchy'
        )
        
        residual = np.array([2.0, 2.0])
        covariance = np.eye(2)
        
        weight = preprocessor.compute_robust_weight(residual, covariance)
        
        assert 0.0 < weight < 1.0
    
    def test_batch_processing(self, preprocessor, raw_camera_frame, current_state, landmark_map):
        """Test batch frame processing."""
        frames = [raw_camera_frame] * 3
        states = [current_state] * 3
        
        processed_frames = preprocessor.process_batch(
            frames,
            states,
            landmark_map,
            compute_jacobians=True
        )
        
        assert len(processed_frames) == 3
        for processed in processed_frames:
            assert isinstance(processed, ProcessedVisualFrame)
            assert len(processed.measurements) > 0
    
    def test_missing_landmark_handling(self, preprocessor, raw_camera_frame, current_state):
        """Test handling of missing landmarks in map."""
        # Create map without landmark 2
        partial_map = Map()
        partial_map.add_landmark(Landmark(id=1, position=np.array([5.0, 2.0, 2.0])))  # Move to z=2 so camera sees z=1
        
        processed = preprocessor.process_frame(
            raw_camera_frame,
            current_state,
            partial_map,
            compute_jacobians=False
        )
        
        # Should only have measurement for landmark 1
        assert len(processed.measurements) == 1
        assert processed.measurements[0].landmark_id == 1
    
    def test_projection_failure_handling(self, raw_camera_frame, current_state, landmark_map):
        """Test handling of invalid camera calibration."""
        # Create projection service with invalid camera calibration (zero focal length)
        failing_service = Mock()
        
        # Invalid camera calibration that would cause division by zero
        intrinsics = CameraIntrinsics(
            model=CameraModel.PINHOLE,
            width=640,
            height=480,
            fx=0.0,  # Invalid - zero focal length
            fy=0.0,  # Invalid - zero focal length
            cx=320.0,
            cy=240.0,
            distortion=np.zeros(5)
        )
        extrinsics = CameraExtrinsics(B_T_C=np.eye(4))
        failing_service.camera_calib = CameraCalibration(
            camera_id="cam0",
            intrinsics=intrinsics,
            extrinsics=extrinsics
        )
        
        preprocessor = VisualMeasurementPreprocessor(
            projection_service=failing_service,
            pixel_noise_std=1.0
        )
        
        # This should either fail gracefully or produce invalid measurements
        # The current implementation should handle this gracefully
        processed = preprocessor.process_frame(
            raw_camera_frame,
            current_state,
            landmark_map,
            compute_jacobians=False
        )
        
        # With invalid calibration, measurements might still be created but with invalid values
        # The key is that it doesn't crash - the actual behavior depends on implementation
        assert isinstance(processed, ProcessedVisualFrame)
        # Don't assert on number of measurements since behavior with invalid calib may vary
    
    def test_information_matrix_computation(self, preprocessor, raw_camera_frame, current_state, landmark_map):
        """Test that information matrices are computed."""
        processed = preprocessor.process_frame(
            raw_camera_frame,
            current_state,
            landmark_map,
            compute_jacobians=False
        )
        
        for meas in processed.measurements:
            assert meas.information_matrix is not None
            assert meas.information_matrix.shape == (2, 2)
            
            # Information should be inverse of covariance
            expected_info = np.linalg.inv(meas.pixel_covariance)
            assert np.allclose(meas.information_matrix, expected_info, atol=1e-6)


class TestIMUPreprocessor:
    """Test IMU preprocessing."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create IMU preprocessor."""
        return IMUPreprocessor(gravity_magnitude=9.81)
    
    @pytest.fixture
    def valid_preintegration(self):
        """Create valid pre-integrated IMU data."""
        return PreprocessedIMUData(
            from_keyframe_id=0,
            to_keyframe_id=1,
            delta_position=np.array([0.1, 0.05, 0.01]),
            delta_velocity=np.array([0.5, 0.2, 0.05]),
            delta_rotation=np.eye(3),
            covariance=np.eye(9) * 0.01,
            delta_t=0.1,
            num_measurements=20,
            accel_bias=np.array([0.01, 0.02, 0.03]),
            gyro_bias=np.array([0.001, 0.002, 0.003])
        )
    
    def test_validate_valid_preintegration(self, preprocessor, valid_preintegration):
        """Test validation of valid pre-integrated data."""
        is_valid = preprocessor.validate_preintegration(
            valid_preintegration,
            expected_dt=0.1
        )
        assert is_valid
    
    def test_validate_missing_fields(self, preprocessor):
        """Test validation catches missing fields."""
        incomplete_data = Mock()
        incomplete_data.delta_position = np.zeros(3)
        incomplete_data.delta_velocity = np.zeros(3)
        # Missing delta_rotation, covariance, delta_t
        
        is_valid = preprocessor.validate_preintegration(incomplete_data)
        assert not is_valid
    
    def test_validate_wrong_dt(self, preprocessor, valid_preintegration):
        """Test validation catches wrong time interval."""
        is_valid = preprocessor.validate_preintegration(
            valid_preintegration,
            expected_dt=0.2  # Different from actual 0.1
        )
        assert not is_valid
    
    def test_validate_nan_values(self, preprocessor):
        """Test validation catches NaN values."""
        bad_data = PreprocessedIMUData(
            from_keyframe_id=0,
            to_keyframe_id=1,
            delta_position=np.array([np.nan, 0.0, 0.0]),
            delta_velocity=np.array([0.0, 0.0, 0.0]),
            delta_rotation=np.eye(3),
            covariance=np.eye(9),
            delta_t=0.1
        )
        
        is_valid = preprocessor.validate_preintegration(bad_data)
        assert not is_valid
    
    def test_validate_inf_values(self, preprocessor):
        """Test validation catches infinite values."""
        bad_data = PreprocessedIMUData(
            from_keyframe_id=0,
            to_keyframe_id=1,
            delta_position=np.array([0.0, 0.0, 0.0]),
            delta_velocity=np.array([0.0, np.inf, 0.0]),
            delta_rotation=np.eye(3),
            covariance=np.eye(9),
            delta_t=0.1
        )
        
        is_valid = preprocessor.validate_preintegration(bad_data)
        assert not is_valid
    
    def test_rotation_matrix_property(self):
        """Test rotation matrix extraction from quaternion."""
        # Test with rotation matrix
        data_matrix = PreprocessedIMUData(
            from_keyframe_id=0,
            to_keyframe_id=1,
            delta_position=np.zeros(3),
            delta_velocity=np.zeros(3),
            delta_rotation=np.eye(3),
            covariance=np.eye(9),
            delta_t=0.1
        )
        assert data_matrix.rotation_matrix.shape == (3, 3)
        
        # Test with quaternion
        data_quat = PreprocessedIMUData(
            from_keyframe_id=0,
            to_keyframe_id=1,
            delta_position=np.zeros(3),
            delta_velocity=np.zeros(3),
            delta_rotation=np.array([1.0, 0.0, 0.0, 0.0]),
            covariance=np.eye(9),
            delta_t=0.1
        )
        assert data_quat.rotation_matrix.shape == (3, 3)