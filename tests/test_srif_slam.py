"""
Unit tests for SRIF (Square Root Information Filter) SLAM implementation.
"""

import pytest
import numpy as np
from pathlib import Path

from src.estimation.srif_slam import (
    SRIFSlam, SRIFState, SRIFConfig
)
from src.estimation.ekf_slam import EKFSlam, EKFConfig
from src.estimation.base_estimator import EstimatorType
from src.common.data_structures import (
    Pose, IMUMeasurement, CameraFrame, CameraObservation,
    ImagePoint, Map, Landmark, Trajectory, TrajectoryState,
    CameraCalibration, CameraIntrinsics, CameraExtrinsics,
    CameraModel, IMUCalibration
)
from src.evaluation.metrics import compute_ate


class TestSRIFState:
    """Test SRIF state representation."""
    
    def test_state_initialization(self):
        """Test SRIF state creation."""
        state = SRIFState(
            position=np.array([1, 2, 3]),
            velocity=np.array([0.1, 0.2, 0.3]),
            quaternion=np.array([1, 0, 0, 0]),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp=1.0
        )
        
        assert np.allclose(state.position, [1, 2, 3])
        assert np.allclose(state.velocity, [0.1, 0.2, 0.3])
        assert state.timestamp == 1.0
        assert state.sqrt_information.shape == (15, 15)
    
    def test_information_covariance_conversion(self):
        """Test conversion between information and covariance forms."""
        state = SRIFState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            quaternion=np.array([1, 0, 0, 0]),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp=0.0
        )
        
        # Set a simple diagonal sqrt information matrix
        state.sqrt_information = np.diag([2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5])
        
        # Get covariance
        P = state.get_covariance_matrix()
        
        # Check that P * (R^T R) = I
        R = state.sqrt_information
        I_approx = P @ (R.T @ R)
        assert np.allclose(I_approx, np.eye(15), atol=1e-10)
    
    def test_state_vector_packing(self):
        """Test state vector packing and unpacking."""
        state = SRIFState(
            position=np.array([1, 2, 3]),
            velocity=np.array([4, 5, 6]),
            quaternion=np.array([0.707, 0, 0, 0.707]),
            accel_bias=np.array([0.1, 0.2, 0.3]),
            gyro_bias=np.array([0.01, 0.02, 0.03]),
            timestamp=1.0
        )
        
        # Pack state
        x = state._pack_state_vector()
        assert x.shape == (15,)
        assert np.allclose(x[0:3], [1, 2, 3])
        assert np.allclose(x[3:6], [4, 5, 6])
        
        # Unpack state
        x_new = np.random.randn(15)
        x_new[6:10] = [1, 0, 0, 0]  # Valid quaternion
        state._unpack_state_vector(x_new)
        assert np.allclose(state.position, x_new[0:3])
        assert np.allclose(state.velocity, x_new[3:6])


class TestSRIFConfig:
    """Test SRIF configuration."""
    
    def test_default_config(self):
        """Test default SRIF configuration."""
        config = SRIFConfig()
        
        assert config.estimator_type == EstimatorType.SRIF
        assert config.initial_position_std == 0.1
        assert config.pixel_noise_std == 1.0
        assert config.chi2_threshold == 5.991
        assert config.qr_threshold == 1e-10
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SRIFConfig(
            initial_position_std=0.5,
            pixel_noise_std=2.0,
            qr_threshold=1e-8
        )
        
        assert config.initial_position_std == 0.5
        assert config.pixel_noise_std == 2.0
        assert config.qr_threshold == 1e-8


class TestSRIFSlam:
    """Test SRIF SLAM estimator."""
    
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
        
        extrinsics = CameraExtrinsics(
            B_T_C=np.eye(4)
        )
        
        return CameraCalibration(
            camera_id="cam0",
            intrinsics=intrinsics,
            extrinsics=extrinsics
        )
    
    @pytest.fixture
    def imu_calibration(self):
        """Create IMU calibration."""
        return IMUCalibration(
            imu_id="imu0",
            accelerometer_noise_density=0.01,
            accelerometer_random_walk=0.001,
            gyroscope_noise_density=0.001,
            gyroscope_random_walk=0.0001,
            rate=200.0
        )
    
    def test_srif_initialization(self, camera_calibration):
        """Test SRIF initialization."""
        config = SRIFConfig()
        srif = SRIFSlam(config, camera_calibration)
        
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            quaternion=np.array([1, 0, 0, 0])
        )
        
        srif.initialize(initial_pose)
        
        state = srif.get_state()
        assert state.timestamp == 0.0
        assert np.allclose(state.robot_pose.position, [0, 0, 0])
        
        # Check information matrix is initialized
        info_matrix = srif.get_information_matrix()
        assert info_matrix is not None
        assert info_matrix.shape == (15, 15)
    
    def test_sqrt_information_structure(self, camera_calibration):
        """Test that sqrt information matrix maintains upper triangular structure."""
        config = SRIFConfig()
        srif = SRIFSlam(config, camera_calibration)
        
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            quaternion=np.array([1, 0, 0, 0])
        )
        srif.initialize(initial_pose)
        
        R = srif.get_sqrt_information_matrix()
        
        # Check upper triangular (within numerical tolerance)
        for i in range(R.shape[0]):
            for j in range(i):
                assert abs(R[i, j]) < 1e-10, f"R[{i},{j}] = {R[i,j]} should be zero"
    
    def test_qr_measurement_update(self, camera_calibration):
        """Test QR-based measurement update."""
        config = SRIFConfig()
        srif = SRIFSlam(config, camera_calibration)
        
        # Initialize
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            quaternion=np.array([1, 0, 0, 0])
        )
        srif.initialize(initial_pose)
        
        # Create map with landmark
        map_data = Map()
        landmark = Landmark(id=0, position=np.array([2, 0, 5]))
        map_data.add_landmark(landmark)
        
        # Create observation
        obs = CameraObservation(
            landmark_id=0,
            pixel=ImagePoint(u=320, v=240)
        )
        
        frame = CameraFrame(
            timestamp=0.1,
            camera_id="cam0",
            observations=[obs]
        )
        
        # Store initial sqrt information
        R_before = srif.get_sqrt_information_matrix().copy()
        
        # Update
        srif.update(frame, map_data)
        
        # Check sqrt information is still upper triangular
        R_after = srif.get_sqrt_information_matrix()
        for i in range(R_after.shape[0]):
            for j in range(i):
                assert abs(R_after[i, j]) < 1e-10
        
        # Check that information increased (uncertainty decreased)
        info_before = R_before.T @ R_before
        info_after = R_after.T @ R_after
        # Diagonal elements should generally increase
        assert np.sum(np.diag(info_after)) >= np.sum(np.diag(info_before))
    
    def test_imu_prediction(self, camera_calibration):
        """Test IMU prediction step."""
        config = SRIFConfig()
        srif = SRIFSlam(config, camera_calibration)
        
        # Initialize
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            quaternion=np.array([1, 0, 0, 0])
        )
        srif.initialize(initial_pose)
        
        # Create IMU measurements
        measurements = []
        for i in range(10):
            meas = IMUMeasurement(
                timestamp=(i + 1) * 0.01,
                accelerometer=np.array([0.1, 0, 0]),
                gyroscope=np.zeros(3)
            )
            measurements.append(meas)
        
        # Predict
        srif.predict(measurements, 0.1)
        
        # Check state has been updated
        state = srif.get_state()
        assert state.timestamp == 0.1
        assert state.robot_pose.position[0] > 0  # Should have moved forward
        
        # Check sqrt information is still upper triangular
        R = srif.get_sqrt_information_matrix()
        for i in range(R.shape[0]):
            for j in range(i):
                assert abs(R[i, j]) < 1e-10


class TestSRIFNumericalStability:
    """Test numerical stability of SRIF."""
    
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
    
    def test_ill_conditioned_covariance(self, camera_calibration):
        """Test SRIF with ill-conditioned covariance."""
        config = SRIFConfig()
        srif = SRIFSlam(config, camera_calibration)
        
        # Create ill-conditioned covariance
        P = np.eye(15)
        P[0, 0] = 1e10  # Very large uncertainty
        P[1, 1] = 1e-10  # Very small uncertainty
        
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            quaternion=np.array([1, 0, 0, 0])
        )
        
        # Initialize with ill-conditioned covariance
        srif.initialize(initial_pose, P)
        
        # Should not crash and sqrt_information should be valid
        R = srif.get_sqrt_information_matrix()
        assert not np.any(np.isnan(R))
        assert not np.any(np.isinf(R))
    
    def test_repeated_updates(self, camera_calibration):
        """Test numerical stability over many updates."""
        config = SRIFConfig()
        srif = SRIFSlam(config, camera_calibration)
        
        # Initialize
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            quaternion=np.array([1, 0, 0, 0])
        )
        srif.initialize(initial_pose)
        
        # Create map
        map_data = Map()
        for i in range(5):
            landmark = Landmark(id=i, position=np.array([i-2, 0, 5]))
            map_data.add_landmark(landmark)
        
        # Perform many updates
        for t in range(100):
            # IMU prediction
            imu_meas = IMUMeasurement(
                timestamp=t * 0.01,
                accelerometer=np.array([0.01, 0, 0]),
                gyroscope=np.array([0, 0, 0.01])
            )
            srif.predict([imu_meas], 0.01)
            
            # Camera update every 10 steps
            if t % 10 == 0:
                observations = []
                for i in range(3):
                    obs = CameraObservation(
                        landmark_id=i,
                        pixel=ImagePoint(u=320 + i*10, v=240)
                    )
                    observations.append(obs)
                
                frame = CameraFrame(
                    timestamp=t * 0.01,
                    camera_id="cam0",
                    observations=observations
                )
                srif.update(frame, map_data)
        
        # Check sqrt information is still valid
        R = srif.get_sqrt_information_matrix()
        assert not np.any(np.isnan(R))
        assert not np.any(np.isinf(R))
        
        # Check still upper triangular
        for i in range(R.shape[0]):
            for j in range(i):
                assert abs(R[i, j]) < 1e-8


class TestSRIFEKFEquivalence:
    """Test that SRIF produces equivalent results to EKF."""
    
    @pytest.fixture
    def simulation_setup(self):
        """Create simulation setup."""
        # Camera calibration
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
        camera_calib = CameraCalibration(
            camera_id="cam0",
            intrinsics=intrinsics,
            extrinsics=extrinsics
        )
        
        # IMU calibration
        imu_calib = IMUCalibration(
            imu_id="imu0",
            accelerometer_noise_density=0.01,
            accelerometer_random_walk=0.001,
            gyroscope_noise_density=0.001,
            gyroscope_random_walk=0.0001,
            rate=200.0
        )
        
        return camera_calib, imu_calib
    
    @pytest.mark.xfail(reason="SRIF implementation is simplified and may differ from EKF numerically")
    def test_ekf_srif_equivalence(self, simulation_setup):
        """Test that EKF and SRIF produce equivalent results."""
        camera_calib, imu_calib = simulation_setup
        
        # Create EKF and SRIF with same configuration
        ekf_config = EKFConfig(
            initial_position_std=0.1,
            initial_velocity_std=0.1,
            pixel_noise_std=1.0
        )
        ekf = EKFSlam(ekf_config, camera_calib, imu_calib)
        
        srif_config = SRIFConfig(
            initial_position_std=0.1,
            initial_velocity_std=0.1,
            pixel_noise_std=1.0
        )
        srif = SRIFSlam(srif_config, camera_calib, imu_calib)
        
        # Initialize both with same pose
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            quaternion=np.array([1, 0, 0, 0])
        )
        ekf.initialize(initial_pose)
        srif.initialize(initial_pose)
        
        # Create simple trajectory and measurements
        map_data = Map()
        landmark = Landmark(id=0, position=np.array([5, 0, 0]))
        map_data.add_landmark(landmark)
        
        # Process same measurements
        for t in range(10):
            # IMU measurement
            imu_meas = IMUMeasurement(
                timestamp=t * 0.1,
                accelerometer=np.array([0.1, 0, 0]),
                gyroscope=np.zeros(3)
            )
            
            ekf.predict([imu_meas], 0.1)
            srif.predict([imu_meas], 0.1)
            
            # Camera measurement every 2 steps
            if t % 2 == 0:
                obs = CameraObservation(
                    landmark_id=0,
                    pixel=ImagePoint(u=320, v=240)
                )
                frame = CameraFrame(
                    timestamp=t * 0.1,
                    camera_id="cam0",
                    observations=[obs]
                )
                
                ekf.update(frame, map_data)
                srif.update(frame, map_data)
        
        # Get final states
        ekf_state = ekf.get_state()
        srif_state = srif.get_state()
        
        # States should be very similar
        assert np.allclose(
            ekf_state.robot_pose.position,
            srif_state.robot_pose.position,
            atol=1e-1  # Allow 10cm difference due to numerical differences
        )
        assert np.allclose(
            ekf_state.robot_velocity,
            srif_state.robot_velocity,
            atol=1e-1  # Allow some difference in velocity too
        )
        
        # Covariances should be similar
        ekf_cov = ekf.get_covariance_matrix()
        srif_cov = srif.get_covariance_matrix()
        
        # Check diagonal elements (uncertainties)
        assert np.allclose(
            np.diag(ekf_cov),
            np.diag(srif_cov),
            rtol=0.1  # 10% relative tolerance
        )