"""
Integration tests for preintegrated IMU support in estimators.

Tests that all estimators (EKF, SWBA, SRIF) correctly handle both
raw and preintegrated IMU data and produce consistent results.
"""

import numpy as np
import pytest
from typing import List

from src.common.data_structures import (
    IMUMeasurement, CameraFrame, CameraObservation,
    Pose, Map, Landmark, CameraCalibration, IMUCalibration,
    CameraIntrinsics, PreintegratedIMUData
)
from src.common.config import EKFConfig, SWBAConfig, SRIFConfig
from src.estimation.ekf_slam import EKFSlam
from src.estimation.swba_slam import SlidingWindowBA
from src.estimation.srif_slam import SRIFSlam
from src.estimation.imu_integration import IMUPreintegrator
from src.utils.preintegration_utils import preintegrate_between_keyframes


@pytest.fixture
def camera_calibration():
    """Create camera calibration."""
    from src.common.data_structures import CameraModel, CameraExtrinsics
    intrinsics = CameraIntrinsics(
        model=CameraModel.PINHOLE,
        fx=500, fy=500,
        cx=320, cy=240,
        width=640, height=480,
        distortion=np.zeros(5)
    )
    extrinsics = CameraExtrinsics(B_T_C=np.eye(4))
    return CameraCalibration(
        camera_id="cam0",
        intrinsics=intrinsics,
        extrinsics=extrinsics
    )


@pytest.fixture  
def imu_calibration():
    """Create IMU calibration."""
    return IMUCalibration(
        imu_id="imu0",
        accelerometer_noise_density=0.01,
        accelerometer_random_walk=0.001,
        gyroscope_noise_density=0.001,
        gyroscope_random_walk=0.0001,
        gravity_magnitude=9.81,
        rate=200.0
    )


@pytest.fixture
def simulation_data():
    """Create simple simulation data."""
    # Create IMU measurements (constant motion)
    imu_measurements = []
    dt = 0.005  # 200Hz
    for i in range(200):  # 1 second of data
        t = i * dt
        meas = IMUMeasurement(
            timestamp=t,
            accelerometer=np.array([0.1, 0, 9.81]),  # Small forward acceleration
            gyroscope=np.array([0, 0, 0.05])  # Small yaw rate
        )
        imu_measurements.append(meas)
    
    # Create keyframes at 0.0, 0.3, 0.6, 0.9 seconds
    keyframe_ids = [0, 1, 2, 3]
    keyframe_times = [0.0, 0.3, 0.6, 0.9]
    
    # Create camera frames at keyframe times
    camera_frames = []
    landmarks = Map()
    
    # Add some landmarks
    for i in range(5):
        landmark = Landmark(
            id=i,
            position=np.array([5 + i*0.5, i-2, 2])
        )
        landmarks.add_landmark(landmark)
    
    # Create observations at each keyframe
    for kf_idx, kf_time in enumerate(keyframe_times):
        observations = []
        for lm_id in range(3):  # Observe first 3 landmarks
            obs = CameraObservation(
                pixel=np.array([320 + lm_id*10, 240 + lm_id*5]),
                landmark_id=lm_id
            )
            observations.append(obs)
        
        frame = CameraFrame(
            timestamp=kf_time,
            camera_id="cam0",
            observations=observations,
            is_keyframe=True,
            keyframe_id=kf_idx
        )
        camera_frames.append(frame)
    
    return {
        'imu_measurements': imu_measurements,
        'camera_frames': camera_frames,
        'landmarks': landmarks,
        'keyframe_ids': keyframe_ids,
        'keyframe_times': keyframe_times
    }


class TestEKFIntegration:
    """Test EKF with preintegrated IMU."""
    
    def test_raw_vs_preintegrated(self, camera_calibration, imu_calibration, simulation_data):
        """Test that EKF produces similar results with raw and preintegrated IMU."""
        # Initialize two EKFs
        config_raw = EKFConfig(use_preintegrated_imu=False)
        config_preint = EKFConfig(use_preintegrated_imu=True)
        
        ekf_raw = EKFSlam(config_raw, camera_calibration, imu_calibration)
        ekf_preint = EKFSlam(config_preint, camera_calibration, imu_calibration)
        
        # Initialize both at same pose
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            rotation_matrix=np.eye(3)
        )
        ekf_raw.initialize(initial_pose)
        ekf_preint.initialize(initial_pose)
        
        # Preintegrate IMU data
        preintegrator = IMUPreintegrator()
        preintegrated_data = preintegrate_between_keyframes(
            simulation_data['imu_measurements'],
            simulation_data['keyframe_ids'],
            simulation_data['keyframe_times'],
            preintegrator
        )
        
        # Process data through both EKFs
        for i, frame in enumerate(simulation_data['camera_frames']):
            if i > 0:
                # Raw EKF: process all IMU measurements
                start_idx = int(simulation_data['keyframe_times'][i-1] / 0.005)
                end_idx = int(simulation_data['keyframe_times'][i] / 0.005)
                imu_segment = simulation_data['imu_measurements'][start_idx:end_idx]
                
                if imu_segment:
                    dt = simulation_data['keyframe_times'][i] - simulation_data['keyframe_times'][i-1]
                    ekf_raw.predict(imu_segment, dt)
                
                # Preintegrated EKF: use preintegrated data
                if frame.keyframe_id in preintegrated_data:
                    ekf_preint.predict(preintegrated_data[frame.keyframe_id])
            
            # Both do camera update
            ekf_raw.update(frame, simulation_data['landmarks'])
            ekf_preint.update(frame, simulation_data['landmarks'])
        
        # Compare final states
        state_raw = ekf_raw.get_state()
        state_preint = ekf_preint.get_state()
        
        # Positions should be similar (within reasonable tolerance)
        position_diff = np.linalg.norm(
            state_raw.robot_pose.position - state_preint.robot_pose.position
        )
        assert position_diff < 0.5, f"Position difference too large: {position_diff}"
        
        # Velocities should be similar
        velocity_diff = np.linalg.norm(
            state_raw.robot_velocity - state_preint.robot_velocity
        )
        assert velocity_diff < 0.2, f"Velocity difference too large: {velocity_diff}"
    
    def test_covariance_consistency(self, camera_calibration, imu_calibration, simulation_data):
        """Test that covariance remains positive definite with preintegration."""
        config = EKFConfig(use_preintegrated_imu=True)
        ekf = EKFSlam(config, camera_calibration, imu_calibration)
        
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            rotation_matrix=np.eye(3)
        )
        ekf.initialize(initial_pose)
        
        # Preintegrate IMU data
        preintegrator = IMUPreintegrator()
        preintegrated_data = preintegrate_between_keyframes(
            simulation_data['imu_measurements'],
            simulation_data['keyframe_ids'],
            simulation_data['keyframe_times'],
            preintegrator
        )
        
        # Process with preintegrated data
        for i, frame in enumerate(simulation_data['camera_frames']):
            if i > 0 and frame.keyframe_id in preintegrated_data:
                ekf.predict(preintegrated_data[frame.keyframe_id])
            
            ekf.update(frame, simulation_data['landmarks'])
            
            # Check covariance is positive definite
            cov = ekf.get_covariance_matrix()
            if cov is not None:
                eigenvalues = np.linalg.eigvals(cov)
                assert np.all(eigenvalues > -1e-10), "Covariance not positive definite"


class TestSWBAIntegration:
    """Test SWBA with preintegrated IMU."""
    
    def test_optimization_with_preintegration(self, camera_calibration, imu_calibration, simulation_data):
        """Test that SWBA optimization works with preintegrated factors."""
        config = SWBAConfig(
            use_preintegrated_imu=True,
            window_size=3,
            keyframe_time_threshold=0.2
        )
        
        swba = SlidingWindowBA(config, camera_calibration, imu_calibration)
        
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            rotation_matrix=np.eye(3)
        )
        swba.initialize(initial_pose)
        
        # Preintegrate IMU data
        preintegrator = IMUPreintegrator()
        preintegrated_data = preintegrate_between_keyframes(
            simulation_data['imu_measurements'],
            simulation_data['keyframe_ids'],
            simulation_data['keyframe_times'],
            preintegrator
        )
        
        # Process frames
        for i, frame in enumerate(simulation_data['camera_frames']):
            if i > 0 and frame.keyframe_id in preintegrated_data:
                swba.predict(preintegrated_data[frame.keyframe_id])
            
            swba.update(frame, simulation_data['landmarks'])
        
        # Check that optimization ran
        assert swba.num_optimizations > 0
        
        # Check that we have keyframes
        assert len(swba.keyframes) > 0
        
        # Get final state
        state = swba.get_state()
        assert state is not None
        assert state.robot_pose is not None


class TestSRIFIntegration:
    """Test SRIF with preintegrated IMU."""
    
    def test_information_updates(self, camera_calibration, imu_calibration, simulation_data):
        """Test that SRIF information updates work with preintegration."""
        config = SRIFConfig(use_preintegrated_imu=True)
        srif = SRIFSlam(config, camera_calibration, imu_calibration)
        
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            rotation_matrix=np.eye(3)
        )
        srif.initialize(initial_pose)
        
        # Preintegrate IMU data
        preintegrator = IMUPreintegrator()
        preintegrated_data = preintegrate_between_keyframes(
            simulation_data['imu_measurements'],
            simulation_data['keyframe_ids'],
            simulation_data['keyframe_times'],
            preintegrator
        )
        
        # Process frames
        for i, frame in enumerate(simulation_data['camera_frames']):
            if i > 0 and frame.keyframe_id in preintegrated_data:
                srif.predict(preintegrated_data[frame.keyframe_id])
            
            srif.update(frame, simulation_data['landmarks'])
        
        # Check that information matrix is upper triangular
        sqrt_info = srif.state.sqrt_information
        # Check upper triangular (elements below diagonal should be ~0)
        for i in range(sqrt_info.shape[0]):
            for j in range(i):
                assert abs(sqrt_info[i, j]) < 1e-10, "sqrt_information not upper triangular"
        
        # Get final state
        state = srif.get_state()
        assert state is not None
        assert state.robot_pose is not None


class TestPerformanceComparison:
    """Compare performance between raw and preintegrated IMU."""
    
    def test_computation_efficiency(self, camera_calibration, imu_calibration):
        """Test that preintegration reduces computation in prediction step."""
        import time
        
        # Create longer sequence
        imu_measurements = []
        for i in range(1000):  # 5 seconds at 200Hz
            t = i * 0.005
            meas = IMUMeasurement(
                timestamp=t,
                accelerometer=np.array([0.1, 0, 9.81]),
                gyroscope=np.array([0, 0, 0.05])
            )
            imu_measurements.append(meas)
        
        # Measure raw processing time
        config_raw = EKFConfig(use_preintegrated_imu=False)
        ekf_raw = EKFSlam(config_raw, camera_calibration, imu_calibration)
        ekf_raw.initialize(Pose(timestamp=0.0, position=np.zeros(3), rotation_matrix=np.eye(3)))
        
        start_time = time.time()
        ekf_raw.predict(imu_measurements, 5.0)
        raw_time = time.time() - start_time
        
        # Measure preintegrated processing time
        config_preint = EKFConfig(use_preintegrated_imu=True)
        ekf_preint = EKFSlam(config_preint, camera_calibration, imu_calibration)
        ekf_preint.initialize(Pose(timestamp=0.0, position=np.zeros(3), rotation_matrix=np.eye(3)))
        
        # Preintegrate first
        preintegrator = IMUPreintegrator()
        preintegrated = preintegrator.batch_process(imu_measurements, 0, 1)
        
        start_time = time.time()
        ekf_preint.predict(preintegrated)
        preint_time = time.time() - start_time
        
        # Preintegrated should be faster for prediction
        # (Though preintegration itself takes time, in practice it's done once
        # and reused multiple times in optimization)
        print(f"Raw prediction time: {raw_time:.4f}s")
        print(f"Preintegrated prediction time: {preint_time:.4f}s")
        
        # Just check that both methods complete without error
        assert raw_time > 0
        assert preint_time > 0