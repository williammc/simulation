"""
Test raw IMU processing in EKF estimator.
"""

import numpy as np
import pytest
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.estimation.legacy.ekf_slam import EKFSlam
from src.common.config import EKFConfig
from src.common.data_structures import (
    IMUMeasurement, CameraFrame, Pose, Map,
    CameraCalibration, CameraIntrinsics, CameraExtrinsics,
    CameraModel, IMUCalibration,
    PreintegratedIMUData
)


class TestRawIMUProcessing:
    """Test suite for raw IMU processing in EKF."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create camera calibration
        intrinsics = CameraIntrinsics(
            model=CameraModel.PINHOLE,
            width=640,
            height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            distortion=np.zeros(4)  # No distortion
        )
        extrinsics = CameraExtrinsics(
            B_T_C=np.eye(4)  # Identity transform
        )
        self.camera_calib = CameraCalibration(
            camera_id="cam0",
            intrinsics=intrinsics,
            extrinsics=extrinsics
        )
        
        # Create IMU calibration
        self.imu_calib = IMUCalibration(
            imu_id="imu0",
            accelerometer_noise_density=0.01,
            gyroscope_noise_density=0.001,
            accelerometer_random_walk=0.0001,
            gyroscope_random_walk=0.00001
        )
        
        # Create EKF config
        self.config = EKFConfig(
            gravity_magnitude=9.81,
            accel_noise_density=0.01,
            gyro_noise_density=0.001,
            accel_bias_random_walk=0.0001,
            gyro_bias_random_walk=0.00001
        )
    
    def test_raw_imu_initialization(self):
        """Test that EKF can be initialized with raw IMU mode."""
        # Create EKF with raw IMU processing
        ekf = EKFSlam(
            self.config, 
            self.camera_calib, 
            self.imu_calib,
            use_preintegrated_imu=False
        )
        
        assert ekf.use_preintegrated_imu == False
        
        # Initialize with identity pose
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            rotation_matrix=np.eye(3)
        )
        ekf.initialize(initial_pose)
        
        assert ekf.state is not None
        assert np.allclose(ekf.state.position, np.zeros(3))
        assert np.allclose(ekf.state.velocity, np.zeros(3))
    
    def test_raw_imu_static(self):
        """Test raw IMU processing with static robot (should not drift)."""
        # Create EKF with raw IMU
        ekf = EKFSlam(
            self.config,
            self.camera_calib,
            self.imu_calib,
            use_preintegrated_imu=False
        )
        
        # Initialize at origin
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            rotation_matrix=np.eye(3)
        )
        ekf.initialize(initial_pose)
        
        # Create static IMU measurements (gravity only)
        # IMU measures specific force: f = a - g
        # For static case: a = 0, so f = -g (in body frame)
        gravity_body = np.array([0, 0, 9.81])  # Upward force when static
        
        imu_measurements = []
        dt = 0.01  # 100 Hz
        for i in range(100):  # 1 second of data
            imu = IMUMeasurement(
                timestamp=i * dt,
                accelerometer=gravity_body + np.random.normal(0, 0.001, 3),  # Small noise
                gyroscope=np.random.normal(0, 0.0001, 3)  # Small noise
            )
            imu_measurements.append(imu)
        
        # Process raw IMU
        ekf.predict(imu_measurements)
        
        # Check that position hasn't drifted much
        final_position = ekf.state.position
        assert np.linalg.norm(final_position) < 0.1, f"Static drift too large: {final_position}"
    
    def test_raw_imu_linear_motion(self):
        """Test raw IMU processing with linear acceleration."""
        # Create EKF with raw IMU
        ekf = EKFSlam(
            self.config,
            self.camera_calib,
            self.imu_calib,
            use_preintegrated_imu=False
        )
        
        # Initialize at origin
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            rotation_matrix=np.eye(3)
        )
        ekf.initialize(initial_pose)
        
        # Create IMU measurements with constant forward acceleration
        # IMU measures: f = a - g
        linear_accel = np.array([1.0, 0, 0])  # 1 m/s² forward
        gravity = np.array([0, 0, -9.81])
        specific_force = linear_accel - gravity  # What IMU measures
        
        imu_measurements = []
        dt = 0.01  # 100 Hz
        for i in range(100):  # 1 second
            imu = IMUMeasurement(
                timestamp=i * dt,
                accelerometer=specific_force,
                gyroscope=np.zeros(3)  # No rotation
            )
            imu_measurements.append(imu)
        
        # Process raw IMU
        ekf.predict(imu_measurements)
        
        # After 1 second with 1 m/s² acceleration:
        # Expected velocity: v = a*t = 1.0 m/s
        # Expected position: x = 0.5*a*t² = 0.5 m
        expected_position = np.array([0.5, 0, 0])
        expected_velocity = np.array([1.0, 0, 0])
        
        assert np.allclose(ekf.state.position, expected_position, atol=0.1)
        assert np.allclose(ekf.state.velocity, expected_velocity, atol=0.1)
    
    def test_raw_vs_preintegrated(self):
        """Compare raw IMU processing with preintegrated (both should give similar results)."""
        # Create two EKF instances
        ekf_raw = EKFSlam(
            self.config,
            self.camera_calib,
            self.imu_calib,
            use_preintegrated_imu=False
        )
        
        ekf_preint = EKFSlam(
            self.config,
            self.camera_calib,
            self.imu_calib,
            use_preintegrated_imu=True
        )
        
        # Initialize both
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            rotation_matrix=np.eye(3)
        )
        ekf_raw.initialize(initial_pose)
        ekf_preint.initialize(initial_pose)
        
        # Create IMU measurements
        imu_measurements = []
        dt = 0.01
        for i in range(50):  # 0.5 seconds
            imu = IMUMeasurement(
                timestamp=i * dt,
                accelerometer=np.array([0.5, 0, 9.81]),  # Some acceleration + gravity
                gyroscope=np.array([0, 0, 0.1])  # Some rotation
            )
            imu_measurements.append(imu)
        
        # Process with raw IMU
        ekf_raw.predict(imu_measurements)
        
        # Create preintegrated data from same measurements
        # (This is simplified - normally done by simulation)
        preintegrated = PreintegratedIMUData(
            delta_position=np.array([0.0625, 0, 0]),  # Approximate
            delta_velocity=np.array([0.25, 0, 0]),    # Approximate
            delta_rotation=np.eye(3),  # Simplified
            dt=0.5,
            covariance=np.eye(15) * 0.01,
            from_keyframe_id=0,
            to_keyframe_id=1,
            num_measurements=50
        )
        
        # Process with preintegrated
        ekf_preint.predict(preintegrated)
        
        # Positions should be somewhat similar (not exact due to different integration)
        pos_diff = np.linalg.norm(ekf_raw.state.position - ekf_preint.state.position)
        assert pos_diff < 2.0, f"Raw and preintegrated differ too much: {pos_diff}"
    
    def test_raw_imu_error_handling(self):
        """Test error handling for raw IMU mode."""
        ekf = EKFSlam(
            self.config,
            self.camera_calib,
            self.imu_calib,
            use_preintegrated_imu=False
        )
        
        ekf.initialize(Pose(
            timestamp=0.0,
            position=np.zeros(3),
            rotation_matrix=np.eye(3)
        ))
        
        # Test with wrong data type (should raise error)
        with pytest.raises(TypeError):
            preintegrated = PreintegratedIMUData(
                delta_position=np.zeros(3),
                delta_velocity=np.zeros(3),
                delta_rotation=np.eye(3),
                dt=0.1,
                covariance=np.eye(15)
            )
            ekf.predict(preintegrated)  # Should fail - expecting raw IMU
        
        # Test with empty list (should handle gracefully)
        ekf.predict([])  # Should not crash
        
        # Test with mixed types in list (should raise error with our current implementation)
        mixed_list = [
            IMUMeasurement(0.0, np.zeros(3), np.zeros(3)),
            "not_an_imu",  # Invalid type
            IMUMeasurement(0.01, np.zeros(3), np.zeros(3))
        ]
        with pytest.raises(TypeError):
            ekf.predict(mixed_list)  # Should fail due to invalid type in list


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])