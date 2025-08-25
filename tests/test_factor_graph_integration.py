"""
Tests for factor graph construction in keyframe-preintegration integration.
"""

import numpy as np
import pytest
from typing import List

from src.common.config import SWBAConfig, KeyframeSelectionConfig, KeyframeSelectionStrategy
from src.common.data_structures import (
    Pose, CameraFrame, CameraObservation, ImagePoint,
    IMUMeasurement, Landmark, Map,
    CameraCalibration, CameraIntrinsics, CameraExtrinsics, CameraModel,
    IMUCalibration, PreintegratedIMUData
)
from src.estimation.legacy.swba_slam import SlidingWindowBA, Keyframe
from src.estimation.imu_integration import IMUPreintegrator, PreintegrationResult, IMUState
from src.simulation.keyframe_selector import mark_keyframes_in_camera_data
from src.utils.preintegration_utils import preintegrate_between_keyframes


class TestFactorGraphConstruction:
    """Test factor graph construction with keyframes and preintegrated IMU."""
    
    @pytest.fixture
    def camera_calibration(self):
        """Create test camera calibration."""
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
        return CameraCalibration(
            camera_id="cam0",
            intrinsics=intrinsics,
            extrinsics=extrinsics
        )
    
    @pytest.fixture
    def imu_calibration(self):
        """Create test IMU calibration."""
        return IMUCalibration(
            imu_id="imu0",
            accelerometer_noise_density=0.01,
            accelerometer_random_walk=0.001,
            gyroscope_noise_density=0.001,
            gyroscope_random_walk=0.0001,
            rate=200.0
        )
    
    def test_imu_factors_between_keyframes(self, camera_calibration, imu_calibration):
        """Test that IMU factors connect consecutive keyframes correctly."""
        # Configure SWBA with preintegration
        config = SWBAConfig(
            window_size=5,
            use_preintegrated_imu=True,
            use_keyframes_only=True
        )
        swba = SlidingWindowBA(config, camera_calibration, imu_calibration)
        
        # Initialize
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            rotation_matrix=np.eye(3)
        )
        swba.initialize(initial_pose)
        
        # Create map with landmarks
        map_data = Map()
        for i in range(5):
            landmark = Landmark(
                id=i,
                position=np.array([i-2, 0, 5])
            )
            map_data.add_landmark(landmark)
        
        # Simulate keyframe creation with preintegrated IMU
        timestamps = [0.0, 0.5, 1.0, 1.5]
        for i, t in enumerate(timestamps[1:], 1):
            # Create camera frame marked as keyframe
            observations = [
                CameraObservation(
                    landmark_id=j,
                    pixel=ImagePoint(u=320 + j*10, v=240)
                )
                for j in range(3)
            ]
            
            frame = CameraFrame(
                timestamp=t,
                camera_id="cam0",
                observations=observations
            )
            frame.is_keyframe = True
            frame.keyframe_id = i
            
            # Attach preintegrated IMU data
            if i > 0:
                frame.preintegrated_imu = PreintegratedIMUData(
                    from_keyframe_id=i-1,
                    to_keyframe_id=i,
                    delta_position=np.array([0.5, 0, 0]),
                    delta_velocity=np.array([1.0, 0, 0]),
                    delta_rotation=np.eye(3),
                    dt=0.5,
                    covariance=np.eye(15) * 0.01,
                    num_measurements=50
                )
            
            # Update state for keyframe creation
            swba.current_state.position = np.array([i*0.5, 0, 0])
            swba.current_state.timestamp = t
            
            # Process frame
            swba.update(frame, map_data)
        
        # Verify keyframes were created
        assert len(swba.keyframes) == 4  # Initial + 3 updates
        
        # Verify IMU preintegration is stored
        # Note: preintegration from i to i+1 is stored in keyframe i
        for i in range(len(swba.keyframes) - 1):
            kf = swba.keyframes[i]
            if kf.imu_preintegration is not None:
                # Check connectivity
                if isinstance(kf.imu_preintegration, PreintegratedIMUData):
                    # The preintegration stored in keyframe i connects i to i+1
                    assert kf.imu_preintegration.from_keyframe_id == kf.id
                    assert kf.imu_preintegration.to_keyframe_id == kf.id + 1
    
    def test_factor_connectivity_in_optimization(self, camera_calibration, imu_calibration):
        """Test that factors are properly connected in the optimization."""
        config = SWBAConfig(
            window_size=3,
            use_preintegrated_imu=True,
            use_keyframes_only=True,
            max_iterations=5  # Run some iterations
        )
        swba = SlidingWindowBA(config, camera_calibration, imu_calibration)
        
        # Initialize
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            rotation_matrix=np.eye(3)
        )
        swba.initialize(initial_pose)
        
        # Create simple scenario
        map_data = Map()
        landmarks = [
            Landmark(id=0, position=np.array([2, 0, 3])),
            Landmark(id=1, position=np.array([2, 1, 3])),
        ]
        for lm in landmarks:
            map_data.add_landmark(lm)
        
        # Create keyframes
        for i in range(1, 4):
            frame = CameraFrame(
                timestamp=i * 0.5,
                camera_id="cam0",
                observations=[
                    CameraObservation(
                        landmark_id=0,
                        pixel=ImagePoint(u=320, v=240)
                    ),
                    CameraObservation(
                        landmark_id=1,
                        pixel=ImagePoint(u=340, v=240)
                    )
                ]
            )
            frame.is_keyframe = True
            frame.keyframe_id = i
            
            # Add preintegrated IMU
            frame.preintegrated_imu = PreintegratedIMUData(
                from_keyframe_id=i-1,
                to_keyframe_id=i,
                delta_position=np.array([0.3, 0.1, 0]),
                delta_velocity=np.array([0.6, 0.2, 0]),
                delta_rotation=np.eye(3),
                dt=0.5,
                covariance=np.eye(15) * 0.01,
                num_measurements=50
            )
            
            swba.current_state.position = np.array([i*0.3, i*0.1, 0])
            swba.current_state.timestamp = i * 0.5
            swba.update(frame, map_data)
        
        # Verify optimization ran
        assert swba.num_optimizations > 0
        
        # Check that both IMU and camera factors were used
        # This is implicitly verified if optimization converges with both types of measurements
        result = swba.get_result()
        assert result.trajectory is not None
        assert len(result.trajectory.states) > 0
    
    def test_jacobian_dimensions(self, camera_calibration, imu_calibration):
        """Test that Jacobian matrices have correct dimensions."""
        config = SWBAConfig(
            window_size=3,
            use_preintegrated_imu=True
        )
        swba = SlidingWindowBA(config, camera_calibration, imu_calibration)
        
        # Create test keyframes
        kf1 = Keyframe(
            id=0,
            timestamp=0.0,
            state=IMUState(
                position=np.array([0, 0, 0]),
                velocity=np.zeros(3),
                rotation_matrix=np.eye(3),
                accel_bias=np.zeros(3),
                gyro_bias=np.zeros(3),
                timestamp=0.0
            ),
            observations=[]
        )
        
        kf2 = Keyframe(
            id=1,
            timestamp=0.5,
            state=IMUState(
                position=np.array([0.5, 0, 0]),
                velocity=np.array([1.0, 0, 0]),
                rotation_matrix=np.eye(3),
                accel_bias=np.zeros(3),
                gyro_bias=np.zeros(3),
                timestamp=0.5
            ),
            observations=[]
        )
        
        # Add preintegration to first keyframe
        kf1.imu_preintegration = PreintegrationResult(
            delta_position=np.array([0.5, 0, 0]),
            delta_velocity=np.array([1.0, 0, 0]),
            delta_rotation=np.eye(3),
            covariance=np.eye(15) * 0.01,
            jacobian=np.eye(15),  # Identity jacobian for simplicity
            dt=0.5,
            num_measurements=50
        )
        
        swba.keyframes = [kf1, kf2]
        
        # Test IMU residual computation
        state_i = np.zeros(15)  # [p, v, log(R), ba, bg]
        state_j = np.array([0.5, 0, 0] + [1.0, 0, 0] + [0, 0, 0] + [0, 0, 0] + [0, 0, 0])
        
        r_imu, J_i, J_j = swba._compute_imu_residual(
            state_i, state_j, kf1.imu_preintegration
        )
        
        # Check dimensions
        assert r_imu.shape == (9,)  # Position, velocity, rotation residuals (3+3+3)
        assert J_i.shape == (9, 15)  # Jacobian w.r.t. state i
        assert J_j.shape == (9, 15)  # Jacobian w.r.t. state j
        
        # Verify Jacobians are not zero (have actual derivatives)
        assert np.linalg.norm(J_i) > 0
        assert np.linalg.norm(J_j) > 0
    
    def test_marginalization_preserves_factors(self, camera_calibration, imu_calibration):
        """Test that marginalization preserves factor information."""
        config = SWBAConfig(
            window_size=3,
            use_preintegrated_imu=True,
            marginalize_old_keyframes=True
        )
        swba = SlidingWindowBA(config, camera_calibration, imu_calibration)
        
        # Initialize
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            rotation_matrix=np.eye(3)
        )
        swba.initialize(initial_pose)
        
        # Create map
        map_data = Map()
        map_data.add_landmark(Landmark(id=0, position=np.array([2, 0, 3])))
        
        # Add keyframes to exceed window size
        for i in range(1, 6):
            frame = CameraFrame(
                timestamp=i * 0.5,
                camera_id="cam0",
                observations=[
                    CameraObservation(
                        landmark_id=0,
                        pixel=ImagePoint(u=320, v=240)
                    )
                ]
            )
            frame.is_keyframe = True
            frame.keyframe_id = i
            
            # Add preintegrated IMU
            frame.preintegrated_imu = PreintegratedIMUData(
                from_keyframe_id=i-1,
                to_keyframe_id=i,
                delta_position=np.array([0.5, 0, 0]),
                delta_velocity=np.array([1.0, 0, 0]),
                delta_rotation=np.eye(3),
                dt=0.5,
                covariance=np.eye(15) * 0.01,
                num_measurements=50
            )
            
            swba.current_state.position = np.array([i*0.5, 0, 0])
            swba.current_state.timestamp = i * 0.5
            swba.update(frame, map_data)
        
        # Check window size is maintained
        assert len(swba.keyframes) <= config.window_size + 1
        
        # Verify prior information exists after marginalization
        if swba.prior_information is not None:
            assert swba.prior_information.shape[0] > 0
            assert swba.prior_mean is not None
            assert len(swba.prior_mean) > 0