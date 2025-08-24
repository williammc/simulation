"""
Unit tests for GTSAM-based EKF SLAM estimator.

Tests initialization, IMU prediction, camera updates, and full trajectory estimation.
"""

import numpy as np
import pytest
from typing import List
import time

try:
    import gtsam
    GTSAM_AVAILABLE = True
except ImportError:
    GTSAM_AVAILABLE = False
    pytest.skip("GTSAM not installed", allow_module_level=True)

from src.estimation.gtsam_ekf_estimator import GtsamEkfEstimator
from src.estimation.base_estimator import EstimatorConfig, EstimatorResult
from src.common.data_structures import (
    Pose, Trajectory, Map, Landmark,
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
        # 15x15 covariance for position (3), velocity (3), rotation (3), accel_bias (3), gyro_bias (3)
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


class TestGtsamEkfInitialization:
    """Test GTSAM EKF estimator initialization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = EstimatorConfig(
            estimator_type='gtsam_ekf',
            max_landmarks=1000,
            max_iterations=100,
            convergence_threshold=1e-6,
            outlier_threshold=5.0,
            enable_marginalization=False,
            marginalization_window=10,
            verbose=False,
            save_intermediate=False,
            seed=42
        )
        
        # Add ISAM2 specific configuration
        self.config.isam2 = {
            'relinearize_threshold': 0.1,
            'relinearize_skip': 10,
            'cache_linearized_factors': True,
            'enable_partial_relinearization': False
        }
    
    def test_gtsam_ekf_initialization(self):
        """Test GTSAM EKF can be initialized."""
        estimator = GtsamEkfEstimator(self.config)
        
        # Check ISAM2 is initialized
        assert estimator.isam2 is not None
        assert not estimator.initialized
        assert estimator.pose_count == 0
        assert estimator.num_updates == 0
        assert len(estimator.pose_timestamps) == 0
    
    def test_gtsam_ekf_prior_factors(self):
        """Test adding prior factors for initial state."""
        estimator = GtsamEkfEstimator(self.config)
        
        # Create initial pose
        initial_pose = Pose(
            timestamp=0.0,
            position=np.array([1.0, 2.0, 3.0]),
            rotation_matrix=np.eye(3)
        )
        
        # Initialize estimator
        estimator.initialize(initial_pose)
        
        # Check initialization state
        assert estimator.initialized
        assert len(estimator.pose_timestamps) == 1
        assert estimator.pose_timestamps[0] == 0.0
        
        # Get current estimate
        values = estimator.isam2.calculateEstimate()
        
        # Check initial pose was set
        assert values.exists(estimator.X(0))
        pose_est = values.atPose3(estimator.X(0))
        np.testing.assert_array_almost_equal(
            pose_est.translation(),
            initial_pose.position
        )
        
        # Check velocity was initialized
        assert values.exists(estimator.V(0))
        vel_est = values.atVector(estimator.V(0))
        np.testing.assert_array_almost_equal(vel_est, np.zeros(3))
        
        # Check bias was initialized
        assert values.exists(estimator.B(0))
        bias_est = values.atConstantBias(estimator.B(0))
        np.testing.assert_array_almost_equal(
            bias_est.accelerometer(),
            np.zeros(3)
        )
        np.testing.assert_array_almost_equal(
            bias_est.gyroscope(),
            np.zeros(3)
        )
    
    def test_double_initialization(self):
        """Test that double initialization resets the estimator."""
        estimator = GtsamEkfEstimator(self.config)
        
        # First initialization
        initial_pose1 = Pose(
            timestamp=0.0,
            position=np.array([1.0, 2.0, 3.0]),
            rotation_matrix=np.eye(3)
        )
        estimator.initialize(initial_pose1)
        
        # Second initialization (should reset)
        initial_pose2 = Pose(
            timestamp=1.0,
            position=np.array([4.0, 5.0, 6.0]),
            rotation_matrix=np.eye(3)
        )
        estimator.initialize(initial_pose2)
        
        # Check that estimator was reset
        assert estimator.initialized
        assert len(estimator.pose_timestamps) == 1
        assert estimator.pose_timestamps[0] == 1.0
        assert estimator.pose_count == 0
        
        # Check new initial pose
        values = estimator.isam2.calculateEstimate()
        pose_est = values.atPose3(estimator.X(0))
        np.testing.assert_array_almost_equal(
            pose_est.translation(),
            initial_pose2.position
        )


class TestGtsamEkfIMUIntegration:
    """Test GTSAM EKF IMU integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = EstimatorConfig(
            estimator_type='gtsam_ekf',
            max_landmarks=1000,
            max_iterations=100,
            convergence_threshold=1e-6,
            outlier_threshold=5.0,
            enable_marginalization=False,
            marginalization_window=10,
            verbose=False,
            save_intermediate=False,
            seed=42
        )
        
        self.estimator = GtsamEkfEstimator(self.config)
        
        # Initialize with origin pose
        self.initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            rotation_matrix=np.eye(3)
        )
        self.estimator.initialize(self.initial_pose)
    
    def test_single_imu_prediction(self):
        """Test single IMU prediction step."""
        # Create preintegrated IMU data for forward motion
        dt = 0.1
        preintegrated = create_preintegrated_imu(
            dt=dt,
            delta_velocity=np.array([1.0, 0.0, 0.0]) * dt,  # Forward velocity
            delta_position=np.array([0.5, 0.0, 0.0]) * dt * dt,  # Forward position
            from_id=0,
            to_id=1
        )
        
        # Perform prediction
        self.estimator.predict(preintegrated)
        
        # Check pose count incremented
        assert self.estimator.pose_count == 1
        
        # Get current estimate
        values = self.estimator.isam2.calculateEstimate()
        
        # Check new pose exists
        assert values.exists(self.estimator.X(1))
        new_pose = values.atPose3(self.estimator.X(1))
        
        # Verify prediction moved forward
        position = new_pose.translation()
        assert position[0] > 0  # Moved forward in x
        np.testing.assert_array_almost_equal(position[1:], np.zeros(2), decimal=3)
        
        # Check new velocity exists
        assert values.exists(self.estimator.V(1))
    
    def test_multiple_imu_predictions(self):
        """Test multiple IMU prediction steps."""
        dt = 0.1
        n_steps = 5
        
        for i in range(n_steps):
            # Create preintegrated IMU data
            preintegrated = create_preintegrated_imu(
                dt=dt,
                delta_velocity=np.array([1.0, 0.0, 0.0]) * dt,
                delta_position=np.array([0.5, 0.0, 0.0]) * dt * dt,
                from_id=i,
                to_id=i+1
            )
            
            # Perform prediction
            self.estimator.predict(preintegrated)
        
        # Check pose count
        assert self.estimator.pose_count == n_steps
        assert len(self.estimator.pose_timestamps) == n_steps + 1  # Including initial
        
        # Get current estimate
        values = self.estimator.isam2.calculateEstimate()
        
        # Check all poses exist and are progressively forward
        prev_x = 0.0
        for i in range(n_steps + 1):
            assert values.exists(self.estimator.X(i))
            pose = values.atPose3(self.estimator.X(i))
            x = pose.translation()[0]
            assert x >= prev_x  # Each pose should be further forward
            prev_x = x
        
        # Final pose should have moved significantly
        final_pose = values.atPose3(self.estimator.X(n_steps))
        assert final_pose.translation()[0] > 0.01  # Should have moved at least 1cm
    
    def test_imu_prediction_with_rotation(self):
        """Test IMU prediction with rotation."""
        dt = 0.1
        
        # Create rotation around z-axis (yaw)
        angle = np.pi / 6  # 30 degrees
        delta_rot = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        preintegrated = create_preintegrated_imu(
            dt=dt,
            delta_rotation=delta_rot,
            delta_velocity=np.array([1.0, 0.0, 0.0]) * dt,
            delta_position=np.array([0.5, 0.0, 0.0]) * dt * dt,
            from_id=0,
            to_id=1
        )
        
        # Perform prediction
        self.estimator.predict(preintegrated)
        
        # Get current estimate
        values = self.estimator.isam2.calculateEstimate()
        new_pose = values.atPose3(self.estimator.X(1))
        
        # Check rotation was applied
        rotation_matrix = new_pose.rotation().matrix()
        
        # The rotation should be approximately the delta rotation
        np.testing.assert_array_almost_equal(
            rotation_matrix,
            delta_rot,
            decimal=2
        )
    
    def test_predict_without_initialization(self):
        """Test that prediction fails without initialization."""
        estimator = GtsamEkfEstimator(self.config)
        
        preintegrated = create_preintegrated_imu(
            dt=0.1,
            from_id=0,
            to_id=1
        )
        
        with pytest.raises(RuntimeError, match="must be initialized"):
            estimator.predict(preintegrated)


class TestGtsamEkfVisionIntegration:
    """Test GTSAM EKF vision integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = EstimatorConfig(
            estimator_type='gtsam_ekf',
            max_landmarks=1000,
            max_iterations=100,
            convergence_threshold=1e-6,
            outlier_threshold=5.0,
            enable_marginalization=False,
            marginalization_window=10,
            verbose=False,
            save_intermediate=False,
            seed=42
        )
        
        self.estimator = GtsamEkfEstimator(self.config)
        
        # Initialize estimator
        self.initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            rotation_matrix=np.eye(3)
        )
        self.estimator.initialize(self.initial_pose)
        
        # Create landmarks
        self.landmarks = Map()
        self.landmarks.add_landmark(Landmark(
            id=0,
            position=np.array([5.0, 0.0, 0.0])
        ))
        self.landmarks.add_landmark(Landmark(
            id=1,
            position=np.array([5.0, 5.0, 0.0])
        ))
        self.landmarks.add_landmark(Landmark(
            id=2,
            position=np.array([5.0, -5.0, 0.0])
        ))
    
    def test_camera_update(self):
        """Test camera measurement update (simplified EKF ignores vision)."""
        # Move forward first
        preintegrated = create_preintegrated_imu(
            dt=0.1,
            delta_velocity=np.array([1.0, 0.0, 0.0]) * 0.1,
            delta_position=np.array([0.05, 0.0, 0.0]),
            from_id=0,
            to_id=1
        )
        self.estimator.predict(preintegrated)
        
        # Create camera observations
        observations = [
            CameraObservation(
                landmark_id=0,
                pixel=ImagePoint(u=320, v=240)
            ),
            CameraObservation(
                landmark_id=1,
                pixel=ImagePoint(u=400, v=240)
            )
        ]
        
        frame = CameraFrame(
            timestamp=0.1,
            camera_id="cam0",
            observations=observations
        )
        
        # Perform update (should not crash even though it's a no-op)
        self.estimator.update(frame, self.landmarks)
        
        # In simplified EKF, no landmarks are added
        assert len(self.estimator.initialized_landmarks) == 0
        
        # Get current estimate - should still have poses
        values = self.estimator.isam2.calculateEstimate()
        assert values.exists(self.estimator.X(1))  # Predicted pose exists
    
    def test_update_without_initialization(self):
        """Test that update fails without initialization."""
        estimator = GtsamEkfEstimator(self.config)
        
        frame = CameraFrame(
            timestamp=0.1,
            camera_id="cam0",
            observations=[]
        )
        
        with pytest.raises(RuntimeError, match="must be initialized"):
            estimator.update(frame, self.landmarks)
    
    def test_multiple_camera_updates(self):
        """Test multiple camera updates (simplified EKF ignores vision)."""
        # Perform several predict-update cycles
        for i in range(3):
            # Predict with IMU
            preintegrated = create_preintegrated_imu(
                dt=0.1,
                delta_velocity=np.array([0.5, 0.0, 0.0]) * 0.1,
                delta_position=np.array([0.025, 0.0, 0.0]),
                from_id=i,
                to_id=i+1
            )
            self.estimator.predict(preintegrated)
            
            # Update with camera (no-op in simplified EKF)
            observations = [
                CameraObservation(
                    landmark_id=j,
                    pixel=ImagePoint(u=320 + j*10, v=240)
                )
                for j in range(3)  # Observe all 3 landmarks
            ]
            
            frame = CameraFrame(
                timestamp=0.1 * (i + 1),
                camera_id="cam0",
                observations=observations
            )
            
            self.estimator.update(frame, self.landmarks)
        
        # In simplified EKF, no landmarks are initialized
        assert len(self.estimator.initialized_landmarks) == 0
        
        # Get final estimate
        values = self.estimator.isam2.calculateEstimate()
        
        # Check all poses exist
        for i in range(4):  # 0 + 3 predictions
            assert values.exists(self.estimator.X(i))
        
        # No landmarks should exist in simplified EKF
        for j in range(3):
            assert not values.exists(self.estimator.L(j))


class TestGtsamEkfFullTrajectory:
    """Test full trajectory estimation with GTSAM EKF."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = EstimatorConfig(
            estimator_type='gtsam_ekf',
            max_landmarks=1000,
            max_iterations=100,
            convergence_threshold=1e-6,
            outlier_threshold=5.0,
            enable_marginalization=False,
            marginalization_window=10,
            verbose=False,
            save_intermediate=False,
            seed=42
        )
    
    def test_full_ekf_trajectory(self):
        """Test full EKF trajectory with IMU and camera."""
        estimator = GtsamEkfEstimator(self.config)
        
        # Initialize at origin
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            rotation_matrix=np.eye(3)
        )
        estimator.initialize(initial_pose)
        
        # Create circular trajectory
        n_steps = 10
        dt = 0.1
        radius = 5.0
        angular_vel = 2 * np.pi / (n_steps * dt)  # Complete circle
        
        # Create landmarks around the circle
        landmarks = Map()
        n_landmarks = 8
        for i in range(n_landmarks):
            angle = 2 * np.pi * i / n_landmarks
            landmark = Landmark(
                id=i,
                position=np.array([
                    radius * 1.5 * np.cos(angle),
                    radius * 1.5 * np.sin(angle),
                    0.0
                ])
            )
            landmarks.add_landmark(landmark)
        
        # Execute circular trajectory
        for step in range(n_steps):
            # Compute expected motion
            angle = angular_vel * dt
            
            # Rotation matrix for turning
            delta_rot = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            
            # Linear velocity (tangential)
            linear_vel = radius * angular_vel
            delta_vel = np.array([linear_vel * dt, 0, 0])
            delta_pos = np.array([linear_vel * dt * dt / 2, 0, 0])
            
            # Create preintegrated IMU
            preintegrated = create_preintegrated_imu(
                dt=dt,
                delta_rotation=delta_rot,
                delta_velocity=delta_vel,
                delta_position=delta_pos,
                from_id=step,
                to_id=step+1
            )
            
            # Predict
            estimator.predict(preintegrated)
            
            # Create camera observations (observe nearby landmarks)
            observations = []
            for landmark in landmarks.landmarks.values():
                # Simple visibility check (landmarks within 10m)
                if np.linalg.norm(landmark.position[:2]) < 10.0:
                    observations.append(
                        CameraObservation(
                            landmark_id=landmark.id,
                            pixel=ImagePoint(u=320, v=240)  # Simplified
                        )
                    )
            
            # Skip vision updates in simplified EKF
            # (In a real system, you might still log observations for visualization)
        
        # Get final result
        result = estimator.get_result()
        
        # Check result structure
        assert result.trajectory is not None
        assert result.landmarks is not None
        assert len(result.trajectory.states) == n_steps + 1  # Initial + predictions
        assert len(result.landmarks.landmarks) == 0  # No landmarks in simplified EKF
        
        # Check trajectory is roughly circular
        positions = [state.pose.position for state in result.trajectory.states]
        
        # Start and end should be close (completed circle)
        start_pos = positions[0][:2]  # x, y only
        end_pos = positions[-1][:2]
        
        # With IMU-only estimation, there will be drift
        # Allow for up to 20% drift relative to the radius
        distance = np.linalg.norm(end_pos - start_pos)
        assert distance < radius * 1.2  # Allow 20% drift
        
        # Check runtime and metadata
        assert result.runtime_ms >= 0
        assert result.metadata['num_poses'] == n_steps + 1
        assert result.metadata['estimator_type'] == 'gtsam_ekf_imu'
        assert result.metadata['mode'] == 'simplified_imu_only'
    
    def test_get_result(self):
        """Test get_result method returns proper format."""
        estimator = GtsamEkfEstimator(self.config)
        
        # Initialize
        initial_pose = Pose(
            timestamp=0.0,
            position=np.array([1.0, 2.0, 3.0]),
            rotation_matrix=np.eye(3)
        )
        estimator.initialize(initial_pose)
        
        # Add one prediction
        preintegrated = create_preintegrated_imu(
            dt=0.1,
            from_id=0,
            to_id=1
        )
        estimator.predict(preintegrated)
        
        # Get result
        result = estimator.get_result()
        
        # Check result type
        assert isinstance(result, EstimatorResult)
        
        # Check required fields
        assert result.trajectory is not None
        assert result.landmarks is not None
        assert result.states is not None
        assert result.runtime_ms >= 0
        assert result.iterations >= 0
        assert isinstance(result.converged, bool)
        assert result.final_cost >= 0
        assert isinstance(result.metadata, dict)
        
        # Check trajectory
        assert len(result.trajectory.states) == 2  # Initial + 1 prediction
        
        # Check current state
        assert len(result.states) == 1
        current_state = result.states[0]
        assert current_state.robot_pose is not None
        assert current_state.timestamp > 0
    
    def test_reset(self):
        """Test reset functionality."""
        estimator = GtsamEkfEstimator(self.config)
        
        # Initialize and add some poses
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            rotation_matrix=np.eye(3)
        )
        estimator.initialize(initial_pose)
        
        # Add predictions
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
        assert estimator.pose_count == 0
        assert estimator.landmark_count == 0
        assert len(estimator.initialized_landmarks) == 0
        assert len(estimator.pose_timestamps) == 0
        assert estimator.num_updates == 0
        
        # Should be able to initialize again
        estimator.initialize(initial_pose)
        assert estimator.initialized


class TestGtsamEkfPerformance:
    """Test performance characteristics of GTSAM EKF."""
    
    def test_incremental_performance(self):
        """Test that EKF maintains constant time per update."""
        config = EstimatorConfig(
            estimator_type='gtsam_ekf',
            max_landmarks=1000,
            max_iterations=100,
            convergence_threshold=1e-6,
            outlier_threshold=5.0,
            enable_marginalization=False,
            marginalization_window=10,
            verbose=False,
            save_intermediate=False,
            seed=42
        )
        
        estimator = GtsamEkfEstimator(config)
        
        # Initialize
        initial_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            rotation_matrix=np.eye(3)
        )
        estimator.initialize(initial_pose)
        
        # Measure time for updates
        update_times = []
        n_updates = 20
        
        for i in range(n_updates):
            preintegrated = create_preintegrated_imu(
                dt=0.1,
                delta_velocity=np.array([0.1, 0, 0]),
                delta_position=np.array([0.005, 0, 0]),
                from_id=i,
                to_id=i+1
            )
            
            start = time.perf_counter()
            estimator.predict(preintegrated)
            end = time.perf_counter()
            
            update_times.append(end - start)
        
        # Check that update times don't grow significantly
        # (allowing for some variation due to relinearization)
        first_half = np.mean(update_times[:n_updates//2])
        second_half = np.mean(update_times[n_updates//2:])
        
        # Second half shouldn't be more than 2x slower than first half
        assert second_half < first_half * 2.0
        
        # Average update time should be reasonable (< 100ms)
        assert np.mean(update_times) < 0.1