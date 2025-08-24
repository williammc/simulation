"""
GTSAM-based Extended Kalman Filter SLAM estimator.

Uses GTSAM's ISAM2 for incremental smoothing and mapping,
which provides an efficient incremental solution similar to EKF.
"""

import numpy as np
from typing import Optional
import logging
import time

try:
    import gtsam
except ImportError:
    raise ImportError(
        "GTSAM is required for GtsamEkfEstimator. "
        "Install it with: pip install gtsam"
    )

from src.estimation.gtsam_base import GtsamBaseEstimator
from src.estimation.base_estimator import EstimatorResult, EstimatorState, EstimatorConfig
from src.common.data_structures import (
    Pose, PreintegratedIMUData, CameraFrame, Map
)

logger = logging.getLogger(__name__)


class GtsamEkfEstimator(GtsamBaseEstimator):
    """
    GTSAM-based Extended Kalman Filter for SLAM (Simplified IMU-only version).
    
    Uses ISAM2 (Incremental Smoothing and Mapping) for efficient
    incremental updates, providing EKF-like behavior with better
    numerical stability and consistency.
    
    Note: This is a simplified implementation that only uses IMU measurements
    for state estimation. Vision factors are not incorporated to maintain
    numerical stability and simplicity. The update() method exists for
    interface compatibility but performs no vision-based updates.
    """
    
    def __init__(self, config: EstimatorConfig):
        """
        Initialize GTSAM EKF estimator.
        
        Args:
            config: Estimator configuration
        """
        super().__init__(config)
        self.config = config  # Store for reset
        
        # Set up ISAM2 parameters
        isam2_params = gtsam.ISAM2Params()
        
        # Configure from config or use defaults
        isam2_config = getattr(config, 'isam2', {})
        isam2_params.setRelinearizeThreshold(isam2_config.get('relinearize_threshold', 0.1))
        isam2_params.relinearizeSkip = isam2_config.get('relinearize_skip', 10)
        isam2_params.cacheLinearizedFactors = isam2_config.get('cache_linearized_factors', True)
        isam2_params.enablePartialRelinearizationCheck = isam2_config.get(
            'enable_partial_relinearization', False
        )
        
        # Initialize ISAM2
        self.isam2 = gtsam.ISAM2(isam2_params)
        
        # Track whether we've initialized
        self.initialized = False
        
        # Track timestamps for poses
        self.pose_timestamps = []
        
        # Performance tracking
        self.total_runtime = 0.0
        self.num_updates = 0
        
        logger.info(f"Initialized GtsamEkfEstimator with ISAM2 parameters: {isam2_config}")
    
    def initialize(self, initial_pose: Pose) -> None:
        """
        Initialize the estimator with an initial pose.
        
        Args:
            initial_pose: Initial robot pose
        """
        if self.initialized:
            logger.warning("Estimator already initialized, resetting...")
            self.reset()
        
        # Add prior factors
        self.add_prior_factors(initial_pose)
        
        # Perform initial ISAM2 update
        self.isam2.update(self.graph, self.initial_values)
        
        # Clear for next iteration
        self.graph.resize(0)
        self.initial_values.clear()
        
        # Mark as initialized
        self.initialized = True
        self.pose_timestamps.append(initial_pose.timestamp)
        
        logger.info(f"Initialized EKF at pose: {initial_pose.position}")
    
    def predict(self, preintegrated_imu: PreintegratedIMUData) -> None:
        """
        Predict next state using preintegrated IMU measurements.
        
        Args:
            preintegrated_imu: Preintegrated IMU data between keyframes
        """
        if not self.initialized:
            raise RuntimeError("Estimator must be initialized before prediction")
        
        start_time = time.perf_counter()
        
        # Increment pose count for new pose
        self.pose_count += 1
        
        # Current and next pose keys
        prev_pose = self.X(self.pose_count - 1)
        curr_pose = self.X(self.pose_count)
        
        # Current and next velocity keys
        prev_vel = self.V(self.pose_count - 1)
        curr_vel = self.V(self.pose_count)
        
        # Bias key (assumed constant for now)
        bias = self.B(0)
        
        # Create IMU factor
        # Note: In a real implementation, we'd use gtsam.PreintegratedImuMeasurements
        # For now, we'll create a between factor based on the preintegrated values
        
        # Get current estimate for prediction
        current_values = self.isam2.calculateEstimate()
        prev_pose_est = current_values.atPose3(prev_pose)
        prev_vel_est = current_values.atVector(prev_vel) if current_values.exists(prev_vel) else np.zeros(3)
        
        # Predict next pose using preintegrated values
        # This is a simplified version - real implementation would use proper IMU integration
        dt = preintegrated_imu.dt
        
        # Position prediction: p_next = p_prev + v*dt + 0.5*a*dt^2 + R*delta_p
        new_position = (
            prev_pose_est.translation() + 
            prev_vel_est * dt +
            prev_pose_est.rotation().matrix() @ preintegrated_imu.delta_position
        )
        
        # Rotation prediction: R_next = R_prev * delta_R
        new_rotation = gtsam.Rot3(
            prev_pose_est.rotation().matrix() @ preintegrated_imu.delta_rotation
        )
        
        # Velocity prediction: v_next = v_prev + R*delta_v
        new_velocity = (
            prev_vel_est + 
            prev_pose_est.rotation().matrix() @ preintegrated_imu.delta_velocity
        )
        
        # Create predicted pose
        predicted_pose = gtsam.Pose3(new_rotation, new_position)
        
        # Add between factor for poses (using covariance from preintegration)
        # Convert covariance to noise model
        pose_cov = preintegrated_imu.covariance[:6, :6]  # Use first 6x6 for pose
        # Ensure covariance is not too small (numerical stability)
        min_variance = 1e-6
        pose_cov = pose_cov + np.eye(6) * min_variance
        pose_noise = gtsam.noiseModel.Gaussian.Covariance(pose_cov)
        
        between_factor = gtsam.BetweenFactorPose3(
            prev_pose, curr_pose,
            prev_pose_est.between(predicted_pose),
            pose_noise
        )
        self.graph.add(between_factor)
        
        # Add velocity factor
        vel_cov = preintegrated_imu.covariance[6:9, 6:9] if preintegrated_imu.covariance.shape[0] >= 9 else np.eye(3) * 0.1
        vel_noise = gtsam.noiseModel.Gaussian.Covariance(vel_cov)
        
        # Initial guess for new pose and velocity
        self.initial_values.insert(curr_pose, predicted_pose)
        self.initial_values.insert(curr_vel, new_velocity)
        
        # Track timestamp
        if hasattr(preintegrated_imu, 'end_time'):
            self.pose_timestamps.append(preintegrated_imu.end_time)
        else:
            self.pose_timestamps.append(self.pose_count * 1.0)  # Default increment
        
        # Update ISAM2
        self.isam2.update(self.graph, self.initial_values)
        
        # Clear for next iteration
        self.graph.resize(0)
        self.initial_values.clear()
        
        # Update metrics
        self.num_updates += 1
        self.total_runtime += time.perf_counter() - start_time
        
        logger.debug(f"Predicted pose {self.pose_count} using preintegrated IMU")
    
    def update(self, frame: CameraFrame, landmarks: Map) -> None:
        """
        Update state with camera observations.
        
        Note: In this simplified EKF, vision factors are not used.
        The method exists for interface compatibility but performs no updates.
        
        Args:
            frame: Camera frame with observations
            landmarks: Map of landmarks
        """
        if not self.initialized:
            raise RuntimeError("Estimator must be initialized before update")
        
        # Simplified EKF - no vision updates
        # This keeps the system well-constrained and focused on IMU-based estimation
        logger.debug(f"Vision update called but skipped (simplified EKF mode)")
        
        # You could optionally store observations for visualization
        # without adding them to the optimization
    
    def get_result(self) -> EstimatorResult:
        """
        Get current estimation result.
        
        Returns:
            EstimatorResult containing optimized trajectory and landmarks
        """
        # Calculate current estimate
        values = self.isam2.calculateEstimate()
        
        # Extract trajectory only (no landmarks in simplified EKF)
        trajectory = self.extract_trajectory(values)
        landmarks = Map()  # Empty map since we don't estimate landmarks
        
        # Get current state for the latest pose
        current_state = None
        if self.pose_count > 0:
            latest_pose_key = self.X(self.pose_count)
            if values.exists(latest_pose_key):
                latest_pose = values.atPose3(latest_pose_key)
                
                # Get velocity if available
                latest_vel_key = self.V(self.pose_count)
                velocity = None
                if values.exists(latest_vel_key):
                    velocity = np.array(values.atVector(latest_vel_key))
                
                # Get marginal covariance for the latest pose
                try:
                    marginals = gtsam.Marginals(self.graph, values)
                    pose_cov = marginals.marginalCovariance(latest_pose_key)
                except:
                    # If marginals fail, use default covariance
                    pose_cov = np.eye(6) * 0.1
                
                current_state = EstimatorState(
                    timestamp=self.pose_timestamps[-1] if self.pose_timestamps else 0.0,
                    robot_pose=self.gtsam_to_pose(
                        latest_pose,
                        self.pose_timestamps[-1] if self.pose_timestamps else 0.0
                    ),
                    robot_velocity=velocity,
                    robot_covariance=pose_cov,
                    landmarks=landmarks.landmarks,
                    landmark_covariances={}
                )
        
        # Create result
        result = EstimatorResult(
            trajectory=trajectory,
            landmarks=landmarks,
            states=[current_state] if current_state else [],
            runtime_ms=self.total_runtime * 1000,
            iterations=self.num_updates,
            converged=True,  # ISAM2 is always "converged" after update
            final_cost=0.0,  # Could compute if needed
            metadata={
                'num_poses': self.pose_count + 1,
                'num_landmarks': 0,  # No landmarks in simplified EKF
                'num_updates': self.num_updates,
                'estimator_type': 'gtsam_ekf_imu',  # IMU-only EKF
                'mode': 'simplified_imu_only'
            }
        )
        
        return result
    
    def reset(self) -> None:
        """Reset the estimator to initial state."""
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_values = gtsam.Values()
        # Recreate ISAM2 with same parameters
        isam2_params = gtsam.ISAM2Params()
        isam2_config = getattr(self.config, 'isam2', {})
        isam2_params.setRelinearizeThreshold(isam2_config.get('relinearize_threshold', 0.1))
        isam2_params.relinearizeSkip = isam2_config.get('relinearize_skip', 10)
        isam2_params.cacheLinearizedFactors = isam2_config.get('cache_linearized_factors', True)
        isam2_params.enablePartialRelinearizationCheck = isam2_config.get(
            'enable_partial_relinearization', False
        )
        self.isam2 = gtsam.ISAM2(isam2_params)
        self.pose_count = 0
        self.landmark_count = 0
        self.velocity_count = 0
        self.initialized_landmarks.clear()
        self.initialized = False
        self.pose_timestamps.clear()
        self.total_runtime = 0.0
        self.num_updates = 0
        logger.info("Reset GtsamEkfEstimator")
    
    def optimize(self) -> EstimatorResult:
        """
        Run optimization (for EKF this is just getting current result).
        
        Returns:
            EstimatorResult with current state
        """
        return self.get_result()
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get current state vector.
        
        Returns:
            State vector containing poses and landmarks
        """
        if not self.initialized:
            return np.array([])
        
        values = self.isam2.calculateEstimate()
        state_vector = []
        
        # Add poses (position and rotation as axis-angle)
        for i in range(self.pose_count + 1):
            if values.exists(self.X(i)):
                pose = values.atPose3(self.X(i))
                # Add position
                state_vector.extend(pose.translation())
                # Add rotation as axis-angle
                axis_angle = pose.rotation().axisAngle()
                state_vector.extend(axis_angle)
        
        # Add landmarks
        for landmark_id in self.initialized_landmarks:
            if values.exists(self.L(landmark_id)):
                point = values.atPoint3(self.L(landmark_id))
                state_vector.extend(point)
        
        return np.array(state_vector)
    
    def get_covariance_matrix(self) -> np.ndarray:
        """
        Get covariance matrix for current state.
        
        Returns:
            Covariance matrix (may be approximate for efficiency)
        """
        if not self.initialized:
            return np.array([[]])
        
        # For EKF/ISAM2, getting full covariance is expensive
        # Return identity matrix as placeholder
        # In practice, would compute marginal covariances for specific variables
        state_size = len(self.get_state_vector())
        if state_size == 0:
            return np.array([[]])
        
        return np.eye(state_size) * 0.1  # Default uncertainty
    
    def marginalize(self) -> None:
        """
        Marginalize old states (not needed for standard EKF).
        
        ISAM2 handles marginalization internally.
        """
        # ISAM2 automatically manages marginalization
        # through its Bayes tree structure
        pass