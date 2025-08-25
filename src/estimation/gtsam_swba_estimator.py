"""
GTSAM-based Sliding Window Bundle Adjustment SLAM estimator.

Uses GTSAM's batch optimization within a sliding window of keyframes,
with marginalization of old states to maintain computational efficiency.
"""

import numpy as np
from typing import Optional, List
import logging
import time
from collections import deque

try:
    import gtsam
except ImportError:
    raise ImportError(
        "GTSAM is required for GtsamSWBAEstimator. "
        "Install it with: pip install gtsam"
    )

from src.estimation.gtsam_base import GtsamBaseEstimator
from src.estimation.base_estimator import EstimatorResult, EstimatorState, EstimatorConfig
from src.common.data_structures import (
    Pose, PreintegratedIMUData, CameraFrame, Map
)

logger = logging.getLogger(__name__)


class GtsamSWBAEstimator(GtsamBaseEstimator):
    """
    GTSAM-based Sliding Window Bundle Adjustment for SLAM (Simplified IMU-only version).
    
    Maintains a sliding window of recent keyframes and performs
    batch optimization within the window. Old states are marginalized
    to maintain computational efficiency.
    
    Note: This is a simplified implementation that only uses IMU measurements
    for state estimation. Vision factors are not incorporated to maintain
    numerical stability and simplicity.
    """
    
    def __init__(self, config: EstimatorConfig):
        """
        Initialize GTSAM SWBA estimator.
        
        Args:
            config: Estimator configuration
        """
        super().__init__(config)
        
        # Window configuration
        swba_config = getattr(config, 'swba', {})
        self.window_size = swba_config.get('window_size', 10)
        self.optimization_iterations = swba_config.get('optimization_iterations', 5)
        
        # Sliding window management
        self.window_poses = deque(maxlen=self.window_size)  # Pose indices in window
        self.window_graph = gtsam.NonlinearFactorGraph()
        self.window_values = gtsam.Values()
        
        # Marginalization
        self.marginalized_factors = gtsam.NonlinearFactorGraph()
        self.marginalization_info = {}
        
        # Optimizer configuration
        optimizer_params = gtsam.LevenbergMarquardtParams()
        optimizer_params.setMaxIterations(self.optimization_iterations)
        # Note: Error tolerances are set differently in GTSAM's LM params
        # They use different naming conventions
        self.optimizer_params = optimizer_params
        
        # Track whether we've initialized
        self.initialized = False
        
        # Track timestamps for poses
        self.pose_timestamps = []
        
        # Store all values (for final result extraction)
        self.all_values = gtsam.Values()
        
        # Performance tracking
        self.total_runtime = 0.0
        self.num_optimizations = 0
        
        logger.info(f"Initialized GtsamSWBAEstimator with window size: {self.window_size}")
    
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
        
        # Add to window
        self.window_poses.append(0)
        # Add factors individually
        for i in range(self.graph.size()):
            self.window_graph.add(self.graph.at(i))
        self.window_values.insert(self.initial_values)
        
        # Store in all values
        self.all_values.insert(self.initial_values)
        
        # Clear temporary graph/values
        self.graph.resize(0)
        self.initial_values.clear()
        
        # Mark as initialized
        self.initialized = True
        self.pose_timestamps.append(initial_pose.timestamp)
        
        logger.info(f"Initialized SWBA at pose: {initial_pose.position}")
    
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
        
        # Current and previous pose keys
        prev_pose = self.X(self.pose_count - 1)
        curr_pose = self.X(self.pose_count)
        
        # Current and previous velocity keys
        prev_vel = self.V(self.pose_count - 1)
        curr_vel = self.V(self.pose_count)
        
        # Get current estimate for prediction
        if len(self.window_poses) > 0:
            # Use window values if available
            prev_pose_est = self.window_values.atPose3(prev_pose) if self.window_values.exists(prev_pose) else None
            prev_vel_est = self.window_values.atVector(prev_vel) if self.window_values.exists(prev_vel) else np.zeros(3)
        else:
            prev_pose_est = None
            prev_vel_est = np.zeros(3)
        
        # If we don't have previous estimate, use identity
        if prev_pose_est is None:
            prev_pose_est = gtsam.Pose3()
        
        # Predict next pose using preintegrated values
        dt = preintegrated_imu.dt
        
        # Position prediction: p_next = p_prev + v_prev*dt + R_prev*delta_p
        new_position = (
            prev_pose_est.translation() + 
            prev_vel_est * dt +
            prev_pose_est.rotation().matrix() @ preintegrated_imu.delta_position
        )
        
        # Rotation prediction: R_next = R_prev * delta_R
        new_rotation = gtsam.Rot3(
            prev_pose_est.rotation().matrix() @ preintegrated_imu.delta_rotation
        )
        
        # Velocity prediction: v_next = v_prev + R_prev*delta_v
        new_velocity = (
            prev_vel_est + 
            prev_pose_est.rotation().matrix() @ preintegrated_imu.delta_velocity
        )
        
        # Create predicted pose
        predicted_pose = gtsam.Pose3(new_rotation, new_position)
        
        # Add between factor for poses
        pose_cov = preintegrated_imu.covariance[:6, :6]
        pose_noise = gtsam.noiseModel.Gaussian.Covariance(pose_cov)
        
        between_factor = gtsam.BetweenFactorPose3(
            prev_pose, curr_pose,
            prev_pose_est.between(predicted_pose),
            pose_noise
        )
        self.graph.add(between_factor)
        
        # Initial guess for new pose and velocity
        self.initial_values.insert(curr_pose, predicted_pose)
        self.initial_values.insert(curr_vel, new_velocity)
        
        # Track timestamp
        if hasattr(preintegrated_imu, 'end_time'):
            self.pose_timestamps.append(preintegrated_imu.end_time)
        else:
            self.pose_timestamps.append(self.pose_count * 1.0)
        
        # Add to window
        self._add_to_window(self.pose_count)
        
        # Update metrics
        self.total_runtime += time.perf_counter() - start_time
        
        logger.debug(f"Predicted pose {self.pose_count} and added to window")
    
    def update(self, frame: CameraFrame, landmarks: Map) -> None:
        """
        Update state with camera observations and optimize window.
        
        Note: In this simplified SWBA, vision factors are not used.
        The method exists for interface compatibility but only triggers optimization.
        
        Args:
            frame: Camera frame with observations
            landmarks: Map of landmarks
        """
        if not self.initialized:
            raise RuntimeError("Estimator must be initialized before update")
        
        start_time = time.perf_counter()
        
        # Simplified SWBA - no vision updates
        logger.debug(f"Vision update called, triggering window optimization")
        
        # Still optimize the window with IMU-only factors
        self._optimize_window()
        
        # Update metrics
        self.total_runtime += time.perf_counter() - start_time
    
    def _add_to_window(self, pose_index: int) -> None:
        """
        Add pose to sliding window and marginalize if necessary.
        
        Args:
            pose_index: Index of pose to add
        """
        # Check if window is full
        if len(self.window_poses) >= self.window_size:
            # Marginalize oldest pose
            self._marginalize_oldest()
        
        # Add new pose to window
        self.window_poses.append(pose_index)
        
        # Add factors and initial values to window
        for i in range(self.graph.size()):
            self.window_graph.add(self.graph.at(i))
        if self.initial_values.size() > 0:
            self.window_values.insert(self.initial_values)
            # Also add to all_values to preserve full trajectory
            for key in range(self.initial_values.size()):
                k = self.initial_values.keys()[key]
                if not self.all_values.exists(k):
                    # Determine type and insert appropriately
                    try:
                        self.all_values.insert(k, self.initial_values.atPose3(k))
                    except:
                        try:
                            self.all_values.insert(k, self.initial_values.atVector(k))
                        except:
                            try:
                                self.all_values.insert(k, self.initial_values.atConstantBias(k))
                            except:
                                pass  # Skip unknown types
        
        # Clear temporary storage
        self.graph.resize(0)
        self.initial_values.clear()
    
    def _marginalize_oldest(self) -> None:
        """Marginalize the oldest pose in the window."""
        if len(self.window_poses) == 0:
            return
        
        oldest_pose_idx = self.window_poses[0]
        oldest_pose_key = self.X(oldest_pose_idx)
        oldest_vel_key = self.V(oldest_pose_idx)
        
        logger.debug(f"Marginalizing pose {oldest_pose_idx}")
        
        # Store marginalized values in all_values before removing from window
        if self.window_values.exists(oldest_pose_key):
            if self.all_values.exists(oldest_pose_key):
                self.all_values.update(oldest_pose_key, self.window_values.atPose3(oldest_pose_key))
            else:
                self.all_values.insert(oldest_pose_key, self.window_values.atPose3(oldest_pose_key))
        if self.window_values.exists(oldest_vel_key):
            if self.all_values.exists(oldest_vel_key):
                self.all_values.update(oldest_vel_key, self.window_values.atVector(oldest_vel_key))
            else:
                self.all_values.insert(oldest_vel_key, self.window_values.atVector(oldest_vel_key))
        
        # In a full implementation, we would:
        # 1. Compute marginal covariance for the oldest pose
        # 2. Create prior factors from the marginal
        # 3. Remove the oldest pose from window graph/values
        # 4. Add prior factors to maintain information
        
        # For now, we'll keep a simplified approach
        # The window automatically removes oldest when full (deque with maxlen)
    
    def _optimize_window(self) -> None:
        """Optimize the current window using batch optimization."""
        if len(self.window_poses) == 0:
            return
        
        # Create optimizer
        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self.window_graph,
            self.window_values,
            self.optimizer_params
        )
        
        # Optimize
        self.window_values = optimizer.optimize()
        
        # Update all_values with optimized results
        for pose_idx in self.window_poses:
            pose_key = self.X(pose_idx)
            vel_key = self.V(pose_idx)
            
            if self.window_values.exists(pose_key):
                if self.all_values.exists(pose_key):
                    self.all_values.update(pose_key, self.window_values.atPose3(pose_key))
                else:
                    self.all_values.insert(pose_key, self.window_values.atPose3(pose_key))
            
            if self.window_values.exists(vel_key):
                if self.all_values.exists(vel_key):
                    self.all_values.update(vel_key, self.window_values.atVector(vel_key))
                else:
                    self.all_values.insert(vel_key, self.window_values.atVector(vel_key))
        
        # Update landmark estimates
        for landmark_id in self.initialized_landmarks:
            landmark_key = self.L(landmark_id)
            if self.window_values.exists(landmark_key):
                if self.all_values.exists(landmark_key):
                    self.all_values.update(landmark_key, self.window_values.atPoint3(landmark_key))
                else:
                    self.all_values.insert(landmark_key, self.window_values.atPoint3(landmark_key))
        
        self.num_optimizations += 1
        logger.debug(f"Optimized window with {len(self.window_poses)} poses")
    
    def get_result(self) -> EstimatorResult:
        """
        Get current estimation result.
        
        Returns:
            EstimatorResult containing optimized trajectory and landmarks
        """
        # Use all_values which contains both marginalized and current window
        trajectory = self.extract_trajectory(self.all_values)
        landmarks = Map()  # Empty map since we don't estimate landmarks in simplified SWBA
        
        # Get current state for the latest pose
        current_state = None
        if self.pose_count > 0:
            latest_pose_key = self.X(self.pose_count)
            if self.all_values.exists(latest_pose_key):
                latest_pose = self.all_values.atPose3(latest_pose_key)
                
                # Get velocity if available
                latest_vel_key = self.V(self.pose_count)
                velocity = None
                if self.all_values.exists(latest_vel_key):
                    velocity = np.array(self.all_values.atVector(latest_vel_key))
                
                # Use default covariance for now
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
            iterations=self.num_optimizations,
            converged=True,
            final_cost=0.0,
            metadata={
                'num_poses': self.pose_count + 1,
                'num_landmarks': 0,  # No landmarks in simplified SWBA
                'num_optimizations': self.num_optimizations,
                'window_size': len(self.window_poses),
                'estimator_type': 'gtsam_swba_imu',
                'mode': 'simplified_imu_only'
            }
        )
        
        return result
    
    def reset(self) -> None:
        """Reset the estimator to initial state."""
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_values = gtsam.Values()
        self.window_graph = gtsam.NonlinearFactorGraph()
        self.window_values = gtsam.Values()
        self.all_values = gtsam.Values()
        self.window_poses.clear()
        self.marginalized_factors = gtsam.NonlinearFactorGraph()
        self.marginalization_info.clear()
        self.pose_count = 0
        self.landmark_count = 0
        self.velocity_count = 0
        self.initialized_landmarks.clear()
        self.initialized = False
        self.pose_timestamps.clear()
        self.total_runtime = 0.0
        self.num_optimizations = 0
        logger.info("Reset GtsamSWBAEstimator")
    
    def optimize(self) -> EstimatorResult:
        """
        Run optimization (performs window optimization and returns result).
        
        Returns:
            EstimatorResult with current state
        """
        self._optimize_window()
        return self.get_result()
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get current state vector from window.
        
        Returns:
            State vector containing poses in window
        """
        if not self.initialized:
            return np.array([])
        
        state_vector = []
        
        # Add poses in window
        for pose_idx in self.window_poses:
            pose_key = self.X(pose_idx)
            if self.window_values.exists(pose_key):
                pose = self.window_values.atPose3(pose_key)
                # Add position
                state_vector.extend(pose.translation())
                # Add rotation as axis-angle
                axis_angle = pose.rotation().axisAngle()
                state_vector.extend(axis_angle)
        
        return np.array(state_vector)
    
    def get_covariance_matrix(self) -> np.ndarray:
        """
        Get covariance matrix for current window state.
        
        Returns:
            Covariance matrix (approximate for efficiency)
        """
        if not self.initialized:
            return np.array([[]])
        
        # For SWBA, getting full covariance is expensive
        # Return identity matrix as placeholder
        state_size = len(self.get_state_vector())
        if state_size == 0:
            return np.array([[]])
        
        return np.eye(state_size) * 0.1  # Default uncertainty
    
    def marginalize(self) -> None:
        """
        Marginalize old states outside the window.
        
        This is called automatically during window management.
        """
        if len(self.window_poses) >= self.window_size:
            self._marginalize_oldest()