"""
Base class for GTSAM-based SLAM estimators.

This module provides common functionality for all GTSAM-based estimators,
including factor graph management, data conversion utilities, and result extraction.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from abc import abstractmethod
import logging

try:
    import gtsam
except ImportError:
    raise ImportError(
        "GTSAM is required for GTSAM-based estimators. "
        "Install it with: pip install gtsam"
    )

from src.estimation.base_estimator import BaseEstimator, EstimatorResult, EstimatorState, EstimatorConfig
from src.common.data_structures import (
    Pose, Trajectory, TrajectoryState, Map, Landmark,
    PreintegratedIMUData, CameraFrame, CameraObservation
)

# Set up logging
logger = logging.getLogger(__name__)


class GtsamBaseEstimator(BaseEstimator):
    """
    Base class for GTSAM-based SLAM estimators.
    
    Provides common functionality including:
    - Factor graph initialization and management
    - Symbol generation for variables
    - Data conversion between custom types and GTSAM types
    - Noise model creation
    - Result extraction from optimized values
    """
    
    def __init__(self, config: EstimatorConfig):
        """
        Initialize GTSAM base estimator.
        
        Args:
            config: Estimator configuration
        """
        super().__init__(config)
        
        # Initialize factor graph and values
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_values = gtsam.Values()
        
        # Symbol counters
        self.pose_count = 0
        self.landmark_count = 0
        self.velocity_count = 0
        
        # Symbol prefixes
        self.POSE_KEY = ord('x')  # X for poses
        self.LANDMARK_KEY = ord('l')  # L for landmarks  
        self.VELOCITY_KEY = ord('v')  # V for velocities
        self.BIAS_KEY = ord('b')  # B for biases
        
        # Noise models from config
        self._setup_noise_models(config)
        
        # Track initialized landmarks
        self.initialized_landmarks = set()
        
        logger.info(f"Initialized {self.__class__.__name__} with config: {config}")
    
    def _setup_noise_models(self, config: EstimatorConfig) -> None:
        """
        Set up noise models from configuration.
        
        Args:
            config: Estimator configuration
        """
        # Default noise parameters (can be overridden by config)
        noise_config = getattr(config, 'noise_models', {})
        
        # Prior noise for initial pose
        prior_pose_sigmas = noise_config.get(
            'prior_pose', 
            [0.1, 0.1, 0.1, 0.05, 0.05, 0.05]  # x,y,z,roll,pitch,yaw
        )
        self.prior_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array(prior_pose_sigmas)
        )
        
        # Prior noise for initial velocity
        prior_vel_sigmas = noise_config.get('prior_velocity', [0.1, 0.1, 0.1])
        self.prior_velocity_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array(prior_vel_sigmas)
        )
        
        # Prior noise for IMU biases
        prior_bias_sigmas = noise_config.get(
            'prior_bias',
            [0.01, 0.01, 0.01, 0.001, 0.001, 0.001]  # accel_bias, gyro_bias
        )
        self.prior_bias_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array(prior_bias_sigmas)
        )
        
        # Projection noise for camera measurements
        projection_sigmas = noise_config.get('projection_noise', [1.0, 1.0])  # pixels
        self.projection_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array(projection_sigmas)
        )
    
    # Symbol generation methods
    
    def X(self, i: int) -> int:
        """Generate symbol for pose i."""
        return gtsam.symbol(chr(self.POSE_KEY), i)
    
    def L(self, j: int) -> int:
        """Generate symbol for landmark j."""
        return gtsam.symbol(chr(self.LANDMARK_KEY), j)
    
    def V(self, i: int) -> int:
        """Generate symbol for velocity i."""
        return gtsam.symbol(chr(self.VELOCITY_KEY), i)
    
    def B(self, i: int) -> int:
        """Generate symbol for IMU bias i."""
        return gtsam.symbol(chr(self.BIAS_KEY), i)
    
    # Data conversion utilities
    
    def pose_to_gtsam(self, pose: Pose) -> gtsam.Pose3:
        """
        Convert custom Pose to GTSAM Pose3.
        
        Args:
            pose: Custom Pose object
            
        Returns:
            GTSAM Pose3 object
        """
        rotation = gtsam.Rot3(pose.rotation_matrix)
        translation = gtsam.Point3(pose.position)
        return gtsam.Pose3(rotation, translation)
    
    def gtsam_to_pose(self, gtsam_pose: gtsam.Pose3, timestamp: float = 0.0) -> Pose:
        """
        Convert GTSAM Pose3 to custom Pose.
        
        Args:
            gtsam_pose: GTSAM Pose3 object
            timestamp: Timestamp for the pose
            
        Returns:
            Custom Pose object
        """
        rotation_matrix = gtsam_pose.rotation().matrix()
        position = gtsam_pose.translation()
        
        return Pose(
            timestamp=timestamp,
            position=np.array(position),
            rotation_matrix=rotation_matrix
        )
    
    def extract_trajectory(self, values: gtsam.Values) -> Trajectory:
        """
        Extract trajectory from GTSAM values.
        
        Args:
            values: Optimized GTSAM values
            
        Returns:
            Trajectory object
        """
        trajectory = Trajectory()
        
        # Extract all poses (including initial pose at index 0)
        for i in range(self.pose_count + 1):
            key = self.X(i)
            if values.exists(key):
                gtsam_pose = values.atPose3(key)
                pose = self.gtsam_to_pose(gtsam_pose, timestamp=float(i))
                
                # Try to get velocity if it exists
                velocity = None
                vel_key = self.V(i)
                if values.exists(vel_key):
                    velocity = np.array(values.atVector(vel_key))
                
                state = TrajectoryState(
                    pose=pose,
                    velocity=velocity
                )
                trajectory.add_state(state)
        
        return trajectory
    
    def extract_landmarks(self, values: gtsam.Values) -> Map:
        """
        Extract landmarks from GTSAM values.
        
        Args:
            values: Optimized GTSAM values
            
        Returns:
            Map object containing landmarks
        """
        landmark_map = Map()
        
        # Extract all landmarks
        for landmark_id in self.initialized_landmarks:
            key = self.L(landmark_id)
            if values.exists(key):
                position = np.array(values.atPoint3(key))
                landmark = Landmark(
                    id=landmark_id,
                    position=position
                )
                landmark_map.add_landmark(landmark)
        
        return landmark_map
    
    def add_prior_factors(self, initial_pose: Pose) -> None:
        """
        Add prior factors for initial state.
        
        Args:
            initial_pose: Initial robot pose
        """
        # Add prior on initial pose
        self.graph.add(gtsam.PriorFactorPose3(
            self.X(0),
            self.pose_to_gtsam(initial_pose),
            self.prior_pose_noise
        ))
        
        # Add prior on initial velocity (zero)
        self.graph.add(gtsam.PriorFactorVector(
            self.V(0),
            np.zeros(3),
            self.prior_velocity_noise
        ))
        
        # Add prior on initial IMU bias (zero)
        self.graph.add(gtsam.PriorFactorConstantBias(
            self.B(0),
            gtsam.imuBias.ConstantBias(),  # Zero bias
            self.prior_bias_noise
        ))
        
        # Initialize values
        self.initial_values.insert(
            self.X(0),
            self.pose_to_gtsam(initial_pose)
        )
        self.initial_values.insert(
            self.V(0),
            np.zeros(3)
        )
        self.initial_values.insert(
            self.B(0),
            gtsam.imuBias.ConstantBias()
        )
        
        logger.debug(f"Added prior factors for initial pose at {initial_pose.position}")
    
    def add_imu_factor(self, preintegrated: PreintegratedIMUData) -> None:
        """
        Add preintegrated IMU factor between consecutive poses.
        
        Args:
            preintegrated: Preintegrated IMU measurements
        """
        # This is a placeholder - specific implementations will override
        # based on their IMU handling strategy
        raise NotImplementedError("Subclasses must implement add_imu_factor")
    
    def add_vision_factor(
        self, 
        pose_key: int,
        observation: CameraObservation,
        landmark: Landmark,
        camera_calibration: Optional[gtsam.Cal3_S2] = None
    ) -> None:
        """
        Add projection factor for landmark observation.
        
        Args:
            pose_key: Symbol key for the pose
            observation: Camera observation
            landmark: Landmark being observed
            camera_calibration: Camera calibration (optional)
        """
        # Default camera calibration if not provided
        if camera_calibration is None:
            camera_calibration = gtsam.Cal3_S2(
                500, 500,  # fx, fy
                0,         # skew
                320, 240   # cx, cy
            )
        
        # Check if landmark is initialized
        landmark_key = self.L(landmark.id)
        if landmark.id not in self.initialized_landmarks:
            # Initialize landmark
            self.initial_values.insert(
                landmark_key,
                gtsam.Point3(landmark.position)
            )
            self.initialized_landmarks.add(landmark.id)
        
        # Create projection factor
        measured = gtsam.Point2(
            observation.pixel.u,
            observation.pixel.v
        )
        
        factor = gtsam.GenericProjectionFactorCal3_S2(
            measured,
            self.projection_noise,
            pose_key,
            landmark_key,
            camera_calibration
        )
        
        self.graph.add(factor)
    
    def get_result(self) -> EstimatorResult:
        """
        Get current estimation result.
        
        Returns:
            EstimatorResult containing trajectory and landmarks
        """
        # This will be implemented by subclasses based on their
        # optimization strategy (incremental vs batch)
        raise NotImplementedError("Subclasses must implement get_result")
    
    @abstractmethod
    def initialize(self, initial_pose: Pose) -> None:
        """Initialize the estimator with an initial pose."""
        pass
    
    @abstractmethod
    def predict(self, preintegrated_imu: PreintegratedIMUData) -> None:
        """Predict next state using preintegrated IMU."""
        pass
    
    @abstractmethod
    def update(self, frame: CameraFrame, landmarks: Map) -> None:
        """Update state with camera observations."""
        pass