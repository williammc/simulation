"""
Abstract base class for SLAM estimators.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
import time
from enum import Enum

from src.common.data_structures import (
    Trajectory, Pose, Landmark, Map,
    IMUMeasurement, CameraFrame, CameraObservation,
    IMUCalibration, CameraCalibration
)


class EstimatorType(Enum):
    """Types of SLAM estimators."""
    EKF = "ekf"          # Extended Kalman Filter
    SWBA = "swba"        # Sliding Window Bundle Adjustment
    SRIF = "srif"        # Square Root Information Filter
    UNKNOWN = "unknown"


@dataclass
class EstimatorState:
    """
    Current state of the estimator.
    
    Attributes:
        timestamp: Current time
        robot_pose: Current robot pose estimate
        robot_velocity: Current velocity estimate (optional)
        robot_covariance: Uncertainty of robot state
        landmarks: Map of estimated landmarks
        landmark_covariances: Uncertainties of landmarks
        state_vector: Full state vector (implementation specific)
        covariance_matrix: Full covariance matrix (optional)
    """
    timestamp: float
    robot_pose: Pose
    robot_velocity: Optional[np.ndarray] = None
    robot_covariance: Optional[np.ndarray] = None
    landmarks: Optional[Map] = None
    landmark_covariances: Dict[int, np.ndarray] = field(default_factory=dict)
    state_vector: Optional[np.ndarray] = None
    covariance_matrix: Optional[np.ndarray] = None
    
    def get_trajectory_point(self) -> Pose:
        """Get current pose for trajectory."""
        return self.robot_pose
    
    def get_landmark_estimate(self, landmark_id: int) -> Optional[Landmark]:
        """Get estimated landmark by ID."""
        if self.landmarks:
            return self.landmarks.landmarks.get(landmark_id)
        return None


@dataclass
class EstimatorConfig:
    """
    Configuration for SLAM estimator.
    
    Attributes:
        estimator_type: Type of estimator
        max_landmarks: Maximum number of landmarks to track
        max_iterations: Maximum optimization iterations
        convergence_threshold: Convergence criteria
        outlier_threshold: Chi-squared threshold for outlier rejection
        enable_marginalization: Whether to marginalize old states
        marginalization_window: Size of sliding window
        verbose: Enable detailed logging
        save_intermediate: Save intermediate results
        seed: Random seed for reproducibility
    """
    estimator_type: EstimatorType = EstimatorType.UNKNOWN
    max_landmarks: int = 1000
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    outlier_threshold: float = 5.991  # Chi2 95% for 2 DOF
    enable_marginalization: bool = False
    marginalization_window: int = 20
    verbose: bool = False
    save_intermediate: bool = False
    seed: Optional[int] = None
    
    # Process noise parameters
    process_noise_position: float = 0.01
    process_noise_orientation: float = 0.001
    process_noise_velocity: float = 0.1
    process_noise_bias: float = 0.001
    
    # Measurement noise parameters
    measurement_noise_camera: float = 1.0  # pixels
    measurement_noise_imu_accel: float = 0.01
    measurement_noise_imu_gyro: float = 0.001


@dataclass
class EstimatorResult:
    """
    Result from SLAM estimation.
    
    Attributes:
        trajectory: Estimated trajectory
        landmarks: Estimated landmark map
        states: List of all states over time
        runtime_ms: Total runtime in milliseconds
        iterations: Number of iterations performed
        converged: Whether optimization converged
        final_cost: Final optimization cost
        metadata: Additional information
    """
    trajectory: Trajectory
    landmarks: Map
    states: List[EstimatorState]
    runtime_ms: float
    iterations: int
    converged: bool
    final_cost: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_final_state(self) -> EstimatorState:
        """Get final estimated state."""
        return self.states[-1] if self.states else None
    
    def get_poses(self) -> List[Pose]:
        """Extract all poses from trajectory."""
        return [state.pose for state in self.trajectory.states]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "estimator_type": self.metadata.get("estimator_type", "unknown"),
            "runtime_ms": self.runtime_ms,
            "iterations": self.iterations,
            "converged": self.converged,
            "final_cost": self.final_cost,
            "num_poses": len(self.trajectory.states),
            "num_landmarks": len(self.landmarks.landmarks),
            "metadata": self.metadata
        }


class BaseEstimator(ABC):
    """
    Abstract base class for SLAM estimators.
    
    This class defines the interface that all SLAM estimators must implement.
    Concrete implementations include EKF, SWBA, and SRIF.
    """
    
    def __init__(
        self,
        config: EstimatorConfig,
        imu_calibration: Optional[IMUCalibration] = None,
        camera_calibration: Optional[CameraCalibration] = None
    ):
        """
        Initialize base estimator.
        
        Args:
            config: Estimator configuration
            imu_calibration: IMU calibration parameters
            camera_calibration: Camera calibration parameters
        """
        self.config = config
        self.imu_calib = imu_calibration
        self.camera_calib = camera_calibration
        
        # Current state
        self.current_state: Optional[EstimatorState] = None
        self.state_history: List[EstimatorState] = []
        
        # Measurements buffer
        self.imu_buffer: List[IMUMeasurement] = []
        self.camera_buffer: List[CameraFrame] = []
        
        # Timing
        self.start_time: Optional[float] = None
        self.current_time: float = 0.0
        
        # Statistics
        self.total_iterations: int = 0
        self.total_predictions: int = 0
        self.total_updates: int = 0
        
        # Set random seed if provided
        if config.seed is not None:
            np.random.seed(config.seed)
    
    @abstractmethod
    def initialize(self, initial_pose: Pose, initial_covariance: Optional[np.ndarray] = None):
        """
        Initialize estimator with initial pose.
        
        Args:
            initial_pose: Initial robot pose
            initial_covariance: Initial uncertainty (optional)
        """
        pass
    
    @abstractmethod
    def predict(self, imu_measurements: List[IMUMeasurement], dt: float):
        """
        Prediction step using IMU measurements.
        
        Args:
            imu_measurements: List of IMU measurements
            dt: Time step
        """
        pass
    
    @abstractmethod
    def update(self, camera_frame: CameraFrame, landmarks: Map):
        """
        Update step using camera observations.
        
        Args:
            camera_frame: Camera observations at current time
            landmarks: Known landmarks (for simulation)
        """
        pass
    
    @abstractmethod
    def optimize(self) -> bool:
        """
        Perform optimization (if applicable).
        
        Returns:
            True if converged, False otherwise
        """
        pass
    
    @abstractmethod
    def marginalize(self):
        """
        Marginalize old states (for sliding window approaches).
        """
        pass
    
    @abstractmethod
    def get_state_vector(self) -> np.ndarray:
        """
        Get current state vector.
        
        Returns:
            State vector
        """
        pass
    
    @abstractmethod
    def get_covariance_matrix(self) -> Optional[np.ndarray]:
        """
        Get current covariance matrix.
        
        Returns:
            Covariance matrix (None if not maintained)
        """
        pass
    
    def process_measurements(
        self,
        imu_data: List[IMUMeasurement],
        camera_data: List[CameraFrame],
        landmarks: Map,
        ground_truth: Optional[Trajectory] = None
    ) -> EstimatorResult:
        """
        Process all measurements and return estimation result.
        
        Args:
            imu_data: All IMU measurements
            camera_data: All camera frames
            landmarks: Known landmarks
            ground_truth: Ground truth trajectory (for evaluation)
        
        Returns:
            Estimation result
        """
        start_time = time.time()
        
        # Initialize if needed
        if self.current_state is None:
            initial_pose = ground_truth.states[0].pose if ground_truth else Pose()
            self.initialize(initial_pose)
        
        # Process measurements chronologically
        imu_idx = 0
        cam_idx = 0
        
        while cam_idx < len(camera_data):
            current_cam = camera_data[cam_idx]
            
            # Collect IMU measurements up to camera time
            imu_batch = []
            while imu_idx < len(imu_data) and imu_data[imu_idx].timestamp <= current_cam.timestamp:
                imu_batch.append(imu_data[imu_idx])
                imu_idx += 1
            
            # Predict with IMU
            if imu_batch:
                dt = imu_batch[-1].timestamp - (imu_batch[0].timestamp if len(imu_batch) > 1 else self.current_time)
                self.predict(imu_batch, dt)
                self.current_time = imu_batch[-1].timestamp
            
            # Update with camera
            self.update(current_cam, landmarks)
            self.current_time = current_cam.timestamp
            
            # Save state
            self.state_history.append(self.get_current_state())
            
            # Marginalize if needed
            if self.config.enable_marginalization and len(self.state_history) > self.config.marginalization_window:
                self.marginalize()
            
            cam_idx += 1
        
        # Final optimization (for batch methods)
        converged = self.optimize()
        
        # Build result
        runtime_ms = (time.time() - start_time) * 1000
        
        # Extract trajectory from states
        trajectory = Trajectory()
        for state in self.state_history:
            from src.common.data_structures import TrajectoryState
            traj_state = TrajectoryState(
                pose=state.robot_pose,
                velocity=state.robot_velocity
            )
            trajectory.add_state(traj_state)
        
        # Get final landmarks
        final_landmarks = self.current_state.landmarks if self.current_state else Map()
        
        result = EstimatorResult(
            trajectory=trajectory,
            landmarks=final_landmarks,
            states=self.state_history,
            runtime_ms=runtime_ms,
            iterations=self.total_iterations,
            converged=converged,
            final_cost=self.compute_cost(),
            metadata={
                "estimator_type": self.config.estimator_type.value,
                "total_predictions": self.total_predictions,
                "total_updates": self.total_updates,
                "num_marginalized": len(self.state_history) - self.config.marginalization_window 
                    if self.config.enable_marginalization else 0
            }
        )
        
        return result
    
    def get_current_state(self) -> EstimatorState:
        """Get current estimator state."""
        if self.current_state is None:
            # Return default state
            return EstimatorState(
                timestamp=self.current_time,
                robot_pose=Pose(),
                state_vector=self.get_state_vector(),
                covariance_matrix=self.get_covariance_matrix()
            )
        return self.current_state
    
    def compute_cost(self) -> float:
        """
        Compute current optimization cost.
        
        Returns:
            Total cost (negative log likelihood)
        """
        # Base implementation returns 0
        # Derived classes should implement actual cost computation
        return 0.0
    
    def check_convergence(self, prev_cost: float, current_cost: float) -> bool:
        """
        Check if optimization has converged.
        
        Args:
            prev_cost: Previous iteration cost
            current_cost: Current iteration cost
        
        Returns:
            True if converged
        """
        if prev_cost == 0:
            return False
        
        relative_change = abs(current_cost - prev_cost) / abs(prev_cost)
        return relative_change < self.config.convergence_threshold
    
    def is_outlier(self, innovation: np.ndarray, S: np.ndarray) -> bool:
        """
        Check if measurement is an outlier using chi-squared test.
        
        Args:
            innovation: Innovation vector (measurement - prediction)
            S: Innovation covariance
        
        Returns:
            True if outlier
        """
        # Mahalanobis distance
        chi2 = innovation.T @ np.linalg.inv(S) @ innovation
        
        # Compare with threshold (e.g., chi2_95 for 2 DOF = 5.991)
        return chi2 > self.config.outlier_threshold