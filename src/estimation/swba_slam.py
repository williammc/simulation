"""
Sliding Window Bundle Adjustment (SWBA) for Visual-Inertial SLAM.

Implements optimization-based SLAM using a sliding window of recent states,
with IMU preintegration and bundle adjustment for camera measurements.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque

from src.estimation.base_estimator import (
    BaseEstimator, EstimatorState, EstimatorConfig,
    EstimatorResult, EstimatorType
)
from src.common.config import SWBAConfig
from src.estimation.imu_integration import (
    IMUPreintegrator, PreintegrationResult, IMUState
)
from src.estimation.camera_model import (
    CameraMeasurementModel, batch_compute_reprojection_errors
)
from src.common.data_structures import (
    IMUMeasurement, CameraFrame, Map, Landmark,
    Trajectory, TrajectoryState, Pose,
    CameraCalibration, IMUCalibration
)
from src.utils.math_utils import (
    rotation_matrix_to_quaternion,
    skew, so3_exp, so3_log
)

logger = logging.getLogger(__name__)


class RobustCostType(Enum):
    """Types of robust cost functions."""
    L2 = "l2"           # Standard least squares
    HUBER = "huber"     # Huber loss
    CAUCHY = "cauchy"   # Cauchy/Lorentzian loss
    TUKEY = "tukey"     # Tukey biweight


@dataclass
class Keyframe:
    """
    Keyframe in the sliding window.
    
    Attributes:
        id: Unique keyframe identifier
        timestamp: Keyframe timestamp
        state: Robot state (pose, velocity, biases)
        observations: Camera observations at this keyframe
        imu_preintegration: Preintegrated IMU to next keyframe
        is_marginalized: Whether this keyframe has been marginalized
    """
    id: int
    timestamp: float
    state: IMUState
    observations: List['CameraObservation'] = field(default_factory=list)
    imu_preintegration: Optional[PreintegrationResult] = None
    is_marginalized: bool = False
    
    def get_pose(self) -> Pose:
        """Get pose from keyframe state."""
        return Pose(
            timestamp=self.timestamp,
            position=self.state.position.copy(),
            rotation_matrix=self.state.rotation_matrix.copy()
        )


@dataclass
class OptimizationProblem:
    """
    Bundle adjustment optimization problem.
    
    Represents the nonlinear least squares problem:
    min_x ||r(x)||^2 where r(x) are the residuals.
    """
    # State vector (stacked keyframe states and landmark positions)
    state_vector: np.ndarray
    
    # Residual vector
    residuals: np.ndarray
    
    # Jacobian matrix (sparse)
    jacobian: np.ndarray
    
    # Cost value
    cost: float
    
    # Information matrices for weighted least squares
    imu_information: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)
    camera_information: Dict[int, np.ndarray] = field(default_factory=dict)
    prior_information: Optional[np.ndarray] = None


# Config now imported from src.common.config


class SlidingWindowBA(BaseEstimator):
    """
    Sliding Window Bundle Adjustment for Visual-Inertial SLAM.
    
    Maintains a sliding window of recent keyframes and optimizes
    their states jointly using IMU and camera measurements.
    """
    
    def __init__(
        self,
        config: SWBAConfig,
        camera_calibration: CameraCalibration,
        imu_calibration: Optional[IMUCalibration] = None
    ):
        """
        Initialize SWBA.
        
        Args:
            config: SWBA configuration
            camera_calibration: Camera calibration
            imu_calibration: Optional IMU calibration
        """
        super().__init__(config)
        self.config = config
        self.camera_calib = camera_calibration
        self.imu_calib = imu_calibration
        
        # Sliding window of keyframes
        self.keyframes: deque[Keyframe] = deque(maxlen=config.window_size)
        self.next_keyframe_id = 0
        
        # Current (non-keyframe) state
        self.current_state: Optional[IMUState] = None
        
        # Landmarks in the map
        self.landmarks: Dict[int, Landmark] = {}
        self.landmark_observations: Dict[int, List[Tuple[int, 'CameraObservation']]] = {}
        
        # Camera model
        self.camera_model = CameraMeasurementModel(camera_calibration)
        
        # IMU preintegrators
        self.current_preintegrator = IMUPreintegrator() if config.use_imu_preintegration else None
        
        # Prior from marginalization
        self.prior_mean: Optional[np.ndarray] = None
        self.prior_information: Optional[np.ndarray] = None
        
        # Statistics
        self.num_optimizations = 0
        self.total_iterations = 0
    
    def initialize(
        self,
        initial_pose: Pose,
        initial_covariance: Optional[np.ndarray] = None
    ) -> None:
        """
        Initialize SWBA with first keyframe.
        
        Args:
            initial_pose: Initial robot pose
            initial_covariance: Initial uncertainty (not used in SWBA)
        """
        # Create initial state
        initial_state = IMUState(
            position=initial_pose.position,
            velocity=np.zeros(3),
            rotation_matrix=initial_pose.rotation_matrix,
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp=initial_pose.timestamp
        )
        
        # Add as first keyframe
        first_kf = Keyframe(
            id=self.next_keyframe_id,
            timestamp=initial_pose.timestamp,
            state=initial_state
        )
        self.keyframes.append(first_kf)
        self.next_keyframe_id += 1
        
        # Set current state
        self.current_state = initial_state.copy()
        
        # Reset preintegrator
        if self.current_preintegrator:
            self.current_preintegrator.reset()
        
        logger.info(f"SWBA initialized with keyframe {first_kf.id} at time {initial_pose.timestamp}")
    
    def predict(self, imu_measurements: List[IMUMeasurement], dt: float) -> None:
        """
        Process IMU measurements.
        
        For SWBA, this updates the current state and accumulates
        preintegration for the next keyframe.
        
        Args:
            imu_measurements: List of IMU measurements
            dt: Total time step
        """
        if not self.keyframes or self.current_state is None:
            logger.warning("SWBA not initialized, skipping prediction")
            return
        
        if not imu_measurements:
            return
        
        # Accumulate in preintegrator if using preintegration
        if self.config.use_imu_preintegration and self.current_preintegrator:
            for i, meas in enumerate(imu_measurements):
                if i == 0:
                    meas_dt = meas.timestamp - self.current_state.timestamp
                else:
                    meas_dt = meas.timestamp - imu_measurements[i-1].timestamp
                
                if meas_dt > 0:
                    self.current_preintegrator.add_measurement(meas, meas_dt)
        
        # Update current state timestamp
        if imu_measurements:
            self.current_state.timestamp = imu_measurements[-1].timestamp
    
    def update(self, camera_frame: CameraFrame, landmarks: Map) -> None:
        """
        Process camera measurements.
        
        Decides whether to create a new keyframe and triggers
        optimization if needed.
        
        Args:
            camera_frame: Camera observations
            landmarks: Known landmarks (for initialization)
        """
        if not self.keyframes or self.current_state is None:
            logger.warning("SWBA not initialized, skipping update")
            return
        
        # Check if we should create a new keyframe
        if self._should_create_keyframe(camera_frame.timestamp):
            self._create_keyframe(camera_frame, landmarks)
        
        # Run optimization if we have enough keyframes
        if len(self.keyframes) >= 2:
            self.optimize()
    
    def _should_create_keyframe(self, timestamp: float) -> bool:
        """
        Decide whether to create a new keyframe.
        
        Args:
            timestamp: Current timestamp
        
        Returns:
            True if a new keyframe should be created
        """
        if len(self.keyframes) == 0:
            return True
        
        last_kf = self.keyframes[-1]
        
        # Time threshold
        time_diff = timestamp - last_kf.timestamp
        if time_diff > self.config.keyframe_time_threshold:
            return True
        
        # Translation threshold
        if self.current_state:
            trans_diff = np.linalg.norm(
                self.current_state.position - last_kf.state.position
            )
            if trans_diff > self.config.keyframe_translation_threshold:
                return True
        
        # Rotation threshold
        if self.current_state:
            R_last = last_kf.state.rotation_matrix
            R_curr = self.current_state.rotation_matrix
            R_diff = R_last.T @ R_curr
            angle_diff = np.linalg.norm(so3_log(R_diff))
            if angle_diff > self.config.keyframe_rotation_threshold:
                return True
        
        return False
    
    def _create_keyframe(self, camera_frame: CameraFrame, landmarks: Map) -> None:
        """
        Create a new keyframe.
        
        Args:
            camera_frame: Camera observations for the keyframe
            landmarks: Known landmarks
        """
        # Store preintegration in previous keyframe
        if len(self.keyframes) > 0 and self.current_preintegrator:
            self.keyframes[-1].imu_preintegration = self.current_preintegrator.get_result()
            self.current_preintegrator.reset()
        
        # Create new keyframe
        new_kf = Keyframe(
            id=self.next_keyframe_id,
            timestamp=camera_frame.timestamp,
            state=self.current_state.copy(),
            observations=camera_frame.observations
        )
        self.next_keyframe_id += 1
        
        # Add landmarks from observations
        for obs in camera_frame.observations:
            # Add to landmark tracking
            if obs.landmark_id not in self.landmark_observations:
                self.landmark_observations[obs.landmark_id] = []
            self.landmark_observations[obs.landmark_id].append((new_kf.id, obs))
            
            # Initialize landmark if not known
            if obs.landmark_id not in self.landmarks:
                if obs.landmark_id in landmarks.landmarks:
                    self.landmarks[obs.landmark_id] = landmarks.landmarks[obs.landmark_id]
        
        # Add keyframe to window
        self.keyframes.append(new_kf)
        
        # Marginalize old keyframe if window is full
        if self.config.marginalize_old_keyframes and len(self.keyframes) > self.config.window_size:
            self._marginalize_oldest_keyframe()
        
        logger.debug(f"Created keyframe {new_kf.id} at time {camera_frame.timestamp}")
    
    def optimize(self) -> None:
        """
        Run bundle adjustment optimization.
        
        Jointly optimizes all keyframe states and landmark positions
        in the sliding window.
        """
        if len(self.keyframes) < 2:
            return
        
        # Build optimization problem
        problem = self._build_optimization_problem()
        
        # Solve using Gauss-Newton with Levenberg-Marquardt
        converged = self._solve_gauss_newton(problem)
        
        # Update states from solution
        if converged:
            self._update_states_from_solution(problem)
        
        self.num_optimizations += 1
    
    def _build_optimization_problem(self) -> OptimizationProblem:
        """
        Build the optimization problem.
        
        Returns:
            OptimizationProblem with state vector, residuals, and Jacobian
        """
        # Collect active landmarks (observed by current window)
        active_landmarks = self._get_active_landmarks()
        
        # Build state vector [kf_states..., landmark_positions...]
        state_dim = len(self.keyframes) * 15 + len(active_landmarks) * 3
        state_vector = np.zeros(state_dim)
        
        # Fill keyframe states
        kf_idx = 0
        for kf in self.keyframes:
            state_vector[kf_idx:kf_idx+3] = kf.state.position
            state_vector[kf_idx+3:kf_idx+6] = kf.state.velocity
            state_vector[kf_idx+6:kf_idx+9] = so3_log(
                kf.state.rotation_matrix
            )
            state_vector[kf_idx+9:kf_idx+12] = kf.state.accel_bias
            state_vector[kf_idx+12:kf_idx+15] = kf.state.gyro_bias
            kf_idx += 15
        
        # Fill landmark positions
        lm_start = len(self.keyframes) * 15
        for i, lm_id in enumerate(active_landmarks):
            if lm_id in self.landmarks:
                state_vector[lm_start + i*3:lm_start + (i+1)*3] = self.landmarks[lm_id].position
        
        # Compute residuals and Jacobian
        residuals, jacobian = self._compute_residuals_and_jacobian(
            state_vector, active_landmarks
        )
        
        # Compute cost
        cost = 0.5 * np.dot(residuals, residuals)
        
        return OptimizationProblem(
            state_vector=state_vector,
            residuals=residuals,
            jacobian=jacobian,
            cost=cost
        )
    
    def _compute_residuals_and_jacobian(
        self,
        state_vector: np.ndarray,
        active_landmarks: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute residuals and Jacobian for current state.
        
        Args:
            state_vector: Current optimization state
            active_landmarks: List of active landmark IDs
        
        Returns:
            (residuals, jacobian)
        """
        residuals = []
        jacobian_rows = []
        
        # IMU residuals between consecutive keyframes
        if self.config.use_imu_preintegration:
            for i in range(len(self.keyframes) - 1):
                kf_i = self.keyframes[i]
                kf_j = self.keyframes[i + 1]
                
                if kf_i.imu_preintegration:
                    # Compute IMU residual
                    r_imu, J_i, J_j = self._compute_imu_residual(
                        state_vector[i*15:(i+1)*15],
                        state_vector[(i+1)*15:(i+2)*15],
                        kf_i.imu_preintegration
                    )
                    
                    # Weight by IMU information
                    r_imu *= self.config.imu_weight
                    J_i *= self.config.imu_weight
                    J_j *= self.config.imu_weight
                    
                    residuals.append(r_imu)
                    
                    # Build Jacobian row
                    J_row = np.zeros((len(r_imu), len(state_vector)))
                    J_row[:, i*15:(i+1)*15] = J_i
                    J_row[:, (i+1)*15:(i+2)*15] = J_j
                    jacobian_rows.append(J_row)
        
        # Camera residuals
        lm_start = len(self.keyframes) * 15
        for lm_idx, lm_id in enumerate(active_landmarks):
            if lm_id not in self.landmark_observations:
                continue
            
            for kf_id, obs in self.landmark_observations[lm_id]:
                # Find keyframe index
                kf_idx = None
                for i, kf in enumerate(self.keyframes):
                    if kf.id == kf_id:
                        kf_idx = i
                        break
                
                if kf_idx is None:
                    continue
                
                # Get keyframe pose from state vector
                kf_state = state_vector[kf_idx*15:(kf_idx+1)*15]
                kf_pose = self._state_vector_to_pose(kf_state)
                
                # Get landmark position
                lm_pos = state_vector[lm_start + lm_idx*3:lm_start + (lm_idx+1)*3]
                
                # Compute reprojection error
                r_cam, J_pose, J_lm = self._compute_camera_residual(
                    kf_pose, lm_pos, obs
                )
                
                # Apply robust cost
                weight = self._compute_robust_weight(r_cam)
                r_cam *= weight * self.config.camera_weight
                J_pose *= weight * self.config.camera_weight
                J_lm *= weight * self.config.camera_weight
                
                residuals.append(r_cam)
                
                # Build Jacobian row
                J_row = np.zeros((2, len(state_vector)))
                # Map pose Jacobian to state vector (only position and rotation)
                J_row[:, kf_idx*15:kf_idx*15+3] = J_pose[:, 3:6]  # position
                J_row[:, kf_idx*15+6:kf_idx*15+9] = J_pose[:, 0:3]  # rotation
                J_row[:, lm_start + lm_idx*3:lm_start + (lm_idx+1)*3] = J_lm
                jacobian_rows.append(J_row)
        
        # Stack residuals and Jacobian
        if residuals:
            residuals = np.concatenate(residuals)
            jacobian = np.vstack(jacobian_rows)
        else:
            residuals = np.array([])
            jacobian = np.zeros((0, len(state_vector)))
        
        return residuals, jacobian
    
    def _compute_imu_residual(
        self,
        state_i: np.ndarray,
        state_j: np.ndarray,
        preint: PreintegrationResult
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute IMU preintegration residual.
        
        Args:
            state_i: State at keyframe i [p, v, log(R), ba, bg]
            state_j: State at keyframe j
            preint: Preintegration result
        
        Returns:
            (residual, jacobian_i, jacobian_j)
        """
        # Extract states
        p_i = state_i[0:3]
        v_i = state_i[3:6]
        theta_i = state_i[6:9]
        ba_i = state_i[9:12]
        bg_i = state_i[12:15]
        
        p_j = state_j[0:3]
        v_j = state_j[3:6]
        theta_j = state_j[6:9]
        
        R_i = so3_exp(theta_i)
        R_j = so3_exp(theta_j)
        
        # Gravity vector
        g = np.array([0, 0, -9.81])
        dt = preint.dt
        
        # Position residual
        r_p = R_i.T @ (p_j - p_i - v_i * dt - 0.5 * g * dt**2) - preint.delta_position
        
        # Velocity residual
        r_v = R_i.T @ (v_j - v_i - g * dt) - preint.delta_velocity
        
        # Rotation residual
        # Convert preintegration delta_rotation from quaternion to rotation matrix
        from src.utils.math_utils import quaternion_to_rotation_matrix
        delta_R = quaternion_to_rotation_matrix(preint.delta_rotation)
        r_R = so3_log(delta_R.T @ R_i.T @ R_j)
        
        # Stack residuals
        residual = np.concatenate([r_p, r_v, r_R])
        
        # Simplified Jacobians (ignoring second-order terms)
        J_i = np.zeros((9, 15))
        J_j = np.zeros((9, 15))
        
        # Position residual Jacobians
        J_i[0:3, 0:3] = -R_i.T
        J_i[0:3, 3:6] = -R_i.T * dt
        J_i[0:3, 6:9] = skew(R_i.T @ (p_j - p_i - v_i * dt - 0.5 * g * dt**2))
        
        J_j[0:3, 0:3] = R_i.T
        
        # Velocity residual Jacobians
        J_i[3:6, 3:6] = -R_i.T
        J_i[3:6, 6:9] = skew(R_i.T @ (v_j - v_i - g * dt))
        
        J_j[3:6, 3:6] = R_i.T
        
        # Rotation residual Jacobians
        J_i[6:9, 6:9] = -np.eye(3)
        J_j[6:9, 6:9] = np.eye(3)
        
        return residual, J_i, J_j
    
    def _compute_camera_residual(
        self,
        pose: Pose,
        landmark_pos: np.ndarray,
        observation: 'CameraObservation'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute camera reprojection residual.
        
        Args:
            pose: Camera pose
            landmark_pos: 3D landmark position
            observation: Camera observation
        
        Returns:
            (residual, pose_jacobian, landmark_jacobian)
        """
        # Create temporary landmark for projection
        temp_landmark = Landmark(id=0, position=landmark_pos)
        
        # Compute reprojection error
        error = self.camera_model.compute_reprojection_error(
            observation, temp_landmark, pose
        )
        
        return error.residual, error.jacobian_pose, error.jacobian_landmark
    
    def _compute_robust_weight(self, residual: np.ndarray) -> float:
        """
        Compute robust cost weight.
        
        Args:
            residual: Residual vector
        
        Returns:
            Weight for robust cost
        """
        r_norm = np.linalg.norm(residual)
        
        if self.config.robust_kernel == "huber":
            if r_norm <= self.config.huber_threshold:
                return 1.0
            else:
                return self.config.huber_threshold / r_norm
        elif self.config.robust_kernel == "cauchy":
            c2 = self.config.huber_threshold ** 2
            return np.sqrt(c2 / (c2 + r_norm**2))
        else:  # L2
            return 1.0
    
    def _solve_gauss_newton(self, problem: OptimizationProblem) -> bool:
        """
        Solve optimization using Gauss-Newton with Levenberg-Marquardt.
        
        Args:
            problem: Optimization problem
        
        Returns:
            True if converged
        """
        x = problem.state_vector.copy()
        lambda_lm = self.config.lambda_init
        
        for iteration in range(self.config.max_iterations):
            # Compute residuals and Jacobian
            active_landmarks = self._get_active_landmarks()
            r, J = self._compute_residuals_and_jacobian(x, active_landmarks)
            
            # Check for empty problem
            if len(r) == 0:
                logger.warning("No residuals in optimization problem")
                return False
            
            # Compute cost
            cost = 0.5 * np.dot(r, r)
            
            # Gauss-Newton normal equations: J^T J dx = -J^T r
            JtJ = J.T @ J
            Jtr = J.T @ r
            
            # Levenberg-Marquardt damping
            damped_JtJ = JtJ + lambda_lm * np.eye(len(x))
            
            try:
                # Solve for update
                dx = -np.linalg.solve(damped_JtJ, Jtr)
            except np.linalg.LinAlgError:
                logger.warning(f"Failed to solve normal equations at iteration {iteration}")
                lambda_lm *= self.config.lambda_factor
                continue
            
            # Apply update
            x_new = x + dx
            
            # Compute new cost
            r_new, _ = self._compute_residuals_and_jacobian(x_new, active_landmarks)
            cost_new = 0.5 * np.dot(r_new, r_new)
            
            # Accept or reject update
            if cost_new < cost:
                x = x_new
                lambda_lm /= self.config.lambda_factor
                
                # Check convergence
                if np.linalg.norm(dx) < self.config.convergence_threshold:
                    logger.debug(f"Converged after {iteration + 1} iterations")
                    problem.state_vector = x
                    self.total_iterations += iteration + 1
                    return True
            else:
                lambda_lm *= self.config.lambda_factor
        
        logger.debug(f"Max iterations reached ({self.config.max_iterations})")
        problem.state_vector = x
        self.total_iterations += self.config.max_iterations
        return False
    
    def _update_states_from_solution(self, problem: OptimizationProblem) -> None:
        """
        Update keyframe states and landmarks from optimization solution.
        
        Args:
            problem: Solved optimization problem
        """
        x = problem.state_vector
        
        # Update keyframe states
        for i, kf in enumerate(self.keyframes):
            state_i = x[i*15:(i+1)*15]
            kf.state.position = state_i[0:3].copy()
            kf.state.velocity = state_i[3:6].copy()
            kf.state.rotation_matrix = so3_exp(state_i[6:9])
            kf.state.accel_bias = state_i[9:12].copy()
            kf.state.gyro_bias = state_i[12:15].copy()
        
        # Update current state to latest keyframe
        if self.keyframes:
            self.current_state = self.keyframes[-1].state.copy()
        
        # Update landmarks
        active_landmarks = self._get_active_landmarks()
        lm_start = len(self.keyframes) * 15
        for i, lm_id in enumerate(active_landmarks):
            if lm_id in self.landmarks:
                self.landmarks[lm_id].position = x[lm_start + i*3:lm_start + (i+1)*3].copy()
    
    def _get_active_landmarks(self) -> List[int]:
        """
        Get landmarks observed by current window.
        
        Returns:
            List of active landmark IDs
        """
        active = set()
        current_kf_ids = {kf.id for kf in self.keyframes}
        
        for lm_id, observations in self.landmark_observations.items():
            # Check if landmark is observed by current window
            for kf_id, _ in observations:
                if kf_id in current_kf_ids:
                    active.add(lm_id)
                    break
        
        return sorted(list(active))
    
    def _state_vector_to_pose(self, state: np.ndarray) -> Pose:
        """
        Convert state vector element to Pose.
        
        Args:
            state: State vector [p, v, log(R), ba, bg]
        
        Returns:
            Pose object
        """
        return Pose(
            timestamp=0.0,  # Not used in optimization
            position=state[0:3].copy(),
            rotation_matrix=so3_exp(state[6:9])
        )
    
    def _marginalize_oldest_keyframe(self) -> None:
        """
        Marginalize the oldest keyframe using Schur complement.
        
        This removes the oldest keyframe from the window while
        preserving its information as a prior factor.
        """
        # TODO: Implement proper marginalization with Schur complement
        # For now, just remove the oldest keyframe
        
        if len(self.keyframes) > 0:
            old_kf = self.keyframes[0]
            old_kf.is_marginalized = True
            
            # Remove observations from this keyframe
            for lm_id in list(self.landmark_observations.keys()):
                self.landmark_observations[lm_id] = [
                    (kf_id, obs) for kf_id, obs in self.landmark_observations[lm_id]
                    if kf_id != old_kf.id
                ]
                
                # Remove landmark if no longer observed
                if len(self.landmark_observations[lm_id]) < self.config.min_observations_per_landmark:
                    del self.landmark_observations[lm_id]
                    if lm_id in self.landmarks:
                        del self.landmarks[lm_id]
            
            logger.debug(f"Marginalized keyframe {old_kf.id}")
    
    # Required abstract methods
    
    def get_state(self) -> EstimatorState:
        """Get current estimator state."""
        if not self.keyframes:
            raise RuntimeError("SWBA not initialized")
        
        # Use latest keyframe state
        latest_kf = self.keyframes[-1]
        
        return EstimatorState(
            timestamp=latest_kf.timestamp,
            robot_pose=latest_kf.get_pose(),
            robot_velocity=latest_kf.state.velocity.copy(),
            landmarks=self.landmarks
        )
    
    def get_trajectory(self) -> Trajectory:
        """Get estimated trajectory."""
        trajectory = Trajectory()
        
        for kf in self.keyframes:
            state = TrajectoryState(
                pose=kf.get_pose(),
                velocity=kf.state.velocity.copy()
            )
            trajectory.add_state(state)
        
        return trajectory
    
    def get_map(self) -> Map:
        """Get estimated map."""
        map_data = Map()
        for landmark in self.landmarks.values():
            map_data.add_landmark(landmark)
        return map_data
    
    def get_result(self) -> EstimatorResult:
        """Get complete estimation result."""
        return EstimatorResult(
            trajectory=self.get_trajectory(),
            landmarks=self.get_map(),
            states=[self.get_state()],
            runtime_ms=0.0,  # Would need timing
            iterations=self.total_iterations,
            converged=True,
            final_cost=0.0,
            metadata={
                "num_keyframes": len(self.keyframes),
                "num_landmarks": len(self.landmarks),
                "num_optimizations": self.num_optimizations,
                "total_iterations": self.total_iterations
            }
        )
    
    def get_state_vector(self) -> np.ndarray:
        """Get state vector."""
        if not self.keyframes:
            return np.array([])
        
        # Build state vector from keyframes
        # Convert rotation matrices to quaternions for backward compatibility
        from src.utils.math_utils import rotation_matrix_to_quaternion
        states = []
        for kf in self.keyframes:
            states.extend([
                kf.state.position,
                kf.state.velocity,
                rotation_matrix_to_quaternion(kf.state.rotation_matrix),
                kf.state.accel_bias,
                kf.state.gyro_bias
            ])
        
        return np.concatenate(states)
    
    def get_covariance_matrix(self) -> Optional[np.ndarray]:
        """
        Get covariance matrix.
        
        For SWBA, we could compute this from the information matrix,
        but it's expensive. Return None for now.
        """
        return None
    
    def marginalize(self) -> None:
        """Marginalize old states."""
        if self.config.marginalize_old_keyframes and len(self.keyframes) > self.config.window_size:
            self._marginalize_oldest_keyframe()
    
    def reset(self) -> None:
        """Reset estimator state."""
        self.keyframes.clear()
        self.next_keyframe_id = 0
        self.current_state = None
        self.landmarks.clear()
        self.landmark_observations.clear()
        if self.current_preintegrator:
            self.current_preintegrator.reset()
        self.prior_mean = None
        self.prior_information = None
        self.num_optimizations = 0
        self.total_iterations = 0