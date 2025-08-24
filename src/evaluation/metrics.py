"""
Evaluation metrics for SLAM trajectory estimation.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

from src.common.data_structures import Trajectory, Pose
from src.estimation.base_estimator import EstimatorState


@dataclass
class TrajectoryMetrics:
    """
    Trajectory error metrics.
    
    Attributes:
        ate_rmse: Absolute Trajectory Error (Root Mean Square Error)
        ate_mean: ATE mean
        ate_median: ATE median
        ate_std: ATE standard deviation
        ate_min: ATE minimum
        ate_max: ATE maximum
        rpe_trans_rmse: Relative Pose Error translation RMSE
        rpe_trans_mean: RPE translation mean
        rpe_trans_median: RPE translation median
        rpe_trans_std: RPE translation standard deviation
        rpe_rot_rmse: Relative Pose Error rotation RMSE (radians)
        rpe_rot_mean: RPE rotation mean (radians)
        rpe_rot_median: RPE rotation median (radians)
        rpe_rot_std: RPE rotation standard deviation (radians)
        trajectory_length: Total trajectory length
        num_poses: Number of poses evaluated
    """
    # ATE metrics
    ate_rmse: float = 0.0
    ate_mean: float = 0.0
    ate_median: float = 0.0
    ate_std: float = 0.0
    ate_min: float = 0.0
    ate_max: float = 0.0
    
    # RPE translation metrics
    rpe_trans_rmse: float = 0.0
    rpe_trans_mean: float = 0.0
    rpe_trans_median: float = 0.0
    rpe_trans_std: float = 0.0
    
    # RPE rotation metrics
    rpe_rot_rmse: float = 0.0
    rpe_rot_mean: float = 0.0
    rpe_rot_median: float = 0.0
    rpe_rot_std: float = 0.0
    
    # Additional info
    trajectory_length: float = 0.0
    num_poses: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ate": {
                "rmse": self.ate_rmse,
                "mean": self.ate_mean,
                "median": self.ate_median,
                "std": self.ate_std,
                "min": self.ate_min,
                "max": self.ate_max
            },
            "rpe_translation": {
                "rmse": self.rpe_trans_rmse,
                "mean": self.rpe_trans_mean,
                "median": self.rpe_trans_median,
                "std": self.rpe_trans_std
            },
            "rpe_rotation": {
                "rmse": self.rpe_rot_rmse,
                "mean": self.rpe_rot_mean,
                "median": self.rpe_rot_median,
                "std": self.rpe_rot_std
            },
            "trajectory_length": self.trajectory_length,
            "num_poses": self.num_poses
        }


@dataclass
class ConsistencyMetrics:
    """
    Consistency metrics for state estimation.
    
    Attributes:
        nees_mean: Normalized Estimation Error Squared mean
        nees_median: NEES median
        nees_std: NEES standard deviation
        nees_chi2_percentage: Percentage within chi-squared bounds
        is_consistent: Whether estimator is consistent (NEES test)
    """
    nees_mean: float = 0.0
    nees_median: float = 0.0
    nees_std: float = 0.0
    nees_chi2_percentage: float = 0.0
    is_consistent: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "nees_mean": self.nees_mean,
            "nees_median": self.nees_median,
            "nees_std": self.nees_std,
            "chi2_percentage": self.nees_chi2_percentage,
            "is_consistent": self.is_consistent
        }


def align_trajectories(
    estimated: Trajectory,
    ground_truth: Trajectory,
    use_sim3: bool = False
) -> Tuple[Trajectory, np.ndarray]:
    """
    Align estimated trajectory to ground truth using Horn's method.
    
    Args:
        estimated: Estimated trajectory
        ground_truth: Ground truth trajectory
        use_sim3: If True, estimate scale as well (7 DOF), else SE3 (6 DOF)
    
    Returns:
        Aligned estimated trajectory and transformation matrix
    """
    # Extract positions
    est_positions = np.array([state.pose.position for state in estimated.states])
    gt_positions = np.array([state.pose.position for state in ground_truth.states])
    
    # Ensure same length
    min_len = min(len(est_positions), len(gt_positions))
    est_positions = est_positions[:min_len]
    gt_positions = gt_positions[:min_len]
    
    # Compute centroids
    est_centroid = np.mean(est_positions, axis=0)
    gt_centroid = np.mean(gt_positions, axis=0)
    
    # Center the point clouds
    est_centered = est_positions - est_centroid
    gt_centered = gt_positions - gt_centroid
    
    # Compute scale if requested
    scale = 1.0
    if use_sim3:
        est_scale = np.sqrt(np.sum(est_centered**2))
        gt_scale = np.sqrt(np.sum(gt_centered**2))
        scale = gt_scale / est_scale if est_scale > 0 else 1.0
        est_centered *= scale
    
    # Compute rotation using SVD
    H = est_centered.T @ gt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = gt_centroid - scale * R @ est_centroid
    
    # Build transformation matrix
    T = np.eye(4)
    T[:3, :3] = R * scale
    T[:3, 3] = t
    
    # Apply transformation to estimated trajectory
    aligned = Trajectory()
    for state in estimated.states:
        # Transform position
        pos_hom = np.append(state.pose.position, 1.0)
        aligned_pos = (T @ pos_hom)[:3]
        
        # Transform rotation (only apply R, not scale)
        R_est = state.pose.rotation_matrix
        R_aligned = R @ R_est
        
        # Create aligned pose
        aligned_pose = Pose(
            timestamp=state.pose.timestamp,
            position=aligned_pos,
            rotation_matrix=R_aligned
        )
        
        # Create aligned state
        from src.common.data_structures import TrajectoryState
        aligned_state = TrajectoryState(
            pose=aligned_pose,
            velocity=state.velocity
        )
        aligned.add_state(aligned_state)
    
    return aligned, T


def compute_ate(
    estimated: Trajectory,
    ground_truth: Trajectory,
    align: bool = True
) -> Tuple[np.ndarray, TrajectoryMetrics]:
    """
    Compute Absolute Trajectory Error (ATE).
    
    ATE measures the absolute distance between estimated and ground truth poses.
    
    Args:
        estimated: Estimated trajectory
        ground_truth: Ground truth trajectory
        align: Whether to align trajectories first
    
    Returns:
        Array of errors per pose and metrics summary
    """
    # Align if requested
    if align:
        estimated, _ = align_trajectories(estimated, ground_truth)
    
    # Match estimated poses to ground truth by timestamp
    errors = []
    
    # Build ground truth lookup by timestamp for efficiency
    gt_by_time = {state.pose.timestamp: state for state in ground_truth.states}
    
    # For each estimated pose, find closest ground truth
    for est_state in estimated.states:
        est_pos = est_state.pose.position
        est_time = est_state.pose.timestamp
        
        # Find exact or closest ground truth match
        if est_time in gt_by_time:
            gt_state = gt_by_time[est_time]
        else:
            # Find closest timestamp
            closest_time = min(gt_by_time.keys(), key=lambda t: abs(t - est_time))
            gt_state = gt_by_time[closest_time]
        
        gt_pos = gt_state.pose.position
        error = np.linalg.norm(est_pos - gt_pos)
        errors.append(error)
    
    min_len = len(estimated.states)  # Use estimated length for metrics
    
    errors = np.array(errors)
    
    # Compute trajectory length
    traj_length = 0.0
    for i in range(1, min_len):
        prev_pos = ground_truth.states[i-1].pose.position
        curr_pos = ground_truth.states[i].pose.position
        traj_length += np.linalg.norm(curr_pos - prev_pos)
    
    # Build metrics
    metrics = TrajectoryMetrics(
        ate_rmse=np.sqrt(np.mean(errors**2)),
        ate_mean=np.mean(errors),
        ate_median=np.median(errors),
        ate_std=np.std(errors),
        ate_min=np.min(errors) if len(errors) > 0 else 0.0,
        ate_max=np.max(errors) if len(errors) > 0 else 0.0,
        trajectory_length=traj_length,
        num_poses=min_len
    )
    
    return errors, metrics


def compute_rpe(
    estimated: Trajectory,
    ground_truth: Trajectory,
    delta: int = 1,
    align: bool = False
) -> Tuple[np.ndarray, np.ndarray, TrajectoryMetrics]:
    """
    Compute Relative Pose Error (RPE).
    
    RPE measures the error in relative motion between poses.
    
    Args:
        estimated: Estimated trajectory
        ground_truth: Ground truth trajectory
        delta: Frame delta for computing relative poses
        align: Whether to align trajectories first (usually False for RPE)
    
    Returns:
        Translation errors, rotation errors, and metrics summary
    """
    if align:
        estimated, _ = align_trajectories(estimated, ground_truth)
    
    # Ensure same length
    min_len = min(len(estimated.states), len(ground_truth.states))
    
    trans_errors = []
    rot_errors = []
    
    for i in range(delta, min_len):
        # Get relative transformations
        est_rel = compute_relative_pose(
            estimated.states[i-delta].pose,
            estimated.states[i].pose
        )
        gt_rel = compute_relative_pose(
            ground_truth.states[i-delta].pose,
            ground_truth.states[i].pose
        )
        
        # Compute error transformation
        error_T = np.linalg.inv(gt_rel) @ est_rel
        
        # Translation error
        trans_error = np.linalg.norm(error_T[:3, 3])
        trans_errors.append(trans_error)
        
        # Rotation error (angle)
        R_error = error_T[:3, :3]
        angle_error = rotation_matrix_to_angle(R_error)
        rot_errors.append(angle_error)
    
    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)
    
    # Compute trajectory length
    traj_length = 0.0
    for i in range(1, min_len):
        prev_pos = ground_truth.states[i-1].pose.position
        curr_pos = ground_truth.states[i].pose.position
        traj_length += np.linalg.norm(curr_pos - prev_pos)
    
    # Build metrics
    metrics = TrajectoryMetrics(
        rpe_trans_rmse=np.sqrt(np.mean(trans_errors**2)) if len(trans_errors) > 0 else 0.0,
        rpe_trans_mean=np.mean(trans_errors) if len(trans_errors) > 0 else 0.0,
        rpe_trans_median=np.median(trans_errors) if len(trans_errors) > 0 else 0.0,
        rpe_trans_std=np.std(trans_errors) if len(trans_errors) > 0 else 0.0,
        rpe_rot_rmse=np.sqrt(np.mean(rot_errors**2)) if len(rot_errors) > 0 else 0.0,
        rpe_rot_mean=np.mean(rot_errors) if len(rot_errors) > 0 else 0.0,
        rpe_rot_median=np.median(rot_errors) if len(rot_errors) > 0 else 0.0,
        rpe_rot_std=np.std(rot_errors) if len(rot_errors) > 0 else 0.0,
        trajectory_length=traj_length,
        num_poses=min_len
    )
    
    return trans_errors, rot_errors, metrics


def compute_nees(
    estimated_states: List[EstimatorState],
    ground_truth: Trajectory,
    state_dim: int = 6
) -> Tuple[np.ndarray, ConsistencyMetrics]:
    """
    Compute Normalized Estimation Error Squared (NEES).
    
    NEES tests if the estimator's uncertainty estimates are consistent.
    
    Args:
        estimated_states: List of estimator states with covariances
        ground_truth: Ground truth trajectory
        state_dim: Dimension of state space (6 for SE3)
    
    Returns:
        NEES values per timestep and consistency metrics
    """
    nees_values = []
    
    min_len = min(len(estimated_states), len(ground_truth.states))
    
    for i in range(min_len):
        est_state = estimated_states[i]
        gt_state = ground_truth.states[i]
        
        # Check if covariance is available
        if est_state.robot_covariance is None:
            continue
        
        # Compute error (position only for simplicity)
        error = est_state.robot_pose.position - gt_state.pose.position
        
        # Get position covariance (upper-left 3x3 block)
        P = est_state.robot_covariance[:3, :3]
        
        # Compute NEES
        try:
            nees = error.T @ np.linalg.inv(P) @ error
            nees_values.append(nees)
        except np.linalg.LinAlgError:
            # Singular covariance matrix
            continue
    
    nees_values = np.array(nees_values)
    
    if len(nees_values) == 0:
        return nees_values, ConsistencyMetrics()
    
    # Chi-squared test (3 DOF for position)
    chi2_lower = 0.352  # Chi2(0.025, 3)
    chi2_upper = 9.348  # Chi2(0.975, 3)
    
    # Compute percentage within bounds
    within_bounds = np.sum((nees_values >= chi2_lower) & (nees_values <= chi2_upper))
    percentage = within_bounds / len(nees_values) * 100
    
    # Build metrics
    metrics = ConsistencyMetrics(
        nees_mean=np.mean(nees_values),
        nees_median=np.median(nees_values),
        nees_std=np.std(nees_values),
        nees_chi2_percentage=percentage,
        is_consistent=percentage >= 95.0  # 95% should be within bounds
    )
    
    return nees_values, metrics


def compute_relative_pose(pose1: Pose, pose2: Pose) -> np.ndarray:
    """
    Compute relative transformation from pose1 to pose2.
    
    Args:
        pose1: First pose
        pose2: Second pose
    
    Returns:
        4x4 transformation matrix
    """
    T1 = pose1.to_matrix()
    T2 = pose2.to_matrix()
    return np.linalg.inv(T1) @ T2


def rotation_matrix_to_angle(R: np.ndarray) -> float:
    """
    Extract rotation angle from rotation matrix.
    
    Args:
        R: 3x3 rotation matrix
    
    Returns:
        Rotation angle in radians
    """
    # Use trace formula: trace(R) = 1 + 2*cos(theta)
    trace = np.trace(R)
    cos_theta = (trace - 1) / 2
    
    # Clamp to avoid numerical issues
    cos_theta = np.clip(cos_theta, -1, 1)
    
    return np.arccos(cos_theta)


def compute_landmark_errors(
    estimated_landmarks: Dict[int, np.ndarray],
    ground_truth_landmarks: Dict[int, np.ndarray]
) -> Dict[str, float]:
    """
    Compute landmark estimation errors.
    
    Args:
        estimated_landmarks: Estimated landmark positions by ID
        ground_truth_landmarks: Ground truth landmark positions by ID
    
    Returns:
        Dictionary of error metrics
    """
    errors = []
    
    for lid, est_pos in estimated_landmarks.items():
        if lid in ground_truth_landmarks:
            gt_pos = ground_truth_landmarks[lid]
            error = np.linalg.norm(est_pos - gt_pos)
            errors.append(error)
    
    if not errors:
        return {
            "rmse": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "num_matched": 0
        }
    
    errors = np.array(errors)
    
    return {
        "rmse": np.sqrt(np.mean(errors**2)),
        "mean": np.mean(errors),
        "median": np.median(errors),
        "std": np.std(errors),
        "num_matched": len(errors)
    }