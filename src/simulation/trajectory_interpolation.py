"""
Trajectory interpolation using splines for smooth motion.
"""

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial.transform import Rotation, Slerp
from typing import List, Optional, Tuple
from dataclasses import dataclass

from src.common.data_structures import Trajectory, TrajectoryState, Pose
from src.utils.math_utils import quaternion_to_rotation_matrix


@dataclass
class SplineTrajectoryConfig:
    """Configuration for spline trajectory interpolation."""
    smoothing_factor: float = 0.0  # 0 = exact interpolation, >0 = smoothing
    boundary_condition: str = "natural"  # "natural", "clamped", "periodic"
    position_spline_order: int = 3  # Cubic splines for position
    velocity_from_spline: bool = True  # Compute velocity from spline derivative


class TrajectoryInterpolator:
    """
    Interpolate trajectory using splines for smooth motion.
    
    Uses cubic splines for position and SLERP for orientation.
    """
    
    def __init__(self, config: Optional[SplineTrajectoryConfig] = None):
        """
        Initialize trajectory interpolator.
        
        Args:
            config: Interpolation configuration
        """
        self.config = config or SplineTrajectoryConfig()
        self.position_spline = None
        self.orientation_slerp = None
        self.time_points = None
    
    def fit(self, trajectory: Trajectory) -> None:
        """
        Fit splines to trajectory waypoints.
        
        Args:
            trajectory: Input trajectory with waypoints
        """
        if len(trajectory.states) < 2:
            raise ValueError("Need at least 2 waypoints for interpolation")
        
        # Extract data
        self.time_points = np.array([state.pose.timestamp for state in trajectory.states])
        positions = np.array([state.pose.position for state in trajectory.states])
        # Convert rotation matrices to quaternions for interpolation
        from src.utils.math_utils import rotation_matrix_to_quaternion
        quaternions = np.array([rotation_matrix_to_quaternion(state.pose.rotation_matrix) 
                               for state in trajectory.states])
        
        # Fit position spline
        if self.config.boundary_condition == "periodic":
            # For periodic trajectories, ensure continuity
            bc_type = "periodic"
        elif self.config.boundary_condition == "clamped":
            # Use velocity information if available
            if trajectory.states[0].velocity is not None and trajectory.states[-1].velocity is not None:
                bc_type = ((1, trajectory.states[0].velocity), 
                          (1, trajectory.states[-1].velocity))
            else:
                bc_type = "clamped"
        else:
            bc_type = "natural"
        
        # Create cubic spline for each position component
        self.position_spline = CubicSpline(
            self.time_points,
            positions,
            bc_type=bc_type,
            extrapolate=False
        )
        
        # Create SLERP interpolator for orientations
        rotations = Rotation.from_quat(quaternions)
        self.orientation_slerp = Slerp(self.time_points, rotations)
    
    def interpolate(
        self,
        timestamps: Optional[np.ndarray] = None,
        num_points: Optional[int] = None,
        rate: Optional[float] = None
    ) -> Trajectory:
        """
        Generate interpolated trajectory.
        
        Args:
            timestamps: Specific timestamps to interpolate at
            num_points: Number of points to generate (if timestamps not provided)
            rate: Sampling rate in Hz (if timestamps not provided)
        
        Returns:
            Interpolated trajectory
        """
        if self.position_spline is None:
            raise RuntimeError("Must call fit() before interpolate()")
        
        # Determine interpolation timestamps
        if timestamps is None:
            if rate is not None:
                dt = 1.0 / rate
                timestamps = np.arange(
                    self.time_points[0],
                    self.time_points[-1],
                    dt
                )
            elif num_points is not None:
                timestamps = np.linspace(
                    self.time_points[0],
                    self.time_points[-1],
                    num_points
                )
            else:
                # Default to 100 Hz
                dt = 0.01
                timestamps = np.arange(
                    self.time_points[0],
                    self.time_points[-1],
                    dt
                )
        
        # Interpolate
        trajectory = Trajectory(frame_id="world")
        
        for t in timestamps:
            # Clamp to valid range
            t_clamped = np.clip(t, self.time_points[0], self.time_points[-1])
            
            # Interpolate position
            position = self.position_spline(t_clamped)
            
            # Interpolate orientation
            rotation = self.orientation_slerp(t_clamped)
            quaternion = rotation.as_quat()
            rotation_matrix = quaternion_to_rotation_matrix(quaternion)
            
            # Compute velocity from spline derivative if requested
            velocity = None
            if self.config.velocity_from_spline:
                velocity = self.position_spline.derivative()(t_clamped)
            
            # Compute angular velocity (numerical differentiation of quaternion)
            angular_velocity = None
            if len(trajectory.states) > 0 and self.config.velocity_from_spline:
                dt = t - trajectory.states[-1].pose.timestamp
                if dt > 0:
                    # Compute relative rotation
                    R_prev = trajectory.states[-1].pose.rotation_matrix
                    R_curr = rotation_matrix
                    R_rel = R_prev.T @ R_curr
                    
                    # Extract angular velocity from relative rotation
                    # Using axis-angle representation
                    angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
                    if angle > 1e-6:
                        axis = np.array([
                            R_rel[2, 1] - R_rel[1, 2],
                            R_rel[0, 2] - R_rel[2, 0],
                            R_rel[1, 0] - R_rel[0, 1]
                        ]) / (2 * np.sin(angle))
                        angular_velocity = axis * angle / dt
                    else:
                        angular_velocity = np.zeros(3)
            
            # Create pose and state
            pose = Pose(
                timestamp=t,
                position=position,
                rotation_matrix=rotation_matrix
            )
            
            state = TrajectoryState(
                pose=pose,
                velocity=velocity,
                angular_velocity=angular_velocity
            )
            
            trajectory.add_state(state)
        
        return trajectory
    
    def interpolate_between_waypoints(
        self,
        waypoints: List[Pose],
        rate: float = 100.0,
        speed: Optional[float] = None
    ) -> Trajectory:
        """
        Create smooth trajectory between waypoints.
        
        Args:
            waypoints: List of poses to interpolate between
            rate: Sampling rate in Hz
            speed: Desired speed (m/s) for constant velocity (if None, uses time-optimal)
        
        Returns:
            Smooth interpolated trajectory
        """
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")
        
        # Create initial trajectory from waypoints
        initial_traj = Trajectory()
        
        if speed is not None:
            # Compute timestamps based on desired speed
            t = 0.0
            for i, waypoint in enumerate(waypoints):
                if i > 0:
                    dist = np.linalg.norm(
                        waypoint.position - waypoints[i-1].position
                    )
                    t += dist / speed
                
                pose = Pose(
                    timestamp=t,
                    position=waypoint.position,
                    quaternion=waypoint.quaternion
                )
                state = TrajectoryState(pose=pose)
                initial_traj.add_state(state)
        else:
            # Use provided timestamps or uniform spacing
            for i, waypoint in enumerate(waypoints):
                if waypoint.timestamp is None:
                    waypoint.timestamp = float(i)
                state = TrajectoryState(pose=waypoint)
                initial_traj.add_state(state)
        
        # Fit and interpolate
        self.fit(initial_traj)
        return self.interpolate(rate=rate)


def smooth_trajectory(
    trajectory: Trajectory,
    window_size: int = 5,
    position_sigma: float = 0.5,
    orientation_sigma: float = 0.1
) -> Trajectory:
    """
    Smooth existing trajectory using Gaussian filtering.
    
    Args:
        trajectory: Input trajectory
        window_size: Size of smoothing window
        position_sigma: Gaussian sigma for position smoothing
        orientation_sigma: Gaussian sigma for orientation smoothing
    
    Returns:
        Smoothed trajectory
    """
    from scipy.ndimage import gaussian_filter1d
    
    # Extract data
    timestamps = np.array([state.pose.timestamp for state in trajectory.states])
    positions = np.array([state.pose.position for state in trajectory.states])
    # Convert rotation matrices to quaternions for smoothing
    from src.utils.math_utils import rotation_matrix_to_quaternion
    quaternions = np.array([rotation_matrix_to_quaternion(state.pose.rotation_matrix) 
                           for state in trajectory.states])
    
    # Smooth positions
    smoothed_positions = np.zeros_like(positions)
    for i in range(3):
        smoothed_positions[:, i] = gaussian_filter1d(
            positions[:, i],
            sigma=position_sigma,
            mode='nearest'
        )
    
    # Smooth orientations (convert to axis-angle, smooth, convert back)
    rotations = Rotation.from_quat(quaternions)
    rotvecs = rotations.as_rotvec()
    
    smoothed_rotvecs = np.zeros_like(rotvecs)
    for i in range(3):
        smoothed_rotvecs[:, i] = gaussian_filter1d(
            rotvecs[:, i],
            sigma=orientation_sigma,
            mode='nearest'
        )
    
    smoothed_rotations = Rotation.from_rotvec(smoothed_rotvecs)
    smoothed_rotation_matrices = smoothed_rotations.as_matrix()
    
    # Rebuild trajectory
    smoothed_traj = Trajectory(frame_id=trajectory.frame_id)
    
    for i, t in enumerate(timestamps):
        # Compute velocities via finite differences
        velocity = None
        if i > 0 and i < len(timestamps) - 1:
            dt_next = timestamps[i+1] - t
            dt_prev = t - timestamps[i-1]
            velocity = (smoothed_positions[i+1] - smoothed_positions[i-1]) / (dt_next + dt_prev)
        
        pose = Pose(
            timestamp=t,
            position=smoothed_positions[i],
            rotation_matrix=smoothed_rotation_matrices[i]
        )
        
        state = TrajectoryState(
            pose=pose,
            velocity=velocity
        )
        
        smoothed_traj.add_state(state)
    
    return smoothed_traj


def create_bezier_trajectory(
    control_points: List[np.ndarray],
    num_points: int = 100,
    duration: float = 10.0
) -> Trajectory:
    """
    Create trajectory using Bezier curves.
    
    Args:
        control_points: List of control points for Bezier curve
        num_points: Number of points to generate
        duration: Total duration of trajectory
    
    Returns:
        Bezier curve trajectory
    """
    from scipy.special import comb
    
    n = len(control_points) - 1  # Degree of Bezier curve
    control_points = np.array(control_points)
    
    # Generate parameter values
    t_params = np.linspace(0, 1, num_points)
    timestamps = np.linspace(0, duration, num_points)
    
    # Compute Bezier curve
    positions = np.zeros((num_points, 3))
    
    for i, t in enumerate(t_params):
        position = np.zeros(3)
        for j, point in enumerate(control_points):
            # Bernstein polynomial
            bernstein = comb(n, j) * (t ** j) * ((1 - t) ** (n - j))
            position += bernstein * point
        positions[i] = position
    
    # Create trajectory
    trajectory = Trajectory(frame_id="world")
    
    for i, (t, pos) in enumerate(zip(timestamps, positions)):
        # Compute velocity from finite differences
        velocity = None
        if i > 0 and i < num_points - 1:
            dt = timestamps[i+1] - timestamps[i-1]
            velocity = (positions[i+1] - positions[i-1]) / dt
        
        # Simple orientation: facing velocity direction
        quaternion = np.array([1, 0, 0, 0])  # Default identity
        R = np.eye(3)
        if velocity is not None and np.linalg.norm(velocity[:2]) > 1e-6:
            yaw = np.arctan2(velocity[1], velocity[0])
            R = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
        
        pose = Pose(
            timestamp=t,
            position=pos,
            rotation_matrix=R
        )
        
        state = TrajectoryState(
            pose=pose,
            velocity=velocity
        )
        
        trajectory.add_state(state)
    
    return trajectory