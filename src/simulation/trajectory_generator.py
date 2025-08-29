"""
Trajectory generators for different motion patterns.
"""

import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from src.common.data_structures import Trajectory, TrajectoryState, Pose


@dataclass
class TrajectoryParams:
    """Base parameters for trajectory generation."""
    duration: float = 20.0  # seconds
    rate: float = 100.0     # Hz (sampling rate)
    start_time: float = 0.0  # Start timestamp


class CircleTrajectory:
    """Generate circular trajectory with constant angular velocity."""
    
    def __init__(
        self,
        radius: float = 2.0,
        height: float = 1.5,
        angular_velocity: Optional[float] = None,
        params: Optional[TrajectoryParams] = None
    ):
        """
        Initialize circle trajectory generator.
        
        Args:
            radius: Circle radius in meters
            height: Constant height above ground in meters
            angular_velocity: Angular velocity in rad/s (computed from duration if None)
            params: Trajectory parameters
        """
        self.radius = radius
        self.height = height
        self.params = params or TrajectoryParams()
        
        # Compute angular velocity to complete one circle in duration
        # Special case: if radius is 0, set angular velocity to 0 (stationary)
        if radius == 0:
            self.angular_velocity = 0.0
        elif angular_velocity is None:
            self.angular_velocity = 2 * np.pi / self.params.duration
        else:
            self.angular_velocity = angular_velocity
    
    def generate(self) -> Trajectory:
        """
        Generate the circular trajectory.
        
        Returns:
            Trajectory with poses, velocities, and angular velocities
        """
        trajectory = Trajectory(frame_id="world")
        
        # Generate timestamps
        dt = 1.0 / self.params.rate
        timestamps = np.arange(
            self.params.start_time,
            self.params.start_time + self.params.duration,
            dt
        )
        
        for t in timestamps:
            # Current angle
            theta = self.angular_velocity * (t - self.params.start_time)
            
            # Position on circle
            x = self.radius * np.cos(theta)
            y = self.radius * np.sin(theta)
            z = self.height
            position = np.array([x, y, z])
            
            # Velocity (tangent to circle)
            vx = -self.radius * self.angular_velocity * np.sin(theta)
            vy = self.radius * self.angular_velocity * np.cos(theta)
            vz = 0.0
            velocity = np.array([vx, vy, vz])
            
            # Orientation: facing tangent direction (forward along velocity)
            # Special case: if radius is 0 (stationary), keep yaw at 0
            if self.radius == 0:
                yaw = 0.0
            else:
                # Yaw angle is theta + pi/2 (perpendicular to radius)
                yaw = theta + np.pi / 2
            
            # Create rotation matrix (yaw only, no pitch or roll)
            R = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
            # Angular velocity (only yaw rate)
            angular_velocity = np.array([0, 0, self.angular_velocity])
            
            # Create pose and state
            pose = Pose(
                timestamp=t,
                position=position,
                rotation_matrix=R
            )
            
            state = TrajectoryState(
                pose=pose,
                velocity=velocity,
                angular_velocity=angular_velocity
            )
            
            trajectory.add_state(state)
        
        return trajectory
    
    def get_analytical_state(self, t: float) -> TrajectoryState:
        """
        Get analytical state at any time t.
        
        Args:
            t: Time in seconds
        
        Returns:
            TrajectoryState at time t
        """
        theta = self.angular_velocity * t
        
        # Position
        position = np.array([
            self.radius * np.cos(theta),
            self.radius * np.sin(theta),
            self.height
        ])
        
        # Velocity
        velocity = np.array([
            -self.radius * self.angular_velocity * np.sin(theta),
            self.radius * self.angular_velocity * np.cos(theta),
            0.0
        ])
        
        # Orientation
        yaw = theta + np.pi / 2
        R = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        # Angular velocity
        angular_velocity = np.array([0, 0, self.angular_velocity])
        
        pose = Pose(
            timestamp=t,
            position=position,
            rotation_matrix=R
        )
        
        return TrajectoryState(
            pose=pose,
            velocity=velocity,
            angular_velocity=angular_velocity
        )




class SpiralTrajectory:
    """Generate spiral trajectory with increasing radius."""
    
    def __init__(
        self,
        initial_radius: float = 0.5,
        final_radius: float = 3.0,
        initial_height: float = 0.5,
        final_height: float = 3.0,
        params: Optional[TrajectoryParams] = None
    ):
        """
        Initialize spiral trajectory generator.
        
        Args:
            initial_radius: Starting radius in meters
            final_radius: Ending radius in meters
            initial_height: Starting height in meters
            final_height: Ending height in meters
            params: Trajectory parameters
        """
        self.initial_radius = initial_radius
        self.final_radius = final_radius
        self.initial_height = initial_height
        self.final_height = final_height
        self.params = params or TrajectoryParams()
        
        # Number of revolutions
        self.n_revolutions = 3.0
        self.omega = 2 * np.pi * self.n_revolutions / self.params.duration
    
    def generate(self) -> Trajectory:
        """
        Generate the spiral trajectory.
        
        Returns:
            Trajectory with poses and velocities
        """
        trajectory = Trajectory(frame_id="world")
        
        # Generate timestamps
        dt = 1.0 / self.params.rate
        timestamps = np.arange(
            self.params.start_time,
            self.params.start_time + self.params.duration,
            dt
        )
        
        for t in timestamps:
            # Normalized time [0, 1]
            s = (t - self.params.start_time) / self.params.duration
            
            # Interpolate radius and height
            radius = self.initial_radius + s * (self.final_radius - self.initial_radius)
            height = self.initial_height + s * (self.final_height - self.initial_height)
            
            # Angle
            theta = self.omega * (t - self.params.start_time)
            
            # Position
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            z = height
            position = np.array([x, y, z])
            
            # Velocity (includes radial expansion and vertical motion)
            dr_dt = (self.final_radius - self.initial_radius) / self.params.duration
            dz_dt = (self.final_height - self.initial_height) / self.params.duration
            
            vx = dr_dt * np.cos(theta) - radius * self.omega * np.sin(theta)
            vy = dr_dt * np.sin(theta) + radius * self.omega * np.cos(theta)
            vz = dz_dt
            velocity = np.array([vx, vy, vz])
            
            # Orientation: facing velocity direction
            if np.linalg.norm(velocity[:2]) > 1e-6:
                yaw = np.arctan2(vy, vx)
            else:
                yaw = 0.0
            
            # Add pitch based on vertical velocity
            pitch = np.arctan2(vz, np.linalg.norm(velocity[:2]))
            
            # Build rotation matrix (yaw, then pitch)
            R_yaw = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
            
            R_pitch = np.array([
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]
            ])
            
            R = R_yaw @ R_pitch
            # Angular velocity
            angular_velocity = np.array([0, 0, self.omega])
            
            # Create pose and state
            pose = Pose(
                timestamp=t,
                position=position,
                rotation_matrix=R
            )
            
            state = TrajectoryState(
                pose=pose,
                velocity=velocity,
                angular_velocity=angular_velocity
            )
            
            trajectory.add_state(state)
        
        return trajectory




def generate_trajectory(
    trajectory_type: str,
    params: Dict[str, Any]
) -> Trajectory:
    """
    Factory function to generate trajectories.
    
    Args:
        trajectory_type: Type of trajectory ("circle" or "spiral")
        params: Parameters for trajectory generation
    
    Returns:
        Generated trajectory
    """
    # Extract common parameters
    traj_params = TrajectoryParams(
        duration=params.get("duration", 20.0),
        rate=params.get("rate", 100.0),
        start_time=params.get("start_time", 0.0)
    )
    
    if trajectory_type == "circle":
        generator = CircleTrajectory(
            radius=params.get("radius", 2.0),
            height=params.get("height", 1.5),
            angular_velocity=params.get("angular_velocity"),
            params=traj_params
        )
    elif trajectory_type == "spiral":
        generator = SpiralTrajectory(
            initial_radius=params.get("initial_radius", 0.5),
            final_radius=params.get("final_radius", 3.0),
            initial_height=params.get("initial_height", 0.5),
            final_height=params.get("final_height", 3.0),
            params=traj_params
        )
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}. Must be 'circle' or 'spiral'")
    
    return generator.generate()