"""
Trajectory generators for different motion patterns.
"""

import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from src.common.data_structures import Trajectory, TrajectoryState, Pose
from src.utils.math_utils import (
    quaternion_multiply,
    rotation_matrix_to_quaternion,
    so3_exp
)


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
        if angular_velocity is None:
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
            # Yaw angle is theta + pi/2 (perpendicular to radius)
            yaw = theta + np.pi / 2
            
            # Create rotation matrix (yaw only, no pitch or roll)
            R = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
            quaternion = rotation_matrix_to_quaternion(R)
            
            # Angular velocity (only yaw rate)
            angular_velocity = np.array([0, 0, self.angular_velocity])
            
            # Create pose and state
            pose = Pose(
                timestamp=t,
                position=position,
                quaternion=quaternion
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
        quaternion = rotation_matrix_to_quaternion(R)
        
        # Angular velocity
        angular_velocity = np.array([0, 0, self.angular_velocity])
        
        pose = Pose(
            timestamp=t,
            position=position,
            quaternion=quaternion
        )
        
        return TrajectoryState(
            pose=pose,
            velocity=velocity,
            angular_velocity=angular_velocity
        )


class Figure8Trajectory:
    """Generate figure-8 trajectory."""
    
    def __init__(
        self,
        scale_x: float = 3.0,
        scale_y: float = 2.0,
        height: float = 1.5,
        params: Optional[TrajectoryParams] = None
    ):
        """
        Initialize figure-8 trajectory generator.
        
        Args:
            scale_x: X-axis scale in meters
            scale_y: Y-axis scale in meters  
            height: Constant height in meters
            params: Trajectory parameters
        """
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.height = height
        self.params = params or TrajectoryParams()
        
        # Angular frequency to complete figure-8 in duration
        self.omega = 2 * np.pi / self.params.duration
    
    def generate(self) -> Trajectory:
        """
        Generate the figure-8 trajectory.
        
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
            tau = self.omega * (t - self.params.start_time)
            
            # Lemniscate parametrization
            x = self.scale_x * np.sin(tau)
            y = self.scale_y * np.sin(tau) * np.cos(tau)
            z = self.height
            position = np.array([x, y, z])
            
            # Velocity
            vx = self.scale_x * self.omega * np.cos(tau)
            vy = self.scale_y * self.omega * (np.cos(tau)**2 - np.sin(tau)**2)
            vz = 0.0
            velocity = np.array([vx, vy, vz])
            
            # Orientation: facing velocity direction
            if np.linalg.norm(velocity[:2]) > 1e-6:
                yaw = np.arctan2(vy, vx)
            else:
                yaw = 0.0
            
            R = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
            quaternion = rotation_matrix_to_quaternion(R)
            
            # Angular velocity (yaw rate)
            # Compute numerically as difference in yaw
            if len(trajectory.states) > 0:
                prev_yaw = np.arctan2(
                    trajectory.states[-1].velocity[1],
                    trajectory.states[-1].velocity[0]
                ) if trajectory.states[-1].velocity is not None else 0.0
                
                dyaw = yaw - prev_yaw
                # Handle angle wrap
                if dyaw > np.pi:
                    dyaw -= 2 * np.pi
                elif dyaw < -np.pi:
                    dyaw += 2 * np.pi
                
                angular_velocity = np.array([0, 0, dyaw / dt])
            else:
                angular_velocity = np.array([0, 0, 0])
            
            # Create pose and state
            pose = Pose(
                timestamp=t,
                position=position,
                quaternion=quaternion
            )
            
            state = TrajectoryState(
                pose=pose,
                velocity=velocity,
                angular_velocity=angular_velocity
            )
            
            trajectory.add_state(state)
        
        return trajectory


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
            quaternion = rotation_matrix_to_quaternion(R)
            
            # Angular velocity
            angular_velocity = np.array([0, 0, self.omega])
            
            # Create pose and state
            pose = Pose(
                timestamp=t,
                position=position,
                quaternion=quaternion
            )
            
            state = TrajectoryState(
                pose=pose,
                velocity=velocity,
                angular_velocity=angular_velocity
            )
            
            trajectory.add_state(state)
        
        return trajectory


class LineTrajectory:
    """Generate straight line trajectory with constant velocity."""
    
    def __init__(
        self,
        start_position: np.ndarray = np.array([0, 0, 1]),
        end_position: np.ndarray = np.array([10, 0, 1]),
        params: Optional[TrajectoryParams] = None
    ):
        """
        Initialize line trajectory generator.
        
        Args:
            start_position: Starting position [x, y, z]
            end_position: Ending position [x, y, z]
            params: Trajectory parameters
        """
        self.start_position = np.asarray(start_position)
        self.end_position = np.asarray(end_position)
        self.params = params or TrajectoryParams()
    
    def generate(self) -> Trajectory:
        """
        Generate the line trajectory.
        
        Returns:
            Trajectory with constant velocity
        """
        trajectory = Trajectory(frame_id="world")
        
        # Compute constant velocity
        displacement = self.end_position - self.start_position
        velocity = displacement / self.params.duration
        
        # Orientation from velocity direction
        if np.linalg.norm(velocity[:2]) > 1e-6:
            yaw = np.arctan2(velocity[1], velocity[0])
        else:
            yaw = 0.0
        
        pitch = np.arctan2(velocity[2], np.linalg.norm(velocity[:2]))
        
        # Build rotation matrix
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
        quaternion = rotation_matrix_to_quaternion(R)
        
        # Generate timestamps
        dt = 1.0 / self.params.rate
        timestamps = np.arange(
            self.params.start_time,
            self.params.start_time + self.params.duration,
            dt
        )
        
        for t in timestamps:
            # Position along line
            s = (t - self.params.start_time) / self.params.duration
            position = self.start_position + s * displacement
            
            # Create pose and state
            pose = Pose(
                timestamp=t,
                position=position,
                quaternion=quaternion
            )
            
            state = TrajectoryState(
                pose=pose,
                velocity=velocity,
                angular_velocity=np.zeros(3)  # No rotation
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
        trajectory_type: Type of trajectory ("circle", "figure8", "spiral", "line")
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
    elif trajectory_type == "figure8":
        generator = Figure8Trajectory(
            scale_x=params.get("scale_x", 3.0),
            scale_y=params.get("scale_y", 2.0),
            height=params.get("height", 1.5),
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
    elif trajectory_type == "line":
        generator = LineTrajectory(
            start_position=np.array(params.get("start_position", [0, 0, 1])),
            end_position=np.array(params.get("end_position", [10, 0, 1])),
            params=traj_params
        )
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")
    
    return generator.generate()