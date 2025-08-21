"""
IMU measurement generation for SLAM simulation.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

from src.common.data_structures import (
    IMUMeasurement, IMUData, IMUCalibration,
    Trajectory, TrajectoryState
)
from src.utils.math_utils import (
    quaternion_to_rotation_matrix,
    so3_log
)


@dataclass
class IMUNoiseConfig:
    """Configuration for IMU noise model."""
    # Accelerometer noise
    accel_noise_density: float = 0.01  # m/s^2/sqrt(Hz)
    accel_random_walk: float = 0.001  # m/s^3/sqrt(Hz)
    accel_bias_initial: np.ndarray = None  # Initial bias
    accel_bias_stability: float = 0.0001  # m/s^2
    
    # Gyroscope noise
    gyro_noise_density: float = 0.001  # rad/s/sqrt(Hz)
    gyro_random_walk: float = 0.0001  # rad/s^2/sqrt(Hz)
    gyro_bias_initial: np.ndarray = None  # Initial bias
    gyro_bias_stability: float = 0.0001  # rad/s
    
    # Gravity
    gravity_magnitude: float = 9.81  # m/s^2
    
    # Random seed
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Initialize default biases if not provided."""
        if self.accel_bias_initial is None:
            self.accel_bias_initial = np.zeros(3)
        if self.gyro_bias_initial is None:
            self.gyro_bias_initial = np.zeros(3)


class IMUModel:
    """IMU measurement model for generating synthetic IMU data."""
    
    def __init__(
        self,
        calibration: Optional[IMUCalibration] = None,
        noise_config: Optional[IMUNoiseConfig] = None
    ):
        """
        Initialize IMU model.
        
        Args:
            calibration: IMU calibration parameters
            noise_config: Noise configuration
        """
        self.calibration = calibration or IMUCalibration(
            imu_id="imu0",
            accelerometer_noise_density=0.01,
            accelerometer_random_walk=0.001,
            gyroscope_noise_density=0.001,
            gyroscope_random_walk=0.0001,
            rate=200.0
        )
        
        self.noise_config = noise_config or IMUNoiseConfig()
        
        # Set random seed
        if self.noise_config.seed is not None:
            np.random.seed(self.noise_config.seed)
        
        # Initialize biases
        self.accel_bias = self.noise_config.accel_bias_initial.copy()
        self.gyro_bias = self.noise_config.gyro_bias_initial.copy()
        
        # Gravity vector in world frame (pointing down in ENU)
        self.gravity_world = np.array([0, 0, -self.noise_config.gravity_magnitude])
        
        # Time step
        self.dt = 1.0 / self.calibration.rate
    
    def generate_perfect_measurements(
        self,
        trajectory: Trajectory
    ) -> IMUData:
        """
        Generate perfect (noiseless) IMU measurements from trajectory.
        
        Args:
            trajectory: Ground truth trajectory with velocities
        
        Returns:
            IMUData with perfect measurements
        """
        imu_data = IMUData(
            sensor_id=self.calibration.imu_id,
            rate=self.calibration.rate
        )
        
        # Generate IMU timestamps at higher rate than trajectory
        traj_start, traj_end = trajectory.get_time_range()
        imu_timestamps = np.arange(traj_start, traj_end, self.dt)
        
        for t in imu_timestamps:
            # Interpolate trajectory state at IMU timestamp
            state = self._interpolate_state(trajectory, t)
            
            if state is None:
                continue
            
            # Compute IMU measurements
            accel, gyro = self._compute_imu_measurements(state)
            
            # Create measurement
            measurement = IMUMeasurement(
                timestamp=t,
                accelerometer=accel,
                gyroscope=gyro
            )
            
            imu_data.add_measurement(measurement)
        
        return imu_data
    
    def generate_noisy_measurements(
        self,
        trajectory: Trajectory
    ) -> IMUData:
        """
        Generate noisy IMU measurements from trajectory.
        
        Args:
            trajectory: Ground truth trajectory
        
        Returns:
            IMUData with noisy measurements
        """
        # First generate perfect measurements
        imu_data = self.generate_perfect_measurements(trajectory)
        
        # Add noise to each measurement
        for measurement in imu_data.measurements:
            # Add white noise
            accel_noise = np.random.normal(
                0, 
                self.noise_config.accel_noise_density * np.sqrt(self.calibration.rate),
                3
            )
            gyro_noise = np.random.normal(
                0,
                self.noise_config.gyro_noise_density * np.sqrt(self.calibration.rate),
                3
            )
            
            # Update biases with random walk
            self.accel_bias += np.random.normal(
                0,
                self.noise_config.accel_random_walk * np.sqrt(self.dt),
                3
            )
            self.gyro_bias += np.random.normal(
                0,
                self.noise_config.gyro_random_walk * np.sqrt(self.dt),
                3
            )
            
            # Limit bias magnitude (bias stability)
            self.accel_bias = np.clip(
                self.accel_bias,
                -3 * self.noise_config.accel_bias_stability,
                3 * self.noise_config.accel_bias_stability
            )
            self.gyro_bias = np.clip(
                self.gyro_bias,
                -3 * self.noise_config.gyro_bias_stability,
                3 * self.noise_config.gyro_bias_stability
            )
            
            # Add noise and bias to measurements
            measurement.accelerometer += accel_noise + self.accel_bias
            measurement.gyroscope += gyro_noise + self.gyro_bias
        
        return imu_data
    
    def _interpolate_state(
        self,
        trajectory: Trajectory,
        timestamp: float
    ) -> Optional[TrajectoryState]:
        """
        Interpolate trajectory state at given timestamp.
        
        Args:
            trajectory: Trajectory to interpolate
            timestamp: Desired timestamp
        
        Returns:
            Interpolated state or None if out of bounds
        """
        # Find surrounding states
        states = trajectory.states
        
        if timestamp < states[0].pose.timestamp or timestamp > states[-1].pose.timestamp:
            return None
        
        # Find bracketing states
        for i in range(len(states) - 1):
            if states[i].pose.timestamp <= timestamp <= states[i+1].pose.timestamp:
                # Linear interpolation for now
                t0 = states[i].pose.timestamp
                t1 = states[i+1].pose.timestamp
                alpha = (timestamp - t0) / (t1 - t0) if t1 > t0 else 0.0
                
                # Interpolate pose
                from src.utils.math_utils import so3_interpolate
                
                pos = (1 - alpha) * states[i].pose.position + alpha * states[i+1].pose.position
                R = so3_interpolate(states[i].pose.rotation_matrix, states[i+1].pose.rotation_matrix, alpha)
                
                # Interpolate velocities if available
                vel = None
                ang_vel = None
                
                if states[i].velocity is not None and states[i+1].velocity is not None:
                    vel = (1 - alpha) * states[i].velocity + alpha * states[i+1].velocity
                
                if states[i].angular_velocity is not None and states[i+1].angular_velocity is not None:
                    ang_vel = (1 - alpha) * states[i].angular_velocity + alpha * states[i+1].angular_velocity
                
                from src.common.data_structures import Pose
                pose = Pose(timestamp=timestamp, position=pos, rotation_matrix=R)
                
                return TrajectoryState(
                    pose=pose,
                    velocity=vel,
                    angular_velocity=ang_vel
                )
        
        return None
    
    def _compute_imu_measurements(
        self,
        state: TrajectoryState
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute IMU measurements from trajectory state.
        
        Args:
            state: Current trajectory state
        
        Returns:
            (accelerometer, gyroscope) measurements in body frame
        """
        # Get rotation from world to body
        R_WB = state.pose.rotation_matrix
        R_BW = R_WB.T
        
        # Angular velocity is already in body frame
        gyroscope = state.angular_velocity if state.angular_velocity is not None else np.zeros(3)
        
        # Compute acceleration
        if state.velocity is not None:
            # For now, use finite differences for acceleration
            # In a more sophisticated model, we'd compute this from the trajectory
            # or store acceleration in the state
            
            # Specific force: a_specific = a_total - g
            # In world frame: a_world = dv/dt (from trajectory)
            # We need to add gravity and transform to body frame
            
            # For simplicity, assume zero acceleration (constant velocity)
            # This is reasonable for smooth trajectories
            accel_world = np.zeros(3)  # Could be computed from trajectory
            
            # Specific force in world frame (remove gravity)
            specific_force_world = accel_world - self.gravity_world
            
            # Transform to body frame
            accelerometer = R_BW @ specific_force_world
        else:
            # If no velocity, just measure gravity
            accelerometer = R_BW @ (-self.gravity_world)
        
        return accelerometer, gyroscope


class IMUPreintegrator:
    """
    IMU preintegration for use in optimization.
    Integrates IMU measurements between two keyframes.
    """
    
    def __init__(self, gravity: np.ndarray = np.array([0, 0, -9.81])):
        """
        Initialize preintegrator.
        
        Args:
            gravity: Gravity vector in world frame
        """
        self.gravity = gravity
        self.reset()
    
    def reset(self):
        """Reset preintegration."""
        self.delta_p = np.zeros(3)  # Position increment
        self.delta_v = np.zeros(3)  # Velocity increment
        self.delta_R = np.eye(3)    # Rotation increment
        self.delta_t = 0.0          # Time duration
        self.measurements = []       # Store measurements for reintegration
    
    def integrate(
        self,
        measurement: IMUMeasurement,
        dt: float
    ):
        """
        Integrate single IMU measurement.
        
        Args:
            measurement: IMU measurement
            dt: Time step
        """
        # Store measurement
        self.measurements.append((measurement, dt))
        
        # Extract measurements
        acc = measurement.accelerometer
        gyro = measurement.gyroscope
        
        # Update rotation (first-order integration)
        from src.utils.math_utils import so3_exp
        dR = so3_exp(gyro * dt)
        
        # Update position and velocity
        self.delta_p += self.delta_v * dt + 0.5 * self.delta_R @ acc * dt * dt
        self.delta_v += self.delta_R @ acc * dt
        self.delta_R = self.delta_R @ dR
        self.delta_t += dt
    
    def predict(
        self,
        state_i: TrajectoryState,
        gravity: Optional[np.ndarray] = None
    ) -> TrajectoryState:
        """
        Predict state at time j given state at time i.
        
        Args:
            state_i: State at time i
            gravity: Gravity vector (uses stored if not provided)
        
        Returns:
            Predicted state at time j
        """
        if gravity is None:
            gravity = self.gravity
        
        R_i = quaternion_to_rotation_matrix(state_i.pose.quaternion)
        p_i = state_i.pose.position
        v_i = state_i.velocity if state_i.velocity is not None else np.zeros(3)
        
        # Predict state at j
        R_j = R_i @ self.delta_R
        v_j = v_i + gravity * self.delta_t + R_i @ self.delta_v
        p_j = p_i + v_i * self.delta_t + 0.5 * gravity * self.delta_t**2 + R_i @ self.delta_p
        
        from src.common.data_structures import Pose
        pose_j = Pose(
            timestamp=state_i.pose.timestamp + self.delta_t,
            position=p_j,
            rotation_matrix=R_j
        )
        
        return TrajectoryState(
            pose=pose_j,
            velocity=v_j,
            angular_velocity=None  # Would need to track this
        )