"""
IMU integration and preintegration for SLAM estimation.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from src.common.data_structures import IMUMeasurement, Pose
from src.utils.math_utils import (
    skew, exp_so3, log_so3,
    quaternion_to_rotation_matrix, rotation_matrix_to_quaternion,
    quaternion_multiply
)


class IntegrationMethod(Enum):
    """Integration methods for IMU measurements."""
    EULER = "euler"          # First-order Euler integration
    RK4 = "rk4"             # Fourth-order Runge-Kutta
    MIDPOINT = "midpoint"   # Midpoint/Trapezoidal rule


@dataclass
class IMUState:
    """
    IMU integration state.
    
    Attributes:
        position: Position in world frame [x, y, z]
        velocity: Velocity in world frame [vx, vy, vz]
        quaternion: Orientation quaternion [qw, qx, qy, qz]
        accel_bias: Accelerometer bias [bax, bay, baz]
        gyro_bias: Gyroscope bias [bgx, bgy, bgz]
        timestamp: Current timestamp
    """
    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    accel_bias: np.ndarray
    gyro_bias: np.ndarray
    timestamp: float
    
    def copy(self) -> 'IMUState':
        """Create a deep copy of the state."""
        return IMUState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            quaternion=self.quaternion.copy(),
            accel_bias=self.accel_bias.copy(),
            gyro_bias=self.gyro_bias.copy(),
            timestamp=self.timestamp
        )


@dataclass
class PreintegrationResult:
    """
    Result of IMU preintegration between two keyframes.
    
    Attributes:
        delta_position: Relative position change
        delta_velocity: Relative velocity change
        delta_rotation: Relative rotation (as quaternion)
        covariance: Uncertainty of preintegrated measurements
        jacobian: Jacobian with respect to bias
        dt: Total time interval
        num_measurements: Number of integrated measurements
    """
    delta_position: np.ndarray
    delta_velocity: np.ndarray
    delta_rotation: np.ndarray
    covariance: np.ndarray
    jacobian: np.ndarray
    dt: float
    num_measurements: int


class IMUIntegrator:
    """
    IMU integration for state propagation.
    
    Integrates IMU measurements to predict state changes.
    """
    
    def __init__(
        self,
        gravity: np.ndarray = np.array([0, 0, -9.81]),
        method: IntegrationMethod = IntegrationMethod.EULER
    ):
        """
        Initialize IMU integrator.
        
        Args:
            gravity: Gravity vector in world frame (default: [0, 0, -9.81])
            method: Integration method to use
        """
        self.gravity = gravity
        self.method = method
    
    def integrate(
        self,
        state: IMUState,
        measurement: IMUMeasurement,
        dt: float
    ) -> IMUState:
        """
        Integrate single IMU measurement.
        
        Args:
            state: Current state
            measurement: IMU measurement
            dt: Time step
        
        Returns:
            Updated state
        """
        if self.method == IntegrationMethod.EULER:
            return self._integrate_euler(state, measurement, dt)
        elif self.method == IntegrationMethod.RK4:
            return self._integrate_rk4(state, measurement, dt)
        elif self.method == IntegrationMethod.MIDPOINT:
            return self._integrate_midpoint(state, measurement, dt)
        else:
            raise ValueError(f"Unknown integration method: {self.method}")
    
    def _integrate_euler(
        self,
        state: IMUState,
        measurement: IMUMeasurement,
        dt: float
    ) -> IMUState:
        """
        First-order Euler integration.
        
        Args:
            state: Current state
            measurement: IMU measurement
            dt: Time step
        
        Returns:
            Updated state
        """
        # Remove bias from measurements
        accel = measurement.accelerometer - state.accel_bias
        gyro = measurement.gyroscope - state.gyro_bias
        
        # Get rotation matrix
        R = quaternion_to_rotation_matrix(state.quaternion)
        
        # Update orientation (integrate angular velocity)
        omega_dt = gyro * dt
        delta_R = exp_so3(omega_dt)
        R_new = R @ delta_R
        q_new = rotation_matrix_to_quaternion(R_new)
        
        # Update velocity (integrate acceleration in world frame)
        accel_world = R @ accel + self.gravity
        v_new = state.velocity + accel_world * dt
        
        # Update position (integrate velocity)
        p_new = state.position + state.velocity * dt + 0.5 * accel_world * dt**2
        
        # Create new state
        new_state = IMUState(
            position=p_new,
            velocity=v_new,
            quaternion=q_new,
            accel_bias=state.accel_bias.copy(),
            gyro_bias=state.gyro_bias.copy(),
            timestamp=state.timestamp + dt
        )
        
        return new_state
    
    def _integrate_rk4(
        self,
        state: IMUState,
        measurement: IMUMeasurement,
        dt: float
    ) -> IMUState:
        """
        Fourth-order Runge-Kutta integration.
        
        Args:
            state: Current state
            measurement: IMU measurement
            dt: Time step
        
        Returns:
            Updated state
        """
        # Remove bias
        accel = measurement.accelerometer - state.accel_bias
        gyro = measurement.gyroscope - state.gyro_bias
        
        # RK4 integration
        # k1
        k1_vel, k1_pos, k1_rot = self._compute_derivatives(
            state.quaternion, state.velocity, accel, gyro
        )
        
        # k2 (at midpoint)
        q_mid1 = self._update_quaternion(state.quaternion, k1_rot * dt/2)
        v_mid1 = state.velocity + k1_vel * dt/2
        k2_vel, k2_pos, k2_rot = self._compute_derivatives(
            q_mid1, v_mid1, accel, gyro
        )
        
        # k3 (at midpoint with k2)
        q_mid2 = self._update_quaternion(state.quaternion, k2_rot * dt/2)
        v_mid2 = state.velocity + k2_vel * dt/2
        k3_vel, k3_pos, k3_rot = self._compute_derivatives(
            q_mid2, v_mid2, accel, gyro
        )
        
        # k4 (at endpoint with k3)
        q_end = self._update_quaternion(state.quaternion, k3_rot * dt)
        v_end = state.velocity + k3_vel * dt
        k4_vel, k4_pos, k4_rot = self._compute_derivatives(
            q_end, v_end, accel, gyro
        )
        
        # Combine (weighted average)
        delta_v = (k1_vel + 2*k2_vel + 2*k3_vel + k4_vel) * dt / 6
        delta_p = (k1_pos + 2*k2_pos + 2*k3_pos + k4_pos) * dt / 6
        delta_rot = (k1_rot + 2*k2_rot + 2*k3_rot + k4_rot) * dt / 6
        
        # Update state
        new_state = IMUState(
            position=state.position + delta_p,
            velocity=state.velocity + delta_v,
            quaternion=self._update_quaternion(state.quaternion, delta_rot),
            accel_bias=state.accel_bias.copy(),
            gyro_bias=state.gyro_bias.copy(),
            timestamp=state.timestamp + dt
        )
        
        return new_state
    
    def _integrate_midpoint(
        self,
        state: IMUState,
        measurement: IMUMeasurement,
        dt: float
    ) -> IMUState:
        """
        Midpoint/Trapezoidal integration.
        
        Args:
            state: Current state
            measurement: IMU measurement
            dt: Time step
        
        Returns:
            Updated state
        """
        # Remove bias
        accel = measurement.accelerometer - state.accel_bias
        gyro = measurement.gyroscope - state.gyro_bias
        
        # Get rotation matrix
        R = quaternion_to_rotation_matrix(state.quaternion)
        
        # Midpoint rotation
        omega_dt_half = gyro * dt / 2
        R_mid = R @ exp_so3(omega_dt_half)
        
        # Acceleration at midpoint
        accel_world_mid = R_mid @ accel + self.gravity
        
        # Update velocity using midpoint acceleration
        v_new = state.velocity + accel_world_mid * dt
        
        # Update position using average velocity
        v_avg = (state.velocity + v_new) / 2
        p_new = state.position + v_avg * dt
        
        # Full rotation update
        omega_dt = gyro * dt
        R_new = R @ exp_so3(omega_dt)
        q_new = rotation_matrix_to_quaternion(R_new)
        
        # Create new state
        new_state = IMUState(
            position=p_new,
            velocity=v_new,
            quaternion=q_new,
            accel_bias=state.accel_bias.copy(),
            gyro_bias=state.gyro_bias.copy(),
            timestamp=state.timestamp + dt
        )
        
        return new_state
    
    def _compute_derivatives(
        self,
        quaternion: np.ndarray,
        velocity: np.ndarray,
        accel: np.ndarray,
        gyro: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute state derivatives for RK4.
        
        Args:
            quaternion: Current orientation
            velocity: Current velocity
            accel: Bias-corrected acceleration
            gyro: Bias-corrected gyroscope
        
        Returns:
            (velocity_dot, position_dot, rotation_rate)
        """
        R = quaternion_to_rotation_matrix(quaternion)
        
        # Acceleration in world frame
        accel_world = R @ accel + self.gravity
        
        # Derivatives
        velocity_dot = accel_world
        position_dot = velocity
        rotation_rate = gyro
        
        return velocity_dot, position_dot, rotation_rate
    
    def _update_quaternion(
        self,
        quaternion: np.ndarray,
        omega_dt: np.ndarray
    ) -> np.ndarray:
        """
        Update quaternion with angular velocity.
        
        Args:
            quaternion: Current quaternion
            omega_dt: Angular velocity * dt
        
        Returns:
            Updated quaternion
        """
        # Convert to rotation matrix, apply rotation, convert back
        R = quaternion_to_rotation_matrix(quaternion)
        delta_R = exp_so3(omega_dt)
        R_new = R @ delta_R
        return rotation_matrix_to_quaternion(R_new)
    
    def integrate_batch(
        self,
        initial_state: IMUState,
        measurements: List[IMUMeasurement]
    ) -> List[IMUState]:
        """
        Integrate batch of IMU measurements.
        
        Args:
            initial_state: Initial state
            measurements: List of IMU measurements
        
        Returns:
            List of states at each measurement time
        """
        states = [initial_state]
        current_state = initial_state.copy()
        
        for i, meas in enumerate(measurements):
            # Compute time step
            if i == 0:
                dt = meas.timestamp - initial_state.timestamp
            else:
                dt = meas.timestamp - measurements[i-1].timestamp
            
            if dt <= 0:
                continue
            
            # Integrate
            current_state = self.integrate(current_state, meas, dt)
            states.append(current_state)
        
        return states


class IMUPreintegrator:
    """
    IMU preintegration for factor graph optimization.
    
    Preintegrates IMU measurements between keyframes to create
    relative constraints that don't depend on absolute states.
    """
    
    def __init__(
        self,
        accel_noise_density: float = 0.01,
        gyro_noise_density: float = 0.001,
        accel_random_walk: float = 0.001,
        gyro_random_walk: float = 0.0001,
        gravity: np.ndarray = np.array([0, 0, -9.81])
    ):
        """
        Initialize IMU preintegrator.
        
        Args:
            accel_noise_density: Accelerometer noise density
            gyro_noise_density: Gyroscope noise density
            accel_random_walk: Accelerometer bias random walk
            gyro_random_walk: Gyroscope bias random walk
            gravity: Gravity vector in world frame
        """
        self.accel_noise_density = accel_noise_density
        self.gyro_noise_density = gyro_noise_density
        self.accel_random_walk = accel_random_walk
        self.gyro_random_walk = gyro_random_walk
        self.gravity = gravity
        
        # Reset preintegration
        self.reset()
    
    def reset(self, bias_accel: Optional[np.ndarray] = None, bias_gyro: Optional[np.ndarray] = None):
        """
        Reset preintegration.
        
        Args:
            bias_accel: Initial accelerometer bias
            bias_gyro: Initial gyroscope bias
        """
        # Preintegrated values
        self.delta_R = np.eye(3)  # Rotation
        self.delta_v = np.zeros(3)  # Velocity
        self.delta_p = np.zeros(3)  # Position
        
        # Jacobians with respect to bias
        self.J_R_bg = np.zeros((3, 3))
        self.J_v_ba = np.zeros((3, 3))
        self.J_v_bg = np.zeros((3, 3))
        self.J_p_ba = np.zeros((3, 3))
        self.J_p_bg = np.zeros((3, 3))
        
        # Covariance
        self.covariance = np.zeros((9, 9))  # [delta_R, delta_v, delta_p]
        
        # Noise matrices
        self.noise_accel = np.eye(3) * self.accel_noise_density**2
        self.noise_gyro = np.eye(3) * self.gyro_noise_density**2
        
        # Bias
        self.bias_accel = bias_accel if bias_accel is not None else np.zeros(3)
        self.bias_gyro = bias_gyro if bias_gyro is not None else np.zeros(3)
        
        # Accumulated time
        self.dt = 0.0
        self.measurements = []
    
    def add_measurement(self, measurement: IMUMeasurement, dt: float):
        """
        Add IMU measurement to preintegration.
        
        Args:
            measurement: IMU measurement
            dt: Time step since last measurement
        """
        # Store measurement
        self.measurements.append(measurement)
        
        # Remove bias
        accel = measurement.accelerometer - self.bias_accel
        gyro = measurement.gyroscope - self.bias_gyro
        
        # Previous values for Jacobian computation
        delta_R_prev = self.delta_R.copy()
        delta_v_prev = self.delta_v.copy()
        
        # Update rotation
        omega_dt = gyro * dt
        delta_R_dt = exp_so3(omega_dt)
        self.delta_R = self.delta_R @ delta_R_dt
        
        # Update velocity
        self.delta_v = self.delta_v + delta_R_prev @ accel * dt
        
        # Update position
        self.delta_p = self.delta_p + self.delta_v * dt + 0.5 * delta_R_prev @ accel * dt**2
        
        # Update Jacobians (simplified first-order approximation)
        A_dt = exp_so3(-omega_dt)
        
        # Rotation Jacobian w.r.t. gyro bias
        self.J_R_bg = A_dt @ self.J_R_bg - dt * np.eye(3)
        
        # Velocity Jacobians
        self.J_v_ba = self.J_v_ba + delta_R_prev * dt
        self.J_v_bg = self.J_v_bg + delta_R_prev @ skew(accel) @ self.J_R_bg * dt
        
        # Position Jacobians
        self.J_p_ba = self.J_p_ba + self.J_v_ba * dt + 0.5 * delta_R_prev * dt**2
        self.J_p_bg = self.J_p_bg + self.J_v_bg * dt + 0.5 * delta_R_prev @ skew(accel) @ self.J_R_bg * dt**2
        
        # Update covariance (simplified)
        F = np.eye(9)  # State transition matrix
        Q = np.zeros((9, 9))  # Process noise
        
        # Add noise contributions
        Q[0:3, 0:3] = self.noise_gyro * dt**2
        Q[3:6, 3:6] = self.noise_accel * dt**2
        Q[6:9, 6:9] = self.noise_accel * dt**4 / 4
        
        self.covariance = F @ self.covariance @ F.T + Q
        
        # Update total time
        self.dt += dt
    
    def get_result(self) -> PreintegrationResult:
        """
        Get preintegration result.
        
        Returns:
            PreintegrationResult with accumulated values
        """
        # Build Jacobian matrix
        jacobian = np.zeros((9, 6))
        jacobian[0:3, 3:6] = self.J_R_bg
        jacobian[3:6, 0:3] = self.J_v_ba
        jacobian[3:6, 3:6] = self.J_v_bg
        jacobian[6:9, 0:3] = self.J_p_ba
        jacobian[6:9, 3:6] = self.J_p_bg
        
        return PreintegrationResult(
            delta_position=self.delta_p.copy(),
            delta_velocity=self.delta_v.copy(),
            delta_rotation=rotation_matrix_to_quaternion(self.delta_R),
            covariance=self.covariance.copy(),
            jacobian=jacobian,
            dt=self.dt,
            num_measurements=len(self.measurements)
        )
    
    def predict(
        self,
        state_i: IMUState,
        bias_update: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> IMUState:
        """
        Predict state at time j given state at time i.
        
        Args:
            state_i: State at time i
            bias_update: Optional bias correction (delta_ba, delta_bg)
        
        Returns:
            Predicted state at time j
        """
        # Get preintegrated values
        result = self.get_result()
        
        # Apply bias correction if provided
        if bias_update is not None:
            delta_ba, delta_bg = bias_update
            # First-order correction
            correction_v = self.J_v_ba @ delta_ba + self.J_v_bg @ delta_bg
            correction_p = self.J_p_ba @ delta_ba + self.J_p_bg @ delta_bg
            
            delta_v_corrected = result.delta_velocity + correction_v
            delta_p_corrected = result.delta_position + correction_p
        else:
            delta_v_corrected = result.delta_velocity
            delta_p_corrected = result.delta_position
        
        # Get rotation matrices
        R_i = quaternion_to_rotation_matrix(state_i.quaternion)
        R_j = R_i @ self.delta_R
        
        # Predict state at time j
        p_j = state_i.position + state_i.velocity * self.dt + \
              0.5 * self.gravity * self.dt**2 + R_i @ delta_p_corrected
        
        v_j = state_i.velocity + self.gravity * self.dt + R_i @ delta_v_corrected
        
        q_j = rotation_matrix_to_quaternion(R_j)
        
        # Create predicted state
        state_j = IMUState(
            position=p_j,
            velocity=v_j,
            quaternion=q_j,
            accel_bias=state_i.accel_bias.copy(),
            gyro_bias=state_i.gyro_bias.copy(),
            timestamp=state_i.timestamp + self.dt
        )
        
        return state_j


def compute_imu_jacobian(
    state: IMUState,
    measurement: IMUMeasurement,
    dt: float,
    gravity: np.ndarray = np.array([0, 0, -9.81])
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Jacobians for IMU integration.
    
    Args:
        state: Current state
        measurement: IMU measurement
        dt: Time step
        gravity: Gravity vector
    
    Returns:
        F: State transition Jacobian (15x15 for [p, R, v, ba, bg])
        G: Noise Jacobian (15x12 for [na, ng, nba, nbg])
    """
    # State dimension: 15 (position, rotation, velocity, biases)
    F = np.eye(15)
    
    # Extract values
    R = quaternion_to_rotation_matrix(state.quaternion)
    accel = measurement.accelerometer - state.accel_bias
    gyro = measurement.gyroscope - state.gyro_bias
    
    # Position derivatives
    F[0:3, 6:9] = np.eye(3) * dt  # dp/dv
    F[0:3, 3:6] = -R @ skew(accel) * dt**2 / 2  # dp/dR
    F[0:3, 9:12] = -R * dt**2 / 2  # dp/dba
    
    # Rotation derivatives
    F[3:6, 3:6] = exp_so3(-gyro * dt)  # dR/dR
    F[3:6, 12:15] = -np.eye(3) * dt  # dR/dbg
    
    # Velocity derivatives
    F[6:9, 3:6] = -R @ skew(accel) * dt  # dv/dR
    F[6:9, 9:12] = -R * dt  # dv/dba
    
    # Noise Jacobian
    G = np.zeros((15, 12))
    G[0:3, 0:3] = -R * dt**2 / 2  # Position noise from accel
    G[3:6, 3:6] = -np.eye(3) * dt  # Rotation noise from gyro
    G[6:9, 0:3] = -R * dt  # Velocity noise from accel
    G[9:12, 6:9] = np.eye(3)  # Accel bias noise
    G[12:15, 9:12] = np.eye(3)  # Gyro bias noise
    
    return F, G