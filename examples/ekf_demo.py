#!/usr/bin/env python3
"""
Simple demo of EKF-SLAM on a circular trajectory.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.estimation.ekf_slam import EKFSlam
from src.common.config import EKFConfig
from src.common.data_structures import (
    Pose, IMUMeasurement, CameraFrame, CameraObservation,
    ImagePoint, Map, Landmark, Trajectory, TrajectoryState,
    CameraCalibration, CameraIntrinsics, CameraExtrinsics,
    CameraModel
)
from src.evaluation.metrics import compute_ate


def create_simple_trajectory(duration=5.0, dt=0.1):
    """Create a simple circular trajectory."""
    trajectory = Trajectory()
    
    radius = 2.0
    omega = 2 * np.pi / duration  # One revolution
    
    t = 0.0
    while t <= duration:
        x = radius * np.cos(omega * t)
        y = radius * np.sin(omega * t)
        z = 0.0
        
        # Simple orientation (facing tangent direction)
        yaw = omega * t + np.pi/2
        qw = np.cos(yaw/2)
        qz = np.sin(yaw/2)
        
        pose = Pose(
            timestamp=t,
            position=np.array([x, y, z]),
            quaternion=np.array([qw, 0, 0, qz])
        )
        
        # Velocity (tangent to circle)
        vx = -radius * omega * np.sin(omega * t)
        vy = radius * omega * np.cos(omega * t)
        
        state = TrajectoryState(
            pose=pose,
            velocity=np.array([vx, vy, 0])
        )
        trajectory.add_state(state)
        
        t += dt
    
    return trajectory


def main():
    """Run EKF-SLAM demo."""
    print("EKF-SLAM Demo")
    print("=" * 40)
    
    # Create camera calibration
    intrinsics = CameraIntrinsics(
        model=CameraModel.PINHOLE,
        width=640,
        height=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        distortion=np.zeros(4)
    )
    
    extrinsics = CameraExtrinsics(B_T_C=np.eye(4))
    camera_calib = CameraCalibration(
        camera_id="cam0",
        intrinsics=intrinsics,
        extrinsics=extrinsics
    )
    
    # Create ground truth trajectory
    print("Generating ground truth trajectory...")
    gt_trajectory = create_simple_trajectory(duration=5.0, dt=0.1)
    print(f"  Generated {len(gt_trajectory.states)} states")
    
    # Create landmarks
    print("Creating landmarks...")
    gt_map = Map()
    landmarks_positions = [
        [0, 0, 1],     # Center, above
        [3, 0, 0],     # Right
        [0, 3, 0],     # Front
        [-3, 0, 0],    # Left
        [0, -3, 0],    # Back
    ]
    
    for i, pos in enumerate(landmarks_positions):
        landmark = Landmark(id=i, position=np.array(pos))
        gt_map.add_landmark(landmark)
    print(f"  Created {len(gt_map.landmarks)} landmarks")
    
    # Initialize EKF
    print("\nInitializing EKF...")
    config = EKFConfig(
        initial_position_std=0.1,
        initial_velocity_std=0.1,
        pixel_noise_std=2.0,
        chi2_threshold=50.0  # More permissive for demo
    )
    ekf = EKFSlam(config, camera_calib)
    
    # Initialize with first state
    initial_state = gt_trajectory.states[0]
    ekf.initialize(initial_state.pose)
    
    # Create estimated trajectory
    estimated_trajectory = Trajectory()
    estimated_trajectory.add_state(initial_state)
    
    # Process measurements
    print("Processing measurements...")
    for i, state in enumerate(gt_trajectory.states[1:], 1):
        # Generate IMU measurement (with noise)
        imu_meas = IMUMeasurement(
            timestamp=state.pose.timestamp,
            accelerometer=np.random.normal(0, 0.01, 3),
            gyroscope=np.random.normal(0, 0.001, 3)
        )
        
        # Predict
        ekf.predict([imu_meas], state.pose.timestamp - gt_trajectory.states[i-1].pose.timestamp)
        
        # Camera update every 5 steps
        if i % 5 == 0:
            # Generate observations
            observations = []
            for landmark in gt_map.landmarks.values():
                # Simple visibility check
                diff = landmark.position - state.pose.position
                distance = np.linalg.norm(diff)
                
                if distance < 5.0:  # Within range
                    # Project to image (simplified)
                    pixel_u = 320 + 100 * diff[0] / distance + np.random.normal(0, 1)
                    pixel_v = 240 + 100 * diff[1] / distance + np.random.normal(0, 1)
                    
                    if 0 <= pixel_u < 640 and 0 <= pixel_v < 480:
                        obs = CameraObservation(
                            landmark_id=landmark.id,
                            pixel=ImagePoint(u=pixel_u, v=pixel_v)
                        )
                        observations.append(obs)
            
            if observations:
                frame = CameraFrame(
                    timestamp=state.pose.timestamp,
                    camera_id="cam0",
                    observations=observations
                )
                ekf.update(frame, gt_map)
        
        # Save estimated state
        ekf_state = ekf.get_state()
        estimated_trajectory.add_state(TrajectoryState(
            pose=ekf_state.robot_pose,
            velocity=ekf_state.robot_velocity
        ))
        
        if i % 10 == 0:
            print(f"  Processed {i}/{len(gt_trajectory.states)-1} states")
    
    # Compute error metrics
    print("\nComputing error metrics...")
    errors, metrics = compute_ate(estimated_trajectory, gt_trajectory, align=False)
    
    print(f"\nResults:")
    print(f"  ATE RMSE: {metrics.ate_rmse:.3f} m")
    print(f"  ATE Mean: {metrics.ate_mean:.3f} m")
    print(f"  ATE Median: {metrics.ate_median:.3f} m")
    print(f"  ATE Std: {metrics.ate_std:.3f} m")
    print(f"  ATE Max: {metrics.ate_max:.3f} m")
    
    # Get final result
    result = ekf.get_result()
    print(f"\nEKF Statistics:")
    print(f"  Updates performed: {result.metadata['num_updates']}")
    print(f"  Outliers rejected: {result.metadata['num_outliers']}")
    
    print("\nDemo completed successfully!")
    
    return estimated_trajectory, gt_trajectory


if __name__ == "__main__":
    estimated, ground_truth = main()