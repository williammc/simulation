#!/usr/bin/env python3
"""
Convert TUM-VI dataset to common JSON format for evaluation.
"""

import argparse
import json
import numpy as np
from pathlib import Path
import csv
from typing import Dict, List, Tuple, Optional
import yaml

# OpenCV is optional - only needed for visual feature extraction
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("Warning: OpenCV not available. Visual feature extraction will be skipped.")

from src.common.data_structures import (
    Trajectory, TrajectoryState, Pose,
    IMUData, IMUMeasurement,
    CameraData, CameraFrame, CameraObservation, ImagePoint,
    Map, Landmark,
    CameraCalibration, CameraIntrinsics, CameraExtrinsics, CameraModel,
    IMUCalibration
)
from src.common.json_io import SimulationData


def load_tumvi_trajectory(groundtruth_file: Path) -> Trajectory:
    """
    Load ground truth trajectory from TUM-VI format.
    
    Args:
        groundtruth_file: Path to groundtruth.txt file
        
    Returns:
        Trajectory object
    """
    trajectory = Trajectory()
    
    with open(groundtruth_file, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            
            # Skip empty or malformed rows
            if len(row) < 8:
                continue
                
            try:
                # Format: timestamp tx ty tz qx qy qz qw
                timestamp = float(row[0])
                position = np.array([float(row[1]), float(row[2]), float(row[3])])
                quaternion = np.array([float(row[7]), float(row[4]), float(row[5]), float(row[6])])  # qw qx qy qz
                
                pose = Pose(timestamp=timestamp, position=position, quaternion=quaternion)
                state = TrajectoryState(pose=pose)
                trajectory.add_state(state)
            except (ValueError, IndexError) as e:
                # Skip malformed lines
                continue
    
    return trajectory


def load_tumvi_imu(imu_file: Path) -> IMUData:
    """
    Load IMU data from TUM-VI format.
    
    Args:
        imu_file: Path to imu.txt file
        
    Returns:
        IMUData object
    """
    imu_data = IMUData()
    imu_data.sensor_id = "imu0"
    imu_data.rate = 200.0
    
    with open(imu_file, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            
            # Skip empty or malformed rows
            if len(row) < 7:
                continue
                
            try:
                # Format: timestamp ax ay az gx gy gz
                timestamp = float(row[0])
                accelerometer = np.array([float(row[1]), float(row[2]), float(row[3])])
                gyroscope = np.array([float(row[4]), float(row[5]), float(row[6])])
                
                measurement = IMUMeasurement(
                    timestamp=timestamp,
                    accelerometer=accelerometer,
                    gyroscope=gyroscope
                )
                imu_data.add_measurement(measurement)
            except (ValueError, IndexError) as e:
                # Skip malformed lines
                continue
    
    return imu_data


def generate_synthetic_landmarks(trajectory: Trajectory, num_landmarks: int = 200,
                                bounds: Optional[np.ndarray] = None) -> Map:
    """
    Generate synthetic 3D landmarks around the trajectory.
    
    Args:
        trajectory: Ground truth trajectory
        num_landmarks: Number of landmarks to generate
        bounds: Optional 6D array [xmin, xmax, ymin, ymax, zmin, zmax]
        
    Returns:
        Map with synthetic landmarks
    """
    landmarks = Map(frame_id="world")
    
    if len(trajectory.states) == 0:
        return landmarks
    
    # Compute trajectory bounds if not provided
    if bounds is None:
        positions = np.array([s.pose.position for s in trajectory.states])
        center = positions.mean(axis=0)
        extent = positions.max(axis=0) - positions.min(axis=0)
        
        # Create bounds around trajectory with some margin
        margin = max(extent) * 1.5
        bounds = np.array([
            center[0] - margin, center[0] + margin,
            center[1] - margin, center[1] + margin,
            center[2] - margin/2, center[2] + margin  # Higher landmarks above trajectory
        ])
    
    # Generate random landmarks within bounds
    np.random.seed(42)  # Reproducible landmarks
    for lid in range(num_landmarks):
        position = np.array([
            np.random.uniform(bounds[0], bounds[1]),
            np.random.uniform(bounds[2], bounds[3]),
            np.random.uniform(bounds[4], bounds[5])
        ])
        
        landmark = Landmark(id=lid, position=position)
        landmarks.add_landmark(landmark)
    
    print(f"Generated {num_landmarks} synthetic landmarks")
    return landmarks


def generate_keyframes_and_observations(trajectory: Trajectory, landmarks: Map,
                                       camera_calib: CameraCalibration,
                                       keyframe_interval: float = 0.1,
                                       pixel_noise_std: float = 1.0,
                                       max_range: float = 20.0) -> CameraData:
    """
    Generate keyframes and camera observations from ground truth.
    
    Args:
        trajectory: Ground truth trajectory
        landmarks: 3D landmarks
        camera_calib: Camera calibration with intrinsics and extrinsics
        keyframe_interval: Time interval between keyframes (seconds)
        pixel_noise_std: Standard deviation of pixel noise
        max_range: Maximum observation range (meters)
        
    Returns:
        CameraData with synthetic observations
    """
    camera_data = CameraData(camera_id=camera_calib.camera_id, rate=1.0/keyframe_interval)
    
    if len(trajectory.states) == 0:
        return camera_data
    
    # Generate keyframes at regular intervals
    start_time = trajectory.states[0].pose.timestamp
    end_time = trajectory.states[-1].pose.timestamp
    keyframe_times = np.arange(start_time, end_time, keyframe_interval)
    
    print(f"Generating {len(keyframe_times)} keyframes with {len(landmarks.landmarks)} landmarks")
    
    for kf_time in keyframe_times:
        # Get interpolated pose at keyframe time
        pose = trajectory.get_pose_at_time(kf_time)
        if pose is None:
            continue
        
        # Transform from world to body frame
        W_T_B = pose.to_matrix()
        B_T_W = np.linalg.inv(W_T_B)
        
        # Transform from body to camera frame using extrinsics
        B_T_C = camera_calib.extrinsics.B_T_C
        C_T_B = np.linalg.inv(B_T_C)
        C_T_W = C_T_B @ B_T_W
        
        observations = []
        
        # Project each landmark to camera
        for landmark in landmarks.landmarks.values():
            # Transform landmark to camera frame
            pt_world = np.append(landmark.position, 1.0)
            pt_camera = C_T_W @ pt_world
            
            # Check if behind camera or too far
            if pt_camera[2] <= 0.1 or pt_camera[2] > max_range:
                continue
            
            # Project to image plane
            u = camera_calib.intrinsics.fx * pt_camera[0] / pt_camera[2] + camera_calib.intrinsics.cx
            v = camera_calib.intrinsics.fy * pt_camera[1] / pt_camera[2] + camera_calib.intrinsics.cy
            
            # Add pixel noise
            if pixel_noise_std > 0:
                u += np.random.normal(0, pixel_noise_std)
                v += np.random.normal(0, pixel_noise_std)
            
            # Check if within image bounds
            if 0 <= u < camera_calib.intrinsics.width and 0 <= v < camera_calib.intrinsics.height:
                obs = CameraObservation(
                    landmark_id=landmark.id,
                    pixel=ImagePoint(u=u, v=v)
                )
                observations.append(obs)
        
        # Create camera frame if there are observations
        if observations:
            frame = CameraFrame(
                timestamp=kf_time,
                camera_id=camera_calib.camera_id,
                observations=observations
            )
            camera_data.add_frame(frame)
    
    print(f"Created {len(camera_data.frames)} keyframes with average {np.mean([len(f.observations) for f in camera_data.frames]):.1f} observations per frame")
    return camera_data


def load_tumvi_calibration(calib_dir: Path) -> Dict:
    """
    Load calibration from TUM-VI format.
    
    Args:
        calib_dir: Directory containing calibration files
        
    Returns:
        Dictionary of calibrations
    """
    calibrations = {}
    
    # Load camera calibration
    cam_calib_file = calib_dir / "camera.yaml"
    if cam_calib_file.exists():
        with open(cam_calib_file, 'r') as f:
            cam_yaml = yaml.safe_load(f)
            
        intrinsics = CameraIntrinsics(
            model=CameraModel.PINHOLE,
            width=cam_yaml.get('resolution', [640, 480])[0],
            height=cam_yaml.get('resolution', [640, 480])[1],
            fx=cam_yaml.get('intrinsics', [500, 500, 320, 240])[0],
            fy=cam_yaml.get('intrinsics', [500, 500, 320, 240])[1],
            cx=cam_yaml.get('intrinsics', [500, 500, 320, 240])[2],
            cy=cam_yaml.get('intrinsics', [500, 500, 320, 240])[3],
            distortion=np.array(cam_yaml.get('distortion_coefficients', [0, 0, 0, 0, 0]))
        )
        
        extrinsics = CameraExtrinsics(
            B_T_C=np.eye(4)  # Identity for now
        )
        
        calibrations['cam0'] = CameraCalibration(
            camera_id="cam0",
            intrinsics=intrinsics,
            extrinsics=extrinsics
        )
    else:
        # Default calibration
        intrinsics = CameraIntrinsics(
            model=CameraModel.PINHOLE,
            width=640,
            height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            distortion=np.zeros(5)
        )
        extrinsics = CameraExtrinsics(B_T_C=np.eye(4))
        calibrations['cam0'] = CameraCalibration(
            camera_id="cam0",
            intrinsics=intrinsics,
            extrinsics=extrinsics
        )
    
    # Load IMU calibration
    imu_calib_file = calib_dir / "imu.yaml"
    if imu_calib_file.exists():
        with open(imu_calib_file, 'r') as f:
            imu_yaml = yaml.safe_load(f)
            
        from src.common.data_structures import IMUExtrinsics
        calibrations['imu0'] = IMUCalibration(
            imu_id="imu0",
            accelerometer_noise_density=imu_yaml.get('accelerometer_noise_density', 0.01),
            accelerometer_random_walk=imu_yaml.get('accelerometer_random_walk', 0.001),
            gyroscope_noise_density=imu_yaml.get('gyroscope_noise_density', 0.001),
            gyroscope_random_walk=imu_yaml.get('gyroscope_random_walk', 0.0001),
            extrinsics=IMUExtrinsics(B_T_S=np.eye(4))
        )
    else:
        # Default IMU calibration
        from src.common.data_structures import IMUExtrinsics
        calibrations['imu0'] = IMUCalibration(
            imu_id="imu0",
            accelerometer_noise_density=0.01,
            accelerometer_random_walk=0.001,
            gyroscope_noise_density=0.001,
            gyroscope_random_walk=0.0001,
            extrinsics=IMUExtrinsics(B_T_S=np.eye(4))
        )
    
    return calibrations


def convert_tumvi_dataset(dataset_dir: Path, output_file: Path, 
                         num_landmarks: int = 200,
                         keyframe_interval: float = 0.1,
                         pixel_noise_std: float = 1.0):
    """
    Convert TUM-VI dataset to common format with synthetic observations.
    
    Args:
        dataset_dir: Path to TUM-VI dataset directory
        output_file: Output JSON file path
        num_landmarks: Number of synthetic landmarks to generate
        keyframe_interval: Time between keyframes in seconds
        pixel_noise_std: Standard deviation of pixel measurement noise
    """
    print(f"Converting TUM-VI dataset: {dataset_dir}")
    print(f"  - Generating {num_landmarks} synthetic landmarks")
    print(f"  - Keyframe interval: {keyframe_interval}s")
    print(f"  - Pixel noise std: {pixel_noise_std} pixels")
    
    # Load ground truth trajectory from mocap
    groundtruth_file = dataset_dir / "groundtruth.txt"
    if groundtruth_file.exists():
        trajectory = load_tumvi_trajectory(groundtruth_file)
        print(f"Loaded {len(trajectory.states)} ground truth poses from mocap")
    else:
        print("Error: No ground truth file found")
        trajectory = Trajectory()
    
    # Load IMU data
    imu_file = dataset_dir / "imu.txt"
    if imu_file.exists():
        imu_data = load_tumvi_imu(imu_file)
        print(f"Loaded {len(imu_data.measurements)} IMU measurements")
    else:
        print("Warning: No IMU file found, creating empty IMU data")
        imu_data = IMUData()
    imu_data.sensor_id = "imu0"
    imu_data.rate = 200.0
    
    # Load calibrations
    calib_dir = dataset_dir / "calibration"
    if not calib_dir.exists():
        calib_dir = dataset_dir
    calibrations = load_tumvi_calibration(calib_dir)
    
    # Generate synthetic 3D landmarks around the trajectory
    landmarks = generate_synthetic_landmarks(trajectory, num_landmarks)
    
    # Generate keyframes and observations using camera calibration
    if 'cam0' in calibrations:
        camera_data = generate_keyframes_and_observations(
            trajectory, landmarks, calibrations['cam0'],
            keyframe_interval, pixel_noise_std
        )
    else:
        print("Warning: No camera calibration found, using default")
        # Create default calibration
        from src.common.data_structures import CameraIntrinsics, CameraExtrinsics
        intrinsics = CameraIntrinsics(
            model=CameraModel.PINHOLE,
            width=640, height=480,
            fx=500.0, fy=500.0,
            cx=320.0, cy=240.0,
            distortion=np.zeros(5)
        )
        extrinsics = CameraExtrinsics(B_T_C=np.eye(4))
        default_calib = CameraCalibration(
            camera_id="cam0",
            intrinsics=intrinsics,
            extrinsics=extrinsics
        )
        camera_data = generate_keyframes_and_observations(
            trajectory, landmarks, default_calib,
            keyframe_interval, pixel_noise_std
        )
    
    # Create simulation data
    sim_data = SimulationData()
    
    # Set metadata
    sim_data.set_metadata(
        trajectory_type='real',
        duration=trajectory.states[-1].pose.timestamp if trajectory.states else 0
    )
    sim_data.metadata['source'] = 'TUM-VI'
    sim_data.metadata['dataset'] = dataset_dir.name
    sim_data.metadata['num_poses'] = len(trajectory.states)
    sim_data.metadata['num_landmarks'] = len(landmarks.landmarks)
    sim_data.metadata['num_imu_measurements'] = len(imu_data.measurements)
    sim_data.metadata['num_camera_frames'] = len(camera_data.frames)
    
    # Set ground truth
    sim_data.set_groundtruth_trajectory(trajectory)
    sim_data.set_groundtruth_landmarks(landmarks)
    
    # Set measurements
    sim_data.set_imu_measurements(imu_data)
    sim_data.add_camera_measurements(camera_data)
    
    # Add calibrations
    if 'cam0' in calibrations:
        sim_data.add_camera_calibration(calibrations['cam0'])
    if 'imu0' in calibrations:
        sim_data.add_imu_calibration(calibrations['imu0'])
    
    # Save to JSON
    sim_data.save(output_file)
    print(f"Saved converted dataset to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert TUM-VI dataset to common format")
    parser.add_argument("dataset_dir", type=Path, help="Path to TUM-VI dataset directory")
    parser.add_argument("output_file", type=Path, help="Output JSON file path")
    parser.add_argument("--num-landmarks", type=int, default=200,
                       help="Number of synthetic landmarks to generate (default: 200)")
    parser.add_argument("--keyframe-interval", type=float, default=0.1,
                       help="Time between keyframes in seconds (default: 0.1)")
    parser.add_argument("--pixel-noise", type=float, default=1.0,
                       help="Standard deviation of pixel noise (default: 1.0)")
    
    args = parser.parse_args()
    
    if not args.dataset_dir.exists():
        print(f"Error: Dataset directory not found: {args.dataset_dir}")
        return 1
    
    # Create output directory if needed
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        convert_tumvi_dataset(
            args.dataset_dir, 
            args.output_file,
            num_landmarks=args.num_landmarks,
            keyframe_interval=args.keyframe_interval,
            pixel_noise_std=args.pixel_noise
        )
        return 0
    except Exception as e:
        print(f"Error converting dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())