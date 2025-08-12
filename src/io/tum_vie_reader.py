"""
TUM-VIE dataset reader for loading calibration and data.
References: https://cvg.cit.tum.de/data/datasets/visual-inertial-event-dataset
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from src.common.data_structures import (
    IMUData, IMUMeasurement,
    CameraData, CameraFrame, CameraObservation, ImagePoint,
    Trajectory, TrajectoryState, Pose,
    Map, Landmark,
    CameraCalibration, IMUCalibration,
    CameraIntrinsics, CameraExtrinsics, CameraModel
)
from src.common.json_io import SimulationData


class TUMVIEReader:
    """Reader for TUM Visual-Inertial-Event datasets."""
    
    def __init__(self, dataset_path: Path):
        """
        Initialize TUM-VIE reader.
        
        Args:
            dataset_path: Path to dataset root directory
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Calibration files (calibration_a.json or calibration_b.json)
        self.calib_a_path = self.dataset_path / "calibration_a.json"
        self.calib_b_path = self.dataset_path / "calibration_b.json"
        
        # Data paths
        self.imu_path = self.dataset_path / "imu" / "data.csv"
        self.cam0_path = self.dataset_path / "cam0" / "data.csv"
        self.cam1_path = self.dataset_path / "cam1" / "data.csv"
        self.mocap_path = self.dataset_path / "mocap0" / "data.csv"
        
        # Check which calibration file exists
        if self.calib_a_path.exists():
            self.calib_path = self.calib_a_path
            self.calibration_type = "a"
        elif self.calib_b_path.exists():
            self.calib_path = self.calib_b_path
            self.calibration_type = "b"
        else:
            raise FileNotFoundError("No calibration file found (calibration_a.json or calibration_b.json)")
        
        # Load calibration
        self.calibration = self._load_calibration()
    
    def _load_calibration(self) -> Dict[str, Any]:
        """Load calibration data from JSON file."""
        with open(self.calib_path, 'r') as f:
            return json.load(f)
    
    def get_camera_calibration(self, camera_id: str = "cam0") -> CameraCalibration:
        """
        Extract camera calibration.
        
        Args:
            camera_id: Camera identifier ("cam0" or "cam1")
        
        Returns:
            CameraCalibration object
        """
        if camera_id not in self.calibration:
            raise ValueError(f"Camera {camera_id} not found in calibration")
        
        cam_data = self.calibration[camera_id]
        
        # Parse intrinsics
        intrinsics_data = cam_data["intrinsics"]
        model_str = cam_data.get("camera_model", "pinhole-radtan")
        
        # Map TUM-VIE model names to our enum
        model_map = {
            "pinhole-radtan": CameraModel.PINHOLE_RADTAN,
            "pinhole": CameraModel.PINHOLE,
            "kannala-brandt": CameraModel.KANNALA_BRANDT
        }
        model = model_map.get(model_str, CameraModel.PINHOLE_RADTAN)
        
        intrinsics = CameraIntrinsics(
            model=model,
            width=cam_data["resolution"][0],
            height=cam_data["resolution"][1],
            fx=intrinsics_data[0],
            fy=intrinsics_data[1],
            cx=intrinsics_data[2],
            cy=intrinsics_data[3],
            distortion=np.array(cam_data.get("distortion_coeffs", []))
        )
        
        # Parse extrinsics (T_cam_imu -> we need T_imu_cam = B_T_C)
        T_cam_imu = np.array(cam_data["T_cam_imu"])
        # Invert to get B_T_C (body/IMU to camera)
        B_T_C = np.linalg.inv(T_cam_imu)
        
        extrinsics = CameraExtrinsics(B_T_C=B_T_C)
        
        return CameraCalibration(
            camera_id=camera_id,
            intrinsics=intrinsics,
            extrinsics=extrinsics
        )
    
    def get_imu_calibration(self) -> IMUCalibration:
        """
        Extract IMU calibration.
        
        Returns:
            IMUCalibration object
        """
        if "imu0" not in self.calibration:
            # Use default values if not specified
            return IMUCalibration(
                imu_id="imu0",
                accelerometer_noise_density=0.01,  # m/s^2/sqrt(Hz)
                accelerometer_random_walk=0.001,   # m/s^3/sqrt(Hz)
                gyroscope_noise_density=0.0001,    # rad/s/sqrt(Hz)
                gyroscope_random_walk=0.00001,     # rad/s^2/sqrt(Hz)
                rate=200.0
            )
        
        imu_data = self.calibration["imu0"]
        
        return IMUCalibration(
            imu_id="imu0",
            accelerometer_noise_density=imu_data.get("accelerometer_noise_density", 0.01),
            accelerometer_random_walk=imu_data.get("accelerometer_random_walk", 0.001),
            gyroscope_noise_density=imu_data.get("gyroscope_noise_density", 0.0001),
            gyroscope_random_walk=imu_data.get("gyroscope_random_walk", 0.00001),
            rate=imu_data.get("rate_hz", 200.0)
        )
    
    def load_imu_data(self, max_samples: Optional[int] = None) -> IMUData:
        """
        Load IMU measurements from CSV.
        
        Args:
            max_samples: Maximum number of samples to load (None for all)
        
        Returns:
            IMUData object with measurements
        """
        if not self.imu_path.exists():
            raise FileNotFoundError(f"IMU data not found: {self.imu_path}")
        
        imu_data = IMUData(sensor_id="imu0")
        
        with open(self.imu_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header if present
            
            count = 0
            for row in reader:
                if max_samples and count >= max_samples:
                    break
                
                # Format: timestamp, wx, wy, wz, ax, ay, az
                timestamp = float(row[0]) / 1e9  # Convert ns to seconds
                gyro = np.array([float(row[1]), float(row[2]), float(row[3])])
                accel = np.array([float(row[4]), float(row[5]), float(row[6])])
                
                measurement = IMUMeasurement(
                    timestamp=timestamp,
                    accelerometer=accel,
                    gyroscope=gyro
                )
                imu_data.add_measurement(measurement)
                count += 1
        
        return imu_data
    
    def load_camera_data(
        self,
        camera_id: str = "cam0",
        max_frames: Optional[int] = None
    ) -> CameraData:
        """
        Load camera frame timestamps from CSV.
        Note: Actual feature observations would need to be extracted from images.
        
        Args:
            camera_id: Camera identifier ("cam0" or "cam1")
            max_frames: Maximum number of frames to load
        
        Returns:
            CameraData object with frame timestamps
        """
        cam_path = self.cam0_path if camera_id == "cam0" else self.cam1_path
        
        if not cam_path.exists():
            raise FileNotFoundError(f"Camera data not found: {cam_path}")
        
        camera_data = CameraData(camera_id=camera_id)
        
        with open(cam_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header if present
            
            count = 0
            for row in reader:
                if max_frames and count >= max_frames:
                    break
                
                # Format: timestamp, filename
                timestamp = float(row[0]) / 1e9  # Convert ns to seconds
                image_path = row[1] if len(row) > 1 else None
                
                # Create frame with no observations (would need feature extraction)
                frame = CameraFrame(
                    timestamp=timestamp,
                    camera_id=camera_id,
                    observations=[],
                    image_path=image_path
                )
                camera_data.add_frame(frame)
                count += 1
        
        return camera_data
    
    def load_ground_truth(self, max_poses: Optional[int] = None) -> Trajectory:
        """
        Load ground truth trajectory from mocap data.
        
        Args:
            max_poses: Maximum number of poses to load
        
        Returns:
            Trajectory object with ground truth poses
        """
        if not self.mocap_path.exists():
            raise FileNotFoundError(f"Mocap data not found: {self.mocap_path}")
        
        trajectory = Trajectory(frame_id="world")
        
        with open(self.mocap_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header if present
            
            count = 0
            for row in reader:
                if max_poses and count >= max_poses:
                    break
                
                # Format: timestamp, px, py, pz, qw, qx, qy, qz
                timestamp = float(row[0]) / 1e9  # Convert ns to seconds
                position = np.array([float(row[1]), float(row[2]), float(row[3])])
                quaternion = np.array([float(row[4]), float(row[5]), float(row[6]), float(row[7])])
                
                pose = Pose(
                    timestamp=timestamp,
                    position=position,
                    quaternion=quaternion
                )
                
                state = TrajectoryState(pose=pose)
                trajectory.add_state(state)
                count += 1
        
        return trajectory
    
    def to_simulation_data(
        self,
        max_imu_samples: Optional[int] = None,
        max_camera_frames: Optional[int] = None,
        max_gt_poses: Optional[int] = None
    ) -> SimulationData:
        """
        Convert TUM-VIE data to SimulationData format.
        
        Args:
            max_imu_samples: Limit IMU samples
            max_camera_frames: Limit camera frames
            max_gt_poses: Limit ground truth poses
        
        Returns:
            SimulationData object
        """
        sim_data = SimulationData()
        
        # Set metadata
        sim_data.set_metadata(
            trajectory_type="tum_vie",
            coordinate_system="ENU"
        )
        
        # Add calibrations
        try:
            cam0_calib = self.get_camera_calibration("cam0")
            sim_data.add_camera_calibration(cam0_calib)
        except (KeyError, ValueError):
            pass
        
        try:
            cam1_calib = self.get_camera_calibration("cam1")
            sim_data.add_camera_calibration(cam1_calib)
        except (KeyError, ValueError):
            pass
        
        imu_calib = self.get_imu_calibration()
        sim_data.add_imu_calibration(imu_calib)
        
        # Load and set data
        try:
            imu_data = self.load_imu_data(max_imu_samples)
            sim_data.set_imu_measurements(imu_data)
        except FileNotFoundError:
            pass
        
        try:
            cam0_data = self.load_camera_data("cam0", max_camera_frames)
            sim_data.add_camera_measurements(cam0_data)
        except FileNotFoundError:
            pass
        
        try:
            trajectory = self.load_ground_truth(max_gt_poses)
            sim_data.set_groundtruth_trajectory(trajectory)
        except FileNotFoundError:
            pass
        
        return sim_data


def load_tum_vie_dataset(
    dataset_path: Path,
    output_path: Optional[Path] = None,
    max_samples: Optional[int] = None
) -> SimulationData:
    """
    Convenience function to load TUM-VIE dataset.
    
    Args:
        dataset_path: Path to dataset root
        output_path: Optional path to save as JSON
        max_samples: Limit number of samples per sensor
    
    Returns:
        SimulationData object
    """
    reader = TUMVIEReader(dataset_path)
    sim_data = reader.to_simulation_data(
        max_imu_samples=max_samples,
        max_camera_frames=max_samples,
        max_gt_poses=max_samples
    )
    
    if output_path:
        sim_data.save(output_path)
        print(f"Saved TUM-VIE data to {output_path}")
    
    return sim_data