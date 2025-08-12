"""
JSON I/O for simulation data following the schema from technical specifications.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

import numpy as np

from src.common.data_structures import (
    IMUData, IMUMeasurement,
    CameraData, CameraFrame, CameraObservation, ImagePoint,
    Trajectory, TrajectoryState, Pose,
    Map, Landmark,
    CameraCalibration, IMUCalibration
)


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays."""
    
    def default(self, obj):
        """Convert numpy arrays to lists."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)


class SimulationData:
    """Complete simulation data container matching JSON schema."""
    
    def __init__(self):
        """Initialize empty simulation data."""
        self.metadata: Dict[str, Any] = {}
        self.calibration: Dict[str, Any] = {
            "cameras": [],
            "imus": []
        }
        self.groundtruth: Dict[str, Any] = {
            "trajectory": None,
            "landmarks": None
        }
        self.measurements: Dict[str, Any] = {
            "imu": None,
            "camera_frames": []
        }
    
    def set_metadata(
        self,
        trajectory_type: str = "unknown",
        duration: float = 0.0,
        coordinate_system: str = "ENU",
        seed: Optional[int] = None
    ):
        """Set simulation metadata."""
        self.metadata = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "trajectory_type": trajectory_type,
            "duration": duration,
            "coordinate_system": coordinate_system,
            "units": {
                "position": "meters",
                "rotation": "quaternion_wxyz",
                "time": "seconds"
            }
        }
        if seed is not None:
            self.metadata["seed"] = seed
    
    def add_camera_calibration(self, calib: CameraCalibration):
        """Add camera calibration."""
        calib_dict = {
            "id": calib.camera_id,
            "model": calib.intrinsics.model.value,
            "width": calib.intrinsics.width,
            "height": calib.intrinsics.height,
            "intrinsics": {
                "fx": calib.intrinsics.fx,
                "fy": calib.intrinsics.fy,
                "cx": calib.intrinsics.cx,
                "cy": calib.intrinsics.cy
            },
            "distortion": calib.intrinsics.distortion.tolist(),
            "T_BC": calib.extrinsics.B_T_C.tolist()
        }
        self.calibration["cameras"].append(calib_dict)
    
    def add_imu_calibration(self, calib: IMUCalibration):
        """Add IMU calibration."""
        calib_dict = {
            "id": calib.imu_id,
            "accelerometer": {
                "noise_density": calib.accelerometer_noise_density,
                "random_walk": calib.accelerometer_random_walk
            },
            "gyroscope": {
                "noise_density": calib.gyroscope_noise_density,
                "random_walk": calib.gyroscope_random_walk
            },
            "sampling_rate": calib.rate
        }
        self.calibration["imus"].append(calib_dict)
    
    def set_groundtruth_trajectory(self, trajectory: Trajectory):
        """Set ground truth trajectory."""
        traj_data = []
        for state in trajectory.states:
            state_dict = {
                "timestamp": state.pose.timestamp,
                "position": state.pose.position.tolist(),
                "quaternion": state.pose.quaternion.tolist()
            }
            if state.velocity is not None:
                state_dict["velocity"] = state.velocity.tolist()
            if state.angular_velocity is not None:
                state_dict["angular_velocity"] = state.angular_velocity.tolist()
            traj_data.append(state_dict)
        
        self.groundtruth["trajectory"] = traj_data
    
    def set_groundtruth_landmarks(self, map_data: Map):
        """Set ground truth landmarks."""
        landmarks_data = []
        for landmark in map_data.landmarks.values():
            lm_dict = {
                "id": landmark.id,
                "position": landmark.position.tolist()
            }
            if landmark.descriptor is not None:
                lm_dict["descriptor"] = landmark.descriptor.tolist()
            landmarks_data.append(lm_dict)
        
        self.groundtruth["landmarks"] = landmarks_data
    
    def set_imu_measurements(self, imu_data: IMUData):
        """Set IMU measurements."""
        measurements = []
        for meas in imu_data.measurements:
            measurements.append({
                "timestamp": meas.timestamp,
                "accelerometer": meas.accelerometer.tolist(),
                "gyroscope": meas.gyroscope.tolist()
            })
        self.measurements["imu"] = measurements
    
    def add_camera_measurements(self, camera_data: CameraData):
        """Add camera measurements."""
        for frame in camera_data.frames:
            observations = []
            for obs in frame.observations:
                obs_dict = {
                    "landmark_id": obs.landmark_id,
                    "pixel": [obs.pixel.u, obs.pixel.v]
                }
                if obs.descriptor is not None:
                    obs_dict["descriptor"] = obs.descriptor.tolist()
                observations.append(obs_dict)
            
            frame_dict = {
                "timestamp": frame.timestamp,
                "camera_id": frame.camera_id,
                "observations": observations
            }
            self.measurements["camera_frames"].append(frame_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary matching JSON schema."""
        return {
            "metadata": self.metadata,
            "calibration": self.calibration,
            "groundtruth": self.groundtruth,
            "measurements": self.measurements
        }
    
    def save(self, filepath: Union[str, Path]):
        """Save to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'SimulationData':
        """Load from JSON file."""
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        sim_data = cls()
        sim_data.metadata = data.get("metadata", {})
        sim_data.calibration = data.get("calibration", {"cameras": [], "imus": []})
        sim_data.groundtruth = data.get("groundtruth", {"trajectory": None, "landmarks": None})
        sim_data.measurements = data.get("measurements", {"imu": None, "camera_frames": []})
        
        return sim_data
    
    def get_trajectory(self) -> Optional[Trajectory]:
        """Extract trajectory from ground truth."""
        if not self.groundtruth.get("trajectory"):
            return None
        
        trajectory = Trajectory()
        for state_dict in self.groundtruth["trajectory"]:
            pose = Pose(
                timestamp=state_dict["timestamp"],
                position=np.array(state_dict["position"]),
                quaternion=np.array(state_dict["quaternion"])
            )
            
            velocity = None
            angular_velocity = None
            if "velocity" in state_dict:
                velocity = np.array(state_dict["velocity"])
            if "angular_velocity" in state_dict:
                angular_velocity = np.array(state_dict["angular_velocity"])
            
            state = TrajectoryState(
                pose=pose,
                velocity=velocity,
                angular_velocity=angular_velocity
            )
            trajectory.add_state(state)
        
        return trajectory
    
    def get_map(self) -> Optional[Map]:
        """Extract map from ground truth landmarks."""
        if not self.groundtruth.get("landmarks"):
            return None
        
        map_data = Map()
        for lm_dict in self.groundtruth["landmarks"]:
            descriptor = None
            if "descriptor" in lm_dict:
                descriptor = np.array(lm_dict["descriptor"])
            
            landmark = Landmark(
                id=lm_dict["id"],
                position=np.array(lm_dict["position"]),
                descriptor=descriptor
            )
            map_data.add_landmark(landmark)
        
        return map_data
    
    def get_imu_data(self) -> Optional[IMUData]:
        """Extract IMU data from measurements."""
        if not self.measurements.get("imu"):
            return None
        
        imu_data = IMUData()
        for meas_dict in self.measurements["imu"]:
            measurement = IMUMeasurement(
                timestamp=meas_dict["timestamp"],
                accelerometer=np.array(meas_dict["accelerometer"]),
                gyroscope=np.array(meas_dict["gyroscope"])
            )
            imu_data.add_measurement(measurement)
        
        return imu_data
    
    def get_camera_data(self, camera_id: Optional[str] = None) -> Optional[CameraData]:
        """Extract camera data from measurements."""
        if not self.measurements.get("camera_frames"):
            return None
        
        # Group frames by camera
        camera_frames: Dict[str, list] = {}
        for frame_dict in self.measurements["camera_frames"]:
            cam_id = frame_dict["camera_id"]
            if camera_id and cam_id != camera_id:
                continue
            
            if cam_id not in camera_frames:
                camera_frames[cam_id] = []
            
            observations = []
            for obs_dict in frame_dict["observations"]:
                descriptor = None
                if "descriptor" in obs_dict:
                    descriptor = np.array(obs_dict["descriptor"])
                
                observation = CameraObservation(
                    landmark_id=obs_dict["landmark_id"],
                    pixel=ImagePoint(u=obs_dict["pixel"][0], v=obs_dict["pixel"][1]),
                    descriptor=descriptor
                )
                observations.append(observation)
            
            frame = CameraFrame(
                timestamp=frame_dict["timestamp"],
                camera_id=cam_id,
                observations=observations
            )
            camera_frames[cam_id].append(frame)
        
        # Return requested camera or first available
        if camera_id and camera_id in camera_frames:
            camera_data = CameraData(camera_id=camera_id)
            for frame in sorted(camera_frames[camera_id], key=lambda f: f.timestamp):
                camera_data.add_frame(frame)
            return camera_data
        elif camera_frames:
            first_cam = list(camera_frames.keys())[0]
            camera_data = CameraData(camera_id=first_cam)
            for frame in sorted(camera_frames[first_cam], key=lambda f: f.timestamp):
                camera_data.add_frame(frame)
            return camera_data
        
        return None


# Convenience functions

def save_simulation_data(
    filepath: Union[str, Path],
    trajectory: Optional[Trajectory] = None,
    landmarks: Optional[Map] = None,
    imu_data: Optional[IMUData] = None,
    camera_data: Optional[Union[CameraData, list[CameraData]]] = None,
    camera_calibrations: Optional[list[CameraCalibration]] = None,
    imu_calibrations: Optional[list[IMUCalibration]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save simulation data to JSON file.
    
    Args:
        filepath: Output file path
        trajectory: Ground truth trajectory
        landmarks: Ground truth landmarks
        imu_data: IMU measurements
        camera_data: Camera measurements (single or list)
        camera_calibrations: Camera calibrations
        imu_calibrations: IMU calibrations
        metadata: Additional metadata
    """
    sim_data = SimulationData()
    
    # Set metadata
    if metadata:
        sim_data.set_metadata(**metadata)
    else:
        sim_data.set_metadata()
    
    # Add calibrations
    if camera_calibrations:
        for calib in camera_calibrations:
            sim_data.add_camera_calibration(calib)
    
    if imu_calibrations:
        for calib in imu_calibrations:
            sim_data.add_imu_calibration(calib)
    
    # Set ground truth
    if trajectory:
        sim_data.set_groundtruth_trajectory(trajectory)
    
    if landmarks:
        sim_data.set_groundtruth_landmarks(landmarks)
    
    # Set measurements
    if imu_data:
        sim_data.set_imu_measurements(imu_data)
    
    if camera_data:
        if isinstance(camera_data, list):
            for cam_data in camera_data:
                sim_data.add_camera_measurements(cam_data)
        else:
            sim_data.add_camera_measurements(camera_data)
    
    # Save to file
    sim_data.save(filepath)


def load_simulation_data(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load simulation data from JSON file.
    
    Args:
        filepath: Input file path
    
    Returns:
        Dictionary with keys:
        - metadata: Simulation metadata
        - trajectory: Ground truth trajectory (if available)
        - landmarks: Ground truth landmarks (if available)
        - imu_data: IMU measurements (if available)
        - camera_data: Camera measurements (if available)
        - camera_calibrations: List of camera calibrations
        - imu_calibrations: List of IMU calibrations
        - raw: Raw SimulationData object
    """
    sim_data = SimulationData.load(filepath)
    
    # Extract calibrations
    camera_calibrations = []
    imu_calibrations = []
    
    if hasattr(sim_data, 'calibration') and sim_data.calibration:
        if hasattr(sim_data.calibration, 'cameras'):
            camera_calibrations = sim_data.calibration.cameras
        if hasattr(sim_data.calibration, 'imus'):
            imu_calibrations = sim_data.calibration.imus
    
    return {
        "metadata": sim_data.metadata,
        "trajectory": sim_data.get_trajectory(),
        "landmarks": sim_data.get_map(),
        "imu_data": sim_data.get_imu_data(),
        "camera_data": sim_data.get_camera_data(),
        "camera_calibrations": camera_calibrations,
        "imu_calibrations": imu_calibrations,
        "raw": sim_data
    }