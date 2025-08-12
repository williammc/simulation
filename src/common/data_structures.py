"""
Core data structures for SLAM simulation.
Following the naming convention: A_X_B means X transforms FROM B TO A.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
import numpy as np
from pydantic import BaseModel, Field, field_validator
from enum import Enum


# ============================================================================
# IMU Data Structures
# ============================================================================

@dataclass
class IMUMeasurement:
    """Single IMU measurement containing accelerometer and gyroscope readings."""
    timestamp: float  # Time in seconds
    accelerometer: np.ndarray  # 3x1 acceleration in m/s² (body frame)
    gyroscope: np.ndarray  # 3x1 angular velocity in rad/s (body frame)
    
    def __post_init__(self):
        """Validate and convert to numpy arrays."""
        self.accelerometer = np.asarray(self.accelerometer).flatten()
        self.gyroscope = np.asarray(self.gyroscope).flatten()
        
        if len(self.accelerometer) != 3:
            raise ValueError(f"Accelerometer must be 3D, got {len(self.accelerometer)}")
        if len(self.gyroscope) != 3:
            raise ValueError(f"Gyroscope must be 3D, got {len(self.gyroscope)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "accelerometer": self.accelerometer.tolist(),
            "gyroscope": self.gyroscope.tolist()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IMUMeasurement':
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            accelerometer=np.array(data["accelerometer"]),
            gyroscope=np.array(data["gyroscope"])
        )


@dataclass
class IMUData:
    """Collection of IMU measurements with metadata."""
    measurements: List[IMUMeasurement] = field(default_factory=list)
    sensor_id: str = "imu0"
    rate: float = 200.0  # Hz
    
    def add_measurement(self, measurement: IMUMeasurement):
        """Add a measurement maintaining time order."""
        if self.measurements and measurement.timestamp <= self.measurements[-1].timestamp:
            raise ValueError("Measurements must be added in chronological order")
        self.measurements.append(measurement)
    
    def get_time_range(self) -> tuple[float, float]:
        """Get the time range of measurements."""
        if not self.measurements:
            return (0.0, 0.0)
        return (self.measurements[0].timestamp, self.measurements[-1].timestamp)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sensor_id": self.sensor_id,
            "rate": self.rate,
            "measurements": [m.to_dict() for m in self.measurements]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IMUData':
        """Create from dictionary."""
        return cls(
            sensor_id=data.get("sensor_id", "imu0"),
            rate=data.get("rate", 200.0),
            measurements=[IMUMeasurement.from_dict(m) for m in data["measurements"]]
        )


# ============================================================================
# Camera Data Structures
# ============================================================================

@dataclass
class ImagePoint:
    """2D point in image coordinates."""
    u: float  # Horizontal coordinate in pixels
    v: float  # Vertical coordinate in pixels
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.u, self.v])
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {"u": self.u, "v": self.v}
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'ImagePoint':
        """Create from dictionary."""
        return cls(u=data["u"], v=data["v"])


@dataclass
class CameraObservation:
    """Single camera observation of a landmark."""
    landmark_id: int  # ID of the observed landmark
    pixel: ImagePoint  # 2D pixel coordinates
    descriptor: Optional[np.ndarray] = None  # Feature descriptor (if available)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "landmark_id": self.landmark_id,
            "pixel": self.pixel.to_dict()
        }
        if self.descriptor is not None:
            result["descriptor"] = self.descriptor.tolist()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraObservation':
        """Create from dictionary."""
        descriptor = None
        if "descriptor" in data and data["descriptor"] is not None:
            descriptor = np.array(data["descriptor"])
        
        return cls(
            landmark_id=data["landmark_id"],
            pixel=ImagePoint.from_dict(data["pixel"]),
            descriptor=descriptor
        )


@dataclass
class CameraFrame:
    """Single camera frame with observations."""
    timestamp: float  # Time in seconds
    camera_id: str  # Camera identifier (e.g., "cam0", "cam1")
    observations: List[CameraObservation] = field(default_factory=list)
    image_path: Optional[str] = None  # Path to actual image file (if stored)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "timestamp": self.timestamp,
            "camera_id": self.camera_id,
            "observations": [obs.to_dict() for obs in self.observations]
        }
        if self.image_path:
            result["image_path"] = self.image_path
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraFrame':
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            camera_id=data["camera_id"],
            observations=[CameraObservation.from_dict(obs) for obs in data["observations"]],
            image_path=data.get("image_path")
        )


@dataclass
class CameraData:
    """Collection of camera frames with metadata."""
    frames: List[CameraFrame] = field(default_factory=list)
    camera_id: str = "cam0"
    rate: float = 30.0  # Hz
    
    def add_frame(self, frame: CameraFrame):
        """Add a frame maintaining time order."""
        if self.frames and frame.timestamp <= self.frames[-1].timestamp:
            raise ValueError("Frames must be added in chronological order")
        self.frames.append(frame)
    
    def get_time_range(self) -> tuple[float, float]:
        """Get the time range of frames."""
        if not self.frames:
            return (0.0, 0.0)
        return (self.frames[0].timestamp, self.frames[-1].timestamp)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "camera_id": self.camera_id,
            "rate": self.rate,
            "frames": [f.to_dict() for f in self.frames]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraData':
        """Create from dictionary."""
        return cls(
            camera_id=data.get("camera_id", "cam0"),
            rate=data.get("rate", 30.0),
            frames=[CameraFrame.from_dict(f) for f in data["frames"]]
        )


# ============================================================================
# Trajectory/Pose Data Structures
# ============================================================================

@dataclass
class Pose:
    """6DOF pose with position and orientation."""
    timestamp: float  # Time in seconds
    position: np.ndarray  # 3x1 position vector [x, y, z]
    quaternion: np.ndarray  # 4x1 quaternion [w, x, y, z]
    
    def __post_init__(self):
        """Validate and normalize quaternion."""
        self.position = np.asarray(self.position).flatten()
        self.quaternion = np.asarray(self.quaternion).flatten()
        
        if len(self.position) != 3:
            raise ValueError(f"Position must be 3D, got {len(self.position)}")
        if len(self.quaternion) != 4:
            raise ValueError(f"Quaternion must be 4D, got {len(self.quaternion)}")
        
        # Normalize quaternion
        norm = np.linalg.norm(self.quaternion)
        if norm > 1e-6:
            self.quaternion = self.quaternion / norm
        else:
            self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 transformation matrix."""
        from src.utils.math_utils import quaternion_to_rotation_matrix
        
        T = np.eye(4)
        T[:3, :3] = quaternion_to_rotation_matrix(self.quaternion)
        T[:3, 3] = self.position
        return T
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "position": self.position.tolist(),
            "quaternion": self.quaternion.tolist()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Pose':
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            position=np.array(data["position"]),
            quaternion=np.array(data["quaternion"])
        )
    
    @classmethod
    def from_matrix(cls, T: np.ndarray, timestamp: float = 0.0) -> 'Pose':
        """Create from 4x4 transformation matrix."""
        from src.utils.math_utils import rotation_matrix_to_quaternion
        
        return cls(
            timestamp=timestamp,
            position=T[:3, 3],
            quaternion=rotation_matrix_to_quaternion(T[:3, :3])
        )


@dataclass
class TrajectoryState:
    """Complete state including pose, velocity, and angular velocity."""
    pose: Pose
    velocity: Optional[np.ndarray] = None  # 3x1 linear velocity in world frame
    angular_velocity: Optional[np.ndarray] = None  # 3x1 angular velocity in body frame
    
    def __post_init__(self):
        """Validate velocities."""
        if self.velocity is not None:
            self.velocity = np.asarray(self.velocity).flatten()
            if len(self.velocity) != 3:
                raise ValueError(f"Velocity must be 3D, got {len(self.velocity)}")
        
        if self.angular_velocity is not None:
            self.angular_velocity = np.asarray(self.angular_velocity).flatten()
            if len(self.angular_velocity) != 3:
                raise ValueError(f"Angular velocity must be 3D, got {len(self.angular_velocity)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {"pose": self.pose.to_dict()}
        
        if self.velocity is not None:
            result["velocity"] = self.velocity.tolist()
        if self.angular_velocity is not None:
            result["angular_velocity"] = self.angular_velocity.tolist()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrajectoryState':
        """Create from dictionary."""
        velocity = None
        angular_velocity = None
        
        if "velocity" in data and data["velocity"] is not None:
            velocity = np.array(data["velocity"])
        if "angular_velocity" in data and data["angular_velocity"] is not None:
            angular_velocity = np.array(data["angular_velocity"])
        
        return cls(
            pose=Pose.from_dict(data["pose"]),
            velocity=velocity,
            angular_velocity=angular_velocity
        )


@dataclass
class Trajectory:
    """Complete trajectory with poses and velocities."""
    states: List[TrajectoryState] = field(default_factory=list)
    frame_id: str = "world"  # Reference frame
    
    def add_state(self, state: TrajectoryState):
        """Add a state maintaining time order."""
        if self.states and state.pose.timestamp <= self.states[-1].pose.timestamp:
            raise ValueError("States must be added in chronological order")
        self.states.append(state)
    
    def get_time_range(self) -> tuple[float, float]:
        """Get the time range of trajectory."""
        if not self.states:
            return (0.0, 0.0)
        return (self.states[0].pose.timestamp, self.states[-1].pose.timestamp)
    
    def get_pose_at_time(self, timestamp: float) -> Optional[Pose]:
        """Get interpolated pose at given timestamp."""
        if not self.states:
            return None
        
        # Find surrounding poses
        for i in range(len(self.states) - 1):
            if self.states[i].pose.timestamp <= timestamp <= self.states[i+1].pose.timestamp:
                # Interpolate between poses
                t1 = self.states[i].pose.timestamp
                t2 = self.states[i+1].pose.timestamp
                alpha = (timestamp - t1) / (t2 - t1) if t2 > t1 else 0.0
                
                # Linear interpolation for position
                pos1 = self.states[i].pose.position
                pos2 = self.states[i+1].pose.position
                position = (1 - alpha) * pos1 + alpha * pos2
                
                # SLERP for orientation
                from src.utils.math_utils import quaternion_slerp
                q1 = self.states[i].pose.quaternion
                q2 = self.states[i+1].pose.quaternion
                quaternion = quaternion_slerp(q1, q2, alpha)
                
                return Pose(timestamp=timestamp, position=position, quaternion=quaternion)
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "frame_id": self.frame_id,
            "states": [s.to_dict() for s in self.states]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trajectory':
        """Create from dictionary."""
        return cls(
            frame_id=data.get("frame_id", "world"),
            states=[TrajectoryState.from_dict(s) for s in data["states"]]
        )


# ============================================================================
# Landmark/Feature Data Structures
# ============================================================================

@dataclass
class Landmark:
    """3D landmark/feature point."""
    id: int  # Unique identifier
    position: np.ndarray  # 3x1 position in world frame
    descriptor: Optional[np.ndarray] = None  # Feature descriptor
    covariance: Optional[np.ndarray] = None  # 3x3 position covariance
    
    def __post_init__(self):
        """Validate dimensions."""
        self.position = np.asarray(self.position).flatten()
        
        if len(self.position) != 3:
            raise ValueError(f"Position must be 3D, got {len(self.position)}")
        
        if self.covariance is not None:
            self.covariance = np.asarray(self.covariance).reshape(3, 3)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "id": self.id,
            "position": self.position.tolist()
        }
        
        if self.descriptor is not None:
            result["descriptor"] = self.descriptor.tolist()
        if self.covariance is not None:
            result["covariance"] = self.covariance.tolist()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Landmark':
        """Create from dictionary."""
        descriptor = None
        covariance = None
        
        if "descriptor" in data and data["descriptor"] is not None:
            descriptor = np.array(data["descriptor"])
        if "covariance" in data and data["covariance"] is not None:
            covariance = np.array(data["covariance"])
        
        return cls(
            id=data["id"],
            position=np.array(data["position"]),
            descriptor=descriptor,
            covariance=covariance
        )


@dataclass
class Map:
    """Collection of landmarks forming a map."""
    landmarks: Dict[int, Landmark] = field(default_factory=dict)
    frame_id: str = "world"
    
    def add_landmark(self, landmark: Landmark):
        """Add or update a landmark."""
        self.landmarks[landmark.id] = landmark
    
    def get_landmark(self, landmark_id: int) -> Optional[Landmark]:
        """Get landmark by ID."""
        return self.landmarks.get(landmark_id)
    
    def get_positions(self) -> np.ndarray:
        """Get all landmark positions as Nx3 array."""
        if not self.landmarks:
            return np.empty((0, 3))
        return np.vstack([lm.position for lm in self.landmarks.values()])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "frame_id": self.frame_id,
            "landmarks": [lm.to_dict() for lm in self.landmarks.values()]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Map':
        """Create from dictionary."""
        map_obj = cls(frame_id=data.get("frame_id", "world"))
        for lm_data in data["landmarks"]:
            landmark = Landmark.from_dict(lm_data)
            map_obj.add_landmark(landmark)
        return map_obj


# ============================================================================
# Calibration Data Structures
# ============================================================================

class CameraModel(str, Enum):
    """Camera projection models."""
    PINHOLE = "pinhole"
    PINHOLE_RADTAN = "pinhole-radtan"
    KANNALA_BRANDT = "kannala-brandt"


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    model: CameraModel
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: np.ndarray  # Distortion coefficients
    
    def __post_init__(self):
        """Validate parameters."""
        self.distortion = np.asarray(self.distortion).flatten()
    
    def project(self, point_3d: np.ndarray) -> Optional[np.ndarray]:
        """Project 3D point to image plane."""
        # Simple pinhole projection (distortion not implemented yet)
        if point_3d[2] <= 0:
            return None
        
        u = self.fx * point_3d[0] / point_3d[2] + self.cx
        v = self.fy * point_3d[1] / point_3d[2] + self.cy
        
        # Check if within image bounds
        if 0 <= u < self.width and 0 <= v < self.height:
            return np.array([u, v])
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model.value,
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "distortion": self.distortion.tolist()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraIntrinsics':
        """Create from dictionary."""
        return cls(
            model=CameraModel(data["model"]),
            width=data["width"],
            height=data["height"],
            fx=data["fx"],
            fy=data["fy"],
            cx=data["cx"],
            cy=data["cy"],
            distortion=np.array(data["distortion"])
        )


@dataclass
class CameraExtrinsics:
    """Camera extrinsic parameters (transformation from another frame)."""
    B_T_C: np.ndarray  # 4x4 transformation from camera to body/reference frame
    
    def __post_init__(self):
        """Validate transformation matrix."""
        self.B_T_C = np.asarray(self.B_T_C).reshape(4, 4)
        
        # Check if valid transformation matrix
        if not np.allclose(self.B_T_C[3, :], [0, 0, 0, 1]):
            raise ValueError("Invalid transformation matrix")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "B_T_C": self.B_T_C.tolist()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraExtrinsics':
        """Create from dictionary."""
        return cls(B_T_C=np.array(data["B_T_C"]))


@dataclass
class CameraCalibration:
    """Complete camera calibration."""
    camera_id: str
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "camera_id": self.camera_id,
            "intrinsics": self.intrinsics.to_dict(),
            "extrinsics": self.extrinsics.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraCalibration':
        """Create from dictionary."""
        return cls(
            camera_id=data["camera_id"],
            intrinsics=CameraIntrinsics.from_dict(data["intrinsics"]),
            extrinsics=CameraExtrinsics.from_dict(data["extrinsics"])
        )


@dataclass
class IMUCalibration:
    """IMU calibration parameters."""
    imu_id: str
    accelerometer_noise_density: float
    accelerometer_random_walk: float
    gyroscope_noise_density: float
    gyroscope_random_walk: float
    rate: float = 200.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "imu_id": self.imu_id,
            "accelerometer_noise_density": self.accelerometer_noise_density,
            "accelerometer_random_walk": self.accelerometer_random_walk,
            "gyroscope_noise_density": self.gyroscope_noise_density,
            "gyroscope_random_walk": self.gyroscope_random_walk,
            "rate": self.rate
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IMUCalibration':
        """Create from dictionary."""
        return cls(**data)