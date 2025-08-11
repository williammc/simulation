"""
Configuration models using Pydantic for type safety and validation.
"""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class TrajectoryType(str, Enum):
    """Available trajectory types for simulation."""
    CIRCLE = "circle"
    FIGURE8 = "figure8"
    SPIRAL = "spiral"
    LINE = "line"
    RANDOM_WALK = "random_walk"


class NoiseModel(str, Enum):
    """Predefined noise models for sensors."""
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    LOW_NOISE = "low_noise"


class EstimatorType(str, Enum):
    """Available estimator algorithms."""
    SWBA = "sliding_window_ba"
    EKF = "ekf"
    SRIF = "srif"


class CameraModel(str, Enum):
    """Camera projection models."""
    PINHOLE = "pinhole"
    PINHOLE_RADTAN = "pinhole-radtan"
    KANNALA_BRANDT = "kannala-brandt"


class CoordinateSystem(str, Enum):
    """Coordinate system conventions."""
    ENU = "ENU"  # East-North-Up
    NED = "NED"  # North-East-Down
    FLU = "FLU"  # Forward-Left-Up
    FRD = "FRD"  # Forward-Right-Down


class IMUNoiseParams(BaseModel):
    """IMU noise parameters."""
    accelerometer_noise_density: float = Field(
        default=0.00018,
        gt=0,
        description="Accelerometer noise density (m/s²/√Hz)"
    )
    accelerometer_random_walk: float = Field(
        default=0.001,
        gt=0,
        description="Accelerometer random walk (m/s²√s)"
    )
    accelerometer_bias_stability: float = Field(
        default=0.0001,
        gt=0,
        description="Accelerometer bias stability (m/s²)"
    )
    gyroscope_noise_density: float = Field(
        default=0.00026,
        gt=0,
        description="Gyroscope noise density (rad/s/√Hz)"
    )
    gyroscope_random_walk: float = Field(
        default=0.0001,
        gt=0,
        description="Gyroscope random walk (rad/s√s)"
    )
    gyroscope_bias_stability: float = Field(
        default=0.0001,
        gt=0,
        description="Gyroscope bias stability (rad/s)"
    )


class CameraIntrinsics(BaseModel):
    """Camera intrinsic parameters."""
    fx: float = Field(..., gt=0, description="Focal length in x (pixels)")
    fy: float = Field(..., gt=0, description="Focal length in y (pixels)")
    cx: float = Field(..., gt=0, description="Principal point x (pixels)")
    cy: float = Field(..., gt=0, description="Principal point y (pixels)")
    width: int = Field(640, gt=0, description="Image width (pixels)")
    height: int = Field(480, gt=0, description="Image height (pixels)")
    distortion: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0],
        description="Distortion coefficients [k1, k2, p1, p2, k3]"
    )
    
    @field_validator('distortion')
    @classmethod
    def validate_distortion_length(cls, v: List[float]) -> List[float]:
        if len(v) not in [4, 5, 8]:
            raise ValueError('Distortion must have 4, 5, or 8 coefficients')
        # Pad to 5 if needed
        if len(v) == 4:
            v.append(0.0)
        return v


class CameraExtrinsics(BaseModel):
    """Camera extrinsic parameters (transformation from body to camera)."""
    translation: List[float] = Field(
        default=[0.0, 0.0, 0.0],
        description="Translation [x, y, z] in meters"
    )
    quaternion: List[float] = Field(
        default=[1.0, 0.0, 0.0, 0.0],
        description="Rotation quaternion [w, x, y, z]"
    )
    
    @field_validator('translation')
    @classmethod
    def validate_translation(cls, v: List[float]) -> List[float]:
        if len(v) != 3:
            raise ValueError('Translation must have exactly 3 components')
        return v
    
    @field_validator('quaternion')
    @classmethod
    def validate_quaternion(cls, v: List[float]) -> List[float]:
        if len(v) != 4:
            raise ValueError('Quaternion must have exactly 4 components')
        # Normalize quaternion
        norm = sum(x**2 for x in v) ** 0.5
        if norm < 1e-6:
            raise ValueError('Quaternion norm is too small')
        return [x / norm for x in v]


class CameraConfig(BaseModel):
    """Complete camera configuration."""
    id: str = Field(default="cam0", description="Camera identifier")
    model: CameraModel = Field(
        default=CameraModel.PINHOLE_RADTAN,
        description="Camera projection model"
    )
    rate: float = Field(30.0, gt=0, le=120, description="Frame rate (Hz)")
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics = Field(default_factory=CameraExtrinsics)
    noise_std: float = Field(
        1.0,
        ge=0,
        description="Measurement noise standard deviation (pixels)"
    )


class IMUConfig(BaseModel):
    """IMU configuration."""
    id: str = Field(default="imu0", description="IMU identifier")
    rate: float = Field(200.0, ge=50, le=1000, description="Sampling rate (Hz)")
    noise_params: IMUNoiseParams = Field(default_factory=IMUNoiseParams)
    noise_model: NoiseModel = Field(
        default=NoiseModel.STANDARD,
        description="Predefined noise model"
    )
    
    @model_validator(mode='after')
    def apply_noise_model(self):
        """Apply predefined noise model parameters."""
        if self.noise_model == NoiseModel.LOW_NOISE:
            self.noise_params.accelerometer_noise_density = 0.00009
            self.noise_params.gyroscope_noise_density = 0.00013
        elif self.noise_model == NoiseModel.AGGRESSIVE:
            self.noise_params.accelerometer_noise_density = 0.00036
            self.noise_params.gyroscope_noise_density = 0.00052
        return self


class TrajectoryConfig(BaseModel):
    """Trajectory generation configuration."""
    type: TrajectoryType = Field(
        default=TrajectoryType.CIRCLE,
        description="Trajectory type"
    )
    duration: float = Field(20.0, gt=0, description="Duration (seconds)")
    params: Dict[str, float] = Field(
        default_factory=dict,
        description="Trajectory-specific parameters"
    )
    
    @model_validator(mode='after')
    def set_default_params(self):
        """Set default parameters based on trajectory type."""
        defaults = {
            TrajectoryType.CIRCLE: {"radius": 2.0, "height": 1.5, "angular_velocity": 0.5},
            TrajectoryType.FIGURE8: {"width": 4.0, "height": 2.0, "period": 15.0},
            TrajectoryType.SPIRAL: {"radius": 2.0, "pitch": 0.5, "turns": 3},
            TrajectoryType.LINE: {"length": 10.0, "velocity": 0.5},
            TrajectoryType.RANDOM_WALK: {"bounds_x": 5.0, "bounds_y": 5.0, "bounds_z": 2.0, "step_size": 0.1}
        }
        
        if self.type in defaults and not self.params:
            self.params = defaults[self.type]
        return self


class EnvironmentConfig(BaseModel):
    """Environment and landmark configuration."""
    num_landmarks: int = Field(
        1000,
        ge=100,
        le=10000,
        description="Number of 3D landmarks"
    )
    landmark_range: List[float] = Field(
        default=[10.0, 10.0, 5.0],
        description="Landmark distribution range [x, y, z] in meters"
    )
    min_distance: float = Field(
        0.5,
        gt=0,
        description="Minimum distance to landmarks (meters)"
    )
    max_distance: float = Field(
        20.0,
        gt=0,
        description="Maximum visible distance (meters)"
    )
    
    @field_validator('landmark_range')
    @classmethod
    def validate_range(cls, v: List[float]) -> List[float]:
        if len(v) != 3:
            raise ValueError('Landmark range must have exactly 3 components')
        if any(x <= 0 for x in v):
            raise ValueError('All range components must be positive')
        return v


class SimulationConfig(BaseModel):
    """Complete simulation configuration."""
    trajectory: TrajectoryConfig = Field(default_factory=TrajectoryConfig)
    cameras: List[CameraConfig] = Field(
        default_factory=list,
        description="Camera configurations"
    )
    imus: List[IMUConfig] = Field(
        default_factory=list,
        description="IMU configurations"
    )
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    coordinate_system: CoordinateSystem = Field(
        default=CoordinateSystem.ENU,
        description="World coordinate system"
    )
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    
    @model_validator(mode='after')
    def validate_sensors(self):
        """Ensure at least one sensor is configured."""
        if not self.cameras and not self.imus:
            raise ValueError('At least one camera or IMU must be configured')
        return self


class SWBAConfig(BaseModel):
    """Sliding Window Bundle Adjustment configuration."""
    window_size: int = Field(10, ge=3, le=30, description="Number of keyframes in window")
    max_iterations: int = Field(10, ge=1, le=100, description="Maximum optimization iterations")
    convergence_threshold: float = Field(
        1e-6,
        gt=0,
        le=0.01,
        description="Convergence threshold for cost reduction"
    )
    robust_kernel: Optional[str] = Field(
        "huber",
        description="Robust cost function (huber, cauchy, or None)"
    )
    huber_delta: float = Field(1.0, gt=0, description="Huber kernel threshold")


class EKFConfig(BaseModel):
    """Extended Kalman Filter configuration."""
    chi2_threshold: float = Field(
        5.991,
        gt=0,
        description="Chi-squared test threshold (95% confidence for 2 DOF)"
    )
    innovation_threshold: float = Field(
        3.0,
        gt=0,
        description="Innovation outlier threshold (sigma multiplier)"
    )
    initial_position_std: float = Field(0.1, gt=0, description="Initial position uncertainty (m)")
    initial_orientation_std: float = Field(0.087, gt=0, description="Initial orientation uncertainty (rad, ~5°)")
    initial_velocity_std: float = Field(0.1, gt=0, description="Initial velocity uncertainty (m/s)")
    initial_bias_std: float = Field(0.01, gt=0, description="Initial bias uncertainty")


class SRIFConfig(BaseModel):
    """Square Root Information Filter configuration."""
    qr_threshold: float = Field(
        1e-10,
        gt=0,
        description="QR decomposition threshold for numerical stability"
    )
    chi2_threshold: float = Field(
        5.991,
        gt=0,
        description="Chi-squared test threshold"
    )
    initial_information_scale: float = Field(
        10.0,
        gt=0,
        description="Scale factor for initial information matrix"
    )


class EstimatorConfig(BaseModel):
    """General estimator configuration."""
    type: EstimatorType = Field(
        default=EstimatorType.EKF,
        description="Estimator algorithm type"
    )
    swba: Optional[SWBAConfig] = None
    ekf: Optional[EKFConfig] = None
    srif: Optional[SRIFConfig] = None
    
    output_rate: float = Field(
        100.0,
        gt=0,
        description="Output rate for estimated trajectory (Hz)"
    )
    
    @model_validator(mode='after')
    def ensure_config_matches_type(self):
        """Ensure the appropriate config is provided for the estimator type."""
        if self.type == EstimatorType.SWBA and self.swba is None:
            self.swba = SWBAConfig()
        elif self.type == EstimatorType.EKF and self.ekf is None:
            self.ekf = EKFConfig()
        elif self.type == EstimatorType.SRIF and self.srif is None:
            self.srif = SRIFConfig()
        return self


def load_simulation_config(path: Union[str, Path]) -> SimulationConfig:
    """Load simulation configuration from YAML file."""
    path = Path(path)
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return SimulationConfig(**data)


def load_estimator_config(path: Union[str, Path]) -> EstimatorConfig:
    """Load estimator configuration from YAML file."""
    path = Path(path)
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return EstimatorConfig(**data)


def save_config(config: BaseModel, path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict and handle enums
    data = config.model_dump(mode='json')
    
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)