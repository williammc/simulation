"""
Configuration models using Pydantic for type safety and validation.
"""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

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
    SWBA = "swba"
    EKF = "ekf"
    SRIF = "srif"
    CPP_BINARY = "cpp_binary"
    UNKNOWN = "unknown"


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


class KeyframeSelectionStrategy(str, Enum):
    """Keyframe selection strategies."""
    FIXED_INTERVAL = "fixed_interval"
    MOTION_BASED = "motion_based"
    HYBRID = "hybrid"


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


class KeyframeSelectionConfig(BaseModel):
    """Configuration for keyframe selection strategies."""
    
    strategy: KeyframeSelectionStrategy = Field(
        default=KeyframeSelectionStrategy.FIXED_INTERVAL,
        description="Keyframe selection strategy"
    )
    
    # Fixed interval parameters
    fixed_interval: int = Field(
        default=10,
        ge=1,
        description="Select every N-th frame as keyframe"
    )
    min_time_gap: float = Field(
        default=0.1,
        gt=0,
        description="Minimum time gap between keyframes (seconds)"
    )
    
    # Motion-based parameters
    translation_threshold: float = Field(
        default=0.5,
        gt=0,
        description="Translation threshold for keyframe selection (meters)"
    )
    rotation_threshold: float = Field(
        default=0.3,
        gt=0,
        description="Rotation threshold for keyframe selection (radians)"
    )
    
    # Hybrid parameters
    max_interval: int = Field(
        default=20,
        ge=1,
        description="Maximum frames between keyframes in hybrid mode"
    )
    force_keyframe_on_motion: bool = Field(
        default=True,
        description="Force keyframe when motion thresholds are exceeded"
    )
    
    @model_validator(mode='after')
    def validate_strategy_params(self):
        """Validate parameters based on selected strategy."""
        if self.strategy == KeyframeSelectionStrategy.FIXED_INTERVAL:
            if self.fixed_interval <= 0:
                raise ValueError("fixed_interval must be positive")
        elif self.strategy == KeyframeSelectionStrategy.MOTION_BASED:
            if self.translation_threshold <= 0 and self.rotation_threshold <= 0:
                raise ValueError("At least one motion threshold must be positive")
        elif self.strategy == KeyframeSelectionStrategy.HYBRID:
            if self.max_interval < self.fixed_interval:
                raise ValueError("max_interval must be >= fixed_interval")
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
    
    # Preintegration settings
    enable_preintegration: bool = Field(
        default=False,
        description="Enable IMU preintegration between keyframes"
    )
    
    # Keyframe selection configuration
    keyframe_selection: KeyframeSelectionConfig = Field(
        default_factory=KeyframeSelectionConfig,
        description="Keyframe selection configuration"
    )
    
    @model_validator(mode='after')
    def validate_sensors(self):
        """Ensure at least one sensor is configured."""
        if not self.cameras and not self.imus:
            raise ValueError('At least one camera or IMU must be configured')
        return self


class BaseEstimatorConfig(BaseModel):
    """Base configuration for all SLAM estimators."""
    # Common fields for all estimators
    estimator_type: Optional[EstimatorType] = Field(
        default=None,
        description="Type of estimator (set in subclasses)"
    )
    max_landmarks: int = Field(1000, ge=1, description="Maximum number of landmarks")
    verbose: bool = Field(False, description="Enable verbose logging")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    
    # Common measurement noise
    pixel_noise_std: float = Field(1.0, gt=0, description="Pixel measurement noise (pixels)")
    
    # Common outlier rejection
    chi2_threshold: float = Field(
        5.991,
        gt=0,
        description="Chi-squared test threshold (95% confidence for 2 DOF)"
    )


class SWBAConfig(BaseEstimatorConfig):
    """Sliding Window Bundle Adjustment configuration."""
    # Set default estimator type
    estimator_type: EstimatorType = Field(
        default=EstimatorType.SWBA,
        description="Type of estimator"
    )
    
    # Keyframe-only processing
    use_keyframes_only: bool = Field(
        True,  # SWBA typically uses keyframes only
        description="Process only keyframes (ignore non-keyframe measurements)"
    )
    
    # Window parameters
    window_size: int = Field(10, ge=3, le=30, description="Number of keyframes in window")
    keyframe_translation_threshold: float = Field(0.5, gt=0, description="Translation threshold for keyframe (m)")
    keyframe_rotation_threshold: float = Field(0.3, gt=0, description="Rotation threshold for keyframe (rad)")
    keyframe_time_threshold: float = Field(0.5, gt=0, description="Time threshold for keyframe (s)")
    
    # Optimization parameters
    max_iterations: int = Field(20, ge=1, le=100, description="Maximum optimization iterations")
    convergence_threshold: float = Field(
        1e-6,
        gt=0,
        le=0.01,
        description="Convergence threshold for cost reduction"
    )
    lambda_init: float = Field(1e-4, gt=0, description="Initial Levenberg-Marquardt damping")
    lambda_factor: float = Field(10.0, gt=1, description="LM damping adjustment factor")
    lambda_min: float = Field(1e-8, gt=0, description="Minimum LM damping")
    lambda_max: float = Field(1e8, gt=0, description="Maximum LM damping")
    
    # Robust cost parameters
    robust_kernel: str = Field("huber", description="Robust cost function type")
    huber_threshold: float = Field(1.0, gt=0, description="Huber kernel threshold")
    huber_delta: float = Field(1.0, gt=0, description="Huber kernel threshold (alias)")
    
    # IMU parameters
    use_preintegrated_imu: bool = Field(True, description="Use preintegrated IMU measurements")
    imu_weight: float = Field(1.0, gt=0, description="IMU measurement weight")
    
    # Camera parameters
    camera_weight: float = Field(1.0, gt=0, description="Camera measurement weight")
    min_observations_per_landmark: int = Field(2, ge=2, description="Minimum observations per landmark")
    
    # Marginalization
    marginalize_old_keyframes: bool = Field(True, description="Marginalize old keyframes")
    prior_weight: float = Field(1.0, gt=0, description="Prior factor weight")


class EKFConfig(BaseEstimatorConfig):
    """Extended Kalman Filter configuration."""
    # Set default estimator type
    estimator_type: EstimatorType = Field(
        default=EstimatorType.EKF,
        description="Type of estimator"
    )
    
    # Preintegrated IMU support
    use_preintegrated_imu: bool = Field(False, description="Use preintegrated IMU measurements")
    
    # Keyframe-only processing
    use_keyframes_only: bool = Field(
        False,
        description="Process only keyframes (ignore non-keyframe measurements)"
    )
    
    # EKF-specific outlier rejection
    innovation_threshold: float = Field(
        3.0,
        gt=0,
        description="Innovation outlier threshold (sigma multiplier)"
    )
    max_iterations: int = Field(5, ge=1, description="Max iterations for outlier rejection")
    
    # Initial uncertainties
    initial_position_std: float = Field(0.1, gt=0, description="Initial position uncertainty (m)")
    initial_orientation_std: float = Field(0.01, gt=0, description="Initial orientation uncertainty (rad)")
    initial_velocity_std: float = Field(0.1, gt=0, description="Initial velocity uncertainty (m/s)")
    initial_accel_bias_std: float = Field(0.01, gt=0, description="Initial accelerometer bias uncertainty (m/s²)")
    initial_gyro_bias_std: float = Field(0.001, gt=0, description="Initial gyroscope bias uncertainty (rad/s)")
    
    # Process noise
    accel_noise_density: float = Field(0.01, gt=0, description="Accelerometer noise density (m/s²/√Hz)")
    gyro_noise_density: float = Field(0.001, gt=0, description="Gyroscope noise density (rad/s/√Hz)")
    accel_bias_random_walk: float = Field(0.001, gt=0, description="Accelerometer bias random walk (m/s³/√Hz)")
    gyro_bias_random_walk: float = Field(0.0001, gt=0, description="Gyroscope bias random walk (rad/s²/√Hz)")
    
    
    # Integration
    integration_method: str = Field("euler", description="IMU integration method (euler, rk4, midpoint)")
    gravity_magnitude: float = Field(9.81, gt=0, description="Gravity magnitude (m/s²)")


class SRIFConfig(BaseEstimatorConfig):
    """Square Root Information Filter configuration."""
    # Set default estimator type
    estimator_type: EstimatorType = Field(
        default=EstimatorType.SRIF,
        description="Type of estimator"
    )
    
    # Preintegrated IMU support
    use_preintegrated_imu: bool = Field(False, description="Use preintegrated IMU measurements")
    
    # Keyframe-only processing
    use_keyframes_only: bool = Field(
        False,
        description="Process only keyframes (ignore non-keyframe measurements)"
    )
    
    # Numerical parameters
    qr_threshold: float = Field(
        1e-10,
        gt=0,
        description="QR decomposition threshold for numerical stability"
    )
    adaptive_threshold: bool = Field(True, description="Use adaptive thresholding")
    
    # Initial uncertainties
    initial_position_std: float = Field(0.1, gt=0, description="Initial position uncertainty (m)")
    initial_velocity_std: float = Field(0.1, gt=0, description="Initial velocity uncertainty (m/s)")
    initial_orientation_std: float = Field(0.01, gt=0, description="Initial orientation uncertainty (rad)")
    initial_bias_std: float = Field(0.01, gt=0, description="Initial bias uncertainty")
    initial_information_scale: float = Field(
        10.0,
        gt=0,
        description="Scale factor for initial information matrix"
    )
    
    # Process noise
    accel_noise_std: float = Field(0.1, gt=0, description="Accelerometer noise standard deviation (m/s²)")
    gyro_noise_std: float = Field(0.01, gt=0, description="Gyroscope noise standard deviation (rad/s)")
    accel_bias_noise_std: float = Field(0.001, gt=0, description="Accelerometer bias noise (m/s²)")
    gyro_bias_noise_std: float = Field(0.0001, gt=0, description="Gyroscope bias noise (rad/s)")
    
    
    # IMU integration
    integration_method: str = Field("rk4", description="IMU integration method (euler, rk4, midpoint)")


class CppBinaryConfig(BaseEstimatorConfig):
    """C++ Binary Estimator configuration."""
    # Set default estimator type
    estimator_type: EstimatorType = Field(
        default=EstimatorType.CPP_BINARY,
        description="Type of estimator"
    )
    
    # Binary execution parameters
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for binary execution"
    )
    
    # Required parameters with defaults
    executable: str = Field(
        "cpp_estimation/build/estimator",
        description="Path to the executable binary"
    )
    timeout: int = Field(300, gt=0, description="Execution timeout in seconds")
    input_file: str = Field("simulation_data.json", description="Input JSON filename")
    output_file: str = Field("estimation_result.json", description="Output JSON filename")
    
    # Optional parameters
    working_dir: Optional[str] = Field(None, description="Working directory for execution")
    args: List[str] = Field(default_factory=list, description="Command line arguments")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    retry_on_failure: bool = Field(False, description="Retry on failure")
    max_retries: int = Field(1, ge=0, description="Maximum number of retries")


class EstimatorConfig(BaseModel):
    """General estimator configuration."""
    type: EstimatorType = Field(
        default=EstimatorType.EKF,
        description="Estimator algorithm type"
    )
    swba: Optional[SWBAConfig] = None
    ekf: Optional[EKFConfig] = None
    srif: Optional[SRIFConfig] = None
    cpp_binary: Optional[CppBinaryConfig] = None
    
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
        elif self.type == EstimatorType.CPP_BINARY and self.cpp_binary is None:
            self.cpp_binary = CppBinaryConfig()
        return self


def load_simulation_config(path: Union[str, Path]) -> SimulationConfig:
    """Load simulation configuration from YAML file."""
    # Try to use ConfigLoader if available, fallback to simple loading
    try:
        from ..utils.config_loader import ConfigLoader
        loader = ConfigLoader()
        data = loader.load_config(path)
    except (ImportError, Exception):
        # Fallback to simple YAML loading for backward compatibility
        path = Path(path)
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
    return SimulationConfig(**data)


def load_estimator_config(path: Union[str, Path]) -> EstimatorConfig:
    """Load estimator configuration from YAML file."""
    # Try to use ConfigLoader if available, fallback to simple loading
    try:
        from ..utils.config_loader import ConfigLoader
        loader = ConfigLoader()
        data = loader.load_config(path)
    except (ImportError, Exception):
        # Fallback to simple YAML loading for backward compatibility
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