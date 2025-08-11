# Technical Specifications for SLAM Simulation System

## 1. Sensor Specifications

### 1.1 IMU Specifications
Based on common MEMS IMUs (similar to BMI055/ICM-20649):

**Accelerometer:**
- Sampling Rate: 200 Hz (configurable: 100-1000 Hz)
- Measurement Range: ±16 g
- Noise Density: 0.00018 m/s²/√Hz (180 μg/√Hz)
- Random Walk: 0.001 m/s²√s
- Bias Stability: 0.0001 m/s² (0.01 mg)
- Bias Initial: Random ±0.1 m/s²

**Gyroscope:**
- Sampling Rate: 200 Hz (matches accelerometer)
- Measurement Range: ±2000 deg/s
- Noise Density: 0.00026 rad/s/√Hz (0.015 deg/s/√Hz)
- Random Walk: 0.0001 rad/s√s
- Bias Stability: 0.0001 rad/s (0.0057 deg/s)
- Bias Initial: Random ±0.01 rad/s

### 1.2 Camera Specifications
Based on standard VGA/HD cameras:

**Primary Configuration:**
- Resolution: 640x480 (VGA) or 1280x720 (HD)
- Frame Rate: 30 FPS
- Field of View: 90° horizontal (typical for webcams)
- Shutter Type: Global shutter (simplified, no rolling shutter)

**Camera Model:**
- Projection: Pinhole + Radial-Tangential distortion
- Distortion Coefficients: [k1, k2, p1, p2, k3]
- Typical values: k1=-0.28, k2=0.07, p1=0.0001, p2=0.0001, k3=0.0

### 1.3 Multi-Sensor Configuration
- Stereo Camera: Baseline 12cm (similar to Intel RealSense)
- Camera-IMU: Rigid transformation with temporal offset < 1ms
- Time Synchronization: Hardware triggered, sub-millisecond accuracy

## 2. Coordinate Systems

### 2.1 Frame Definitions
Following ROS REP-103 and computer vision conventions:

**World Frame (W):**
- ENU (East-North-Up) or NED (North-East-Down)
- Fixed inertial frame, Z-axis aligned with gravity

**Body/IMU Frame (B):**
- Origin: IMU center
- X: Forward, Y: Left, Z: Up (FLU convention)
- Device origin = First IMU position

**Camera Frame (C):**
- Origin: Optical center
- Z: Forward (optical axis), X: Right, Y: Down
- Standard computer vision convention

### 2.2 Transformations
```
T_BC: Body to Camera transformation (extrinsic calibration)
T_WB: World to Body transformation (pose to estimate)
T_WC = T_WB * T_BC: World to Camera transformation
```

## 3. Data Formats

### 3.1 Simulation Output JSON Schema
```json
{
  "metadata": {
    "version": "1.0",
    "timestamp": "ISO-8601",
    "coordinate_system": "ENU",
    "units": {
      "position": "meters",
      "rotation": "quaternion_wxyz",
      "time": "seconds"
    }
  },
  "calibration": {
    "cameras": [{
      "id": "cam0",
      "model": "pinhole-radtan",
      "width": 640,
      "height": 480,
      "intrinsics": {
        "fx": 458.654,
        "fy": 457.296,
        "cx": 367.215,
        "cy": 248.375
      },
      "distortion": [-0.28, 0.07, 0.0001, 0.0001, 0.0],
      "T_BC": {
        "translation": [0.065, 0.0, 0.0],
        "quaternion": [0.0, 0.0, 0.0, 1.0]
      }
    }],
    "imus": [{
      "id": "imu0",
      "accelerometer": {
        "noise_density": 0.00018,
        "random_walk": 0.001
      },
      "gyroscope": {
        "noise_density": 0.00026,
        "random_walk": 0.0001
      },
      "sampling_rate": 200
    }]
  },
  "groundtruth": {
    "trajectory": [{
      "timestamp": 0.0,
      "position": [0.0, 0.0, 0.0],
      "quaternion": [1.0, 0.0, 0.0, 0.0],
      "velocity": [0.0, 0.0, 0.0],
      "angular_velocity": [0.0, 0.0, 0.0]
    }],
    "landmarks": [{
      "id": 0,
      "position": [1.0, 0.0, 0.5],
      "descriptor": null
    }]
  },
  "measurements": {
    "imu": [{
      "timestamp": 0.0,
      "accelerometer": [0.0, 0.0, 9.81],
      "gyroscope": [0.0, 0.0, 0.0]
    }],
    "camera_frames": [{
      "timestamp": 0.0,
      "camera_id": "cam0",
      "observations": [{
        "landmark_id": 0,
        "pixel": [320.5, 240.2],
        "descriptor": null
      }]
    }]
  }
}
```

### 3.2 SLAM KPIs JSON Schema
```json
{
  "run_id": "swba_20240311_143022",
  "algorithm": "sliding_window_ba",
  "dataset": "simulation_circle",
  "timestamp": "2024-03-11T14:30:22Z",
  "metrics": {
    "trajectory_error": {
      "ate_rmse": 0.023,  // Absolute Trajectory Error (m)
      "ate_mean": 0.018,
      "ate_std": 0.014,
      "rpe_rmse": 0.012,  // Relative Pose Error (m)
      "rpe_mean": 0.009,
      "rpe_std": 0.008
    },
    "rotation_error": {
      "are_rmse": 0.015,  // Absolute Rotation Error (rad)
      "rre_rmse": 0.008   // Relative Rotation Error (rad)
    },
    "landmark_error": {
      "mean_error": 0.045,
      "std_error": 0.032,
      "num_landmarks": 500
    },
    "computational": {
      "total_time": 12.34,
      "avg_iteration_time": 0.023,
      "peak_memory_mb": 256
    },
    "convergence": {
      "iterations": 534,
      "final_cost": 0.0023,
      "converged": true
    }
  }
}
```

## 4. Simulation Scenarios

### 4.1 Trajectory Types
```python
trajectory_types = {
    "circle": {"radius": 2.0, "height": 1.5, "period": 10.0},
    "figure8": {"width": 4.0, "height": 2.0, "period": 15.0},
    "spiral": {"radius": 2.0, "pitch": 0.5, "turns": 3},
    "line": {"length": 10.0, "velocity": 0.5},
    "random_walk": {"bounds": [5, 5, 2], "step_size": 0.1}
}
```

### 4.2 Feature Generation
- Point cloud density: 500-2000 features
- Distribution: Uniform random in bounding box
- Visibility: Frustum culling + max range (10m)
- Minimum parallax: 1 pixel between frames

## 5. Estimator Specifications

### 5.1 State Vector
For Visual-Inertial estimation:
```
x = [p_WB, q_WB, v_WB, b_a, b_g, features]
```
- p_WB: Position (3x1)
- q_WB: Orientation quaternion (4x1)
- v_WB: Velocity (3x1)
- b_a: Accelerometer bias (3x1)
- b_g: Gyroscope bias (3x1)
- features: 3D landmark positions (3xN)

### 5.2 Sliding Window Bundle Adjustment (SWBA)
- Window size: 10 keyframes
- Marginalization: Schur complement
- Feature parametrization: Inverse depth or XYZ
- Robust cost: Huber loss (δ=1.0)
- Optimizer: Levenberg-Marquardt or Gauss-Newton

### 5.3 Extended Kalman Filter (EKF)
- Prediction: IMU integration at 200Hz
- Update: Camera measurements at 30Hz
- State covariance initialization:
  - Position: 0.1m std
  - Orientation: 5° std
  - Velocity: 0.1 m/s std
  - Biases: From sensor specs

### 5.4 Square Root Information Filter (SRIF)
- Information matrix: R^T R form (Cholesky)
- QR updates for numerical stability
- Identical state vector as EKF
- Better numerical properties for embedded systems

## 6. Evaluation Metrics

### 6.1 Trajectory Metrics (from TUM/EuRoC standards)
- **ATE (Absolute Trajectory Error)**: RMSE after SE3 alignment
- **RPE (Relative Pose Error)**: Error over fixed time intervals
- **Drift Rate**: Error growth per meter traveled

### 6.2 Estimation Quality
- **NEES (Normalized Estimation Error Squared)**: χ² test for consistency
- **Mahalanobis Distance**: For outlier detection
- **Covariance Realism**: Actual vs estimated uncertainty

### 6.3 Computational Metrics
- Processing time per frame
- Memory usage
- Convergence rate
- Number of iterations

## 7. Configuration Files

### 7.0 Pydantic Configuration Models
All configuration files must be validated using Pydantic models for type safety and validation:

```python
from pydantic import BaseModel, Field, validator
from typing import List, Literal, Optional
from enum import Enum

class TrajectoryType(str, Enum):
    CIRCLE = "circle"
    FIGURE8 = "figure8"
    SPIRAL = "spiral"
    LINE = "line"
    RANDOM_WALK = "random_walk"

class NoiseModel(str, Enum):
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    LOW_NOISE = "low_noise"

class EstimatorType(str, Enum):
    SWBA = "sliding_window_ba"
    EKF = "ekf"
    SRIF = "srif"

class IMUNoiseParams(BaseModel):
    accelerometer_noise_density: float = Field(0.00018, gt=0)
    accelerometer_random_walk: float = Field(0.001, gt=0)
    gyroscope_noise_density: float = Field(0.00026, gt=0)
    gyroscope_random_walk: float = Field(0.0001, gt=0)

class CameraIntrinsics(BaseModel):
    fx: float = Field(..., gt=0)
    fy: float = Field(..., gt=0)
    cx: float = Field(..., gt=0)
    cy: float = Field(..., gt=0)
    distortion: List[float] = Field(default_factory=lambda: [0, 0, 0, 0, 0])
    
    @validator('distortion')
    def validate_distortion_length(cls, v):
        if len(v) != 5:
            raise ValueError('Distortion must have exactly 5 coefficients')
        return v

class SimulationConfig(BaseModel):
    trajectory_type: TrajectoryType
    trajectory_params: dict
    imu_rate: int = Field(200, ge=50, le=1000)
    camera_rate: int = Field(30, ge=10, le=60)
    camera_resolution: tuple[int, int] = (640, 480)
    num_landmarks: int = Field(1000, ge=100, le=10000)
    noise_model: NoiseModel = NoiseModel.STANDARD
    seed: Optional[int] = None

class EstimatorConfig(BaseModel):
    type: EstimatorType
    window_size: Optional[int] = Field(10, ge=5, le=20)  # For SWBA
    max_iterations: int = Field(10, ge=1, le=100)
    convergence_threshold: float = Field(1e-6, gt=0, le=0.01)
    chi2_threshold: float = Field(5.991, gt=0)  # 95% confidence
    
    @validator('window_size')
    def window_size_for_swba_only(cls, v, values):
        if 'type' in values and values['type'] != EstimatorType.SWBA and v is not None:
            raise ValueError('window_size is only valid for SWBA')
        return v

# Usage example:
# config = SimulationConfig.parse_file("config/simulation.yaml")
# config.dict()  # Export to dict
# config.json()  # Export to JSON
```

Benefits of Pydantic:
- **Type validation** at runtime
- **Default values** with validation
- **Custom validators** for complex constraints
- **Automatic schema generation** for documentation
- **Serialization/deserialization** to/from JSON/YAML
- **IDE support** with type hints

### 7.1 Simulation Config (YAML)
```yaml
simulation:
  trajectory:
    type: circle
    params:
      radius: 2.0
      height: 1.5
      duration: 20.0
  
  sensors:
    imu:
      rate: 200
      noise_model: "standard"  # or "aggressive", "low_noise"
    camera:
      rate: 30
      resolution: [640, 480]
      noise_std: 1.0  # pixels
  
  environment:
    num_landmarks: 1000
    landmark_range: [10, 10, 5]
    seed: 42
```

### 7.2 Estimator Config
```yaml
estimator:
  type: "sliding_window_ba"  # or "ekf", "srif"
  
  swba:
    window_size: 10
    max_iterations: 10
    convergence_threshold: 1e-6
    robust_kernel: "huber"
    
  ekf:
    chi2_threshold: 5.991  # 95% confidence
    innovation_threshold: 3.0
    
  srif:
    qr_threshold: 1e-10
```

## 8. Implementation Notes

### 8.1 Simplifications for Estimator Evaluation
Since focus is on estimator comparison, we can simplify:
- No feature detection/matching (use ground truth associations)
- No loop closure detection
- No map management/culling
- Perfect data association
- No outlier measurements (unless testing robustness)

### 8.2 Essential Components
- IMU preintegration (for efficiency)
- Camera projection/unprojection
- SE3 operations and manifold optimization
- Covariance propagation
- Error metric computation

### 8.3 Data Generation Pipeline
1. Generate smooth trajectory (splines/polynomials)
2. Sample IMU measurements with noise
3. Generate 3D landmarks
4. Project to cameras with noise
5. Export to JSON format
6. Provide to estimators
7. Compare with ground truth