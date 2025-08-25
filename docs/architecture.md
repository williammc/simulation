# SLAM Simulation System Architecture

## Table of Contents
1. [System Overview](#system-overview)
2. [Core Architecture](#core-architecture)
3. [Module Organization](#module-organization)
4. [Data Flow](#data-flow)
5. [Component Details](#component-details)
6. [Design Patterns](#design-patterns)
7. [External Interfaces](#external-interfaces)
8. [Performance Considerations](#performance-considerations)

## System Overview

The SLAM Simulation System is a comprehensive framework for evaluating Visual-Inertial SLAM algorithms through both synthetic and real-world datasets. The system follows a modular architecture that separates concerns into distinct layers: simulation, estimation, evaluation, and visualization.

### Key Design Principles
- **Modularity**: Each component has a single, well-defined responsibility
- **Extensibility**: New estimators and datasets can be added without modifying core code
- **Reproducibility**: All simulations and evaluations are deterministic with fixed seeds
- **Interoperability**: Common JSON format for data exchange between components
- **Parallelization**: Support for concurrent execution of independent tasks

## Core Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI Interface                         │
│                     (tools/cli.py)                          │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Orchestration Layer                       │
│              (src/evaluation/orchestrator.py)               │
└─────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  Simulation  │      │  Estimation  │      │  Evaluation  │
│    Layer     │      │    Layer     │      │    Layer     │
└──────────────┘      └──────────────┘      └──────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer (JSON)                       │
│                   (src/common/json_io.py)                   │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Visualization Layer                       │
│                  (src/plotting/, Plotly)                    │
└─────────────────────────────────────────────────────────────┘
```

## Module Organization

### Directory Structure
```
slam_simulation/
├── src/                      # Core source code
│   ├── common/              # Shared data structures and utilities
│   │   ├── data_structures.py # Core data structures
│   │   ├── config.py       # Configuration classes
│   │   ├── json_io.py      # JSON serialization/deserialization
│   │   └── transforms.py   # SE(3) transformations
│   │
│   ├── simulation/          # Data generation
│   │   ├── simulator.py    # Main simulation orchestrator
│   │   ├── trajectory_generator.py # Trajectory generators
│   │   ├── imu_simulator.py # IMU sensor simulation
│   │   ├── camera_simulator.py # Camera simulation
│   │   ├── landmark_generator.py # 3D landmark generation
│   │   └── sensor_sync.py  # Multi-sensor synchronization
│   │
│   ├── estimation/          # SLAM algorithms
│   │   ├── base_estimator.py # Abstract estimator interface
│   │   ├── ekf_slam.py     # Extended Kalman Filter
│   │   ├── gtsam_ekf_estimator.py # GTSAM-based EKF with IMU preintegration
│   │   ├── gtsam_swba_estimator.py # GTSAM Sliding Window Bundle Adjustment
│   │   ├── srif_slam.py    # Square Root Information Filter
│   │   ├── gtsam_imu_preintegration.py # IMU preintegration utilities
│   │   └── factory.py      # Estimator factory
│   │
│   ├── evaluation/          # Metrics and comparison
│   │   ├── trajectory_evaluator.py # Trajectory metrics (ATE, RPE)
│   │   ├── landmark_evaluator.py # Map quality assessment
│   │   ├── metrics.py      # Core metric implementations
│   │   ├── statistical_analysis.py # Statistical tools
│   │   └── benchmark.py    # Benchmarking framework
│   │
│   ├── plotting/            # Visualization
│   │   ├── trajectory_plotter.py # 3D trajectory visualization
│   │   ├── sensor_plotter.py # IMU/camera data plots
│   │   ├── error_plotter.py # Error metrics visualization
│   │   ├── dashboard_generator.py # Multi-panel dashboards
│   │   └── interactive_viewer.py # Real-time visualization
│   │
│   ├── utils/               # Utilities
│   │   ├── math_utils.py   # Mathematical operations
│   │   ├── config_loader.py # Configuration loading
│   │   └── gtsam_integration_utils.py # GTSAM conversion utilities
│   │
│   └── io/                  # Input/Output
│       ├── result_io.py    # Result serialization
│       └── dataset_loader.py # Dataset loading utilities
│
├── tools/                   # CLI and scripts
│   ├── cli.py              # Main command-line interface
│   ├── download.py         # Dataset downloader
│   └── inspect_data.py     # Data inspection utilities
│
├── config/                  # Configuration files
│   ├── simulation.yaml     # Simulation parameters
│   ├── estimators.yaml     # Estimator configurations
│   └── evaluation_config.yaml # Evaluation pipeline config
│
├── tests/                   # Unit and integration tests
│   ├── test_*.py           # Test modules
│   └── fixtures/           # Test data
│
└── data/                    # Dataset storage
    ├── TUM-VIE/            # TUM Visual-Inertial-Event datasets
    └── simulated/          # Generated synthetic data
```

## Data Flow

### 1. Simulation Pipeline
```
Trajectory Generator → Landmark Generator → Sensor Simulator → Noise Addition → JSON Export
      ↓                      ↓                    ↓                 ↓
   (poses)              (3D points)        (measurements)    (corrupted)
```

### 2. Estimation Pipeline
```
JSON Data → Preprocessor → Estimator → State Propagation → Map Update → Result Export
     ↓           ↓            ↓              ↓                ↓             ↓
  (input)   (validated)  (algorithm)    (prediction)     (correction)   (output)
```

### 3. Evaluation Pipeline
```
Ground Truth + Estimates → Alignment → Metric Computation → Statistical Analysis → Dashboard
        ↓                     ↓              ↓                     ↓                   ↓
   (reference)           (registered)   (ATE/RPE/NEES)        (p-values)          (HTML)
```

## Component Details

### Common Layer (`src/common/`)

#### Data Structures
- **SimulationData**: Container for all simulation data
  - Metadata (timestamps, configuration)
  - Ground truth (trajectory, landmarks)
  - Measurements (IMU, camera observations)
  - Calibration (sensor intrinsics/extrinsics)

- **State Representation**:
  ```python
  State = {
      'position': np.ndarray(3),      # [x, y, z]
      'orientation': np.ndarray(4),    # quaternion [w, x, y, z]
      'velocity': np.ndarray(3),       # [vx, vy, vz]
      'imu_bias': {
          'accelerometer': np.ndarray(3),
          'gyroscope': np.ndarray(3)
      }
  }
  ```

#### Coordinate Conventions
- **World Frame (W)**: ENU (East-North-Up) or NED configurable
- **Body Frame (B)**: FLU (Forward-Left-Up), IMU center
- **Camera Frame (C)**: Optical frame, Z-forward
- **Transformations**: `A_T_B` transforms FROM B TO A

### Simulation Layer (`src/simulation/`)

#### Trajectory Generators
- **Circle**: Circular motion in horizontal plane
- **Figure-8**: Lemniscate trajectory
- **Spiral**: Ascending/descending helix
- **Line**: Linear motion
- **Random Walk**: Brownian motion with bounds

#### Sensor Models

**IMU Model**:
```python
# Accelerometer measurement
a_measured = R_WB.T @ (a_true - g) + bias_a + noise_a
# Gyroscope measurement  
w_measured = w_true + bias_w + noise_w
```

**Camera Model**:
```python
# Pinhole projection with radial-tangential distortion
# 1. Transform to camera frame
P_C = T_CW @ P_W
# 2. Project to normalized plane
[u_n, v_n] = [X/Z, Y/Z]
# 3. Apply distortion
[u_d, v_d] = distort(u_n, v_n, k1, k2, p1, p2, k3)
# 4. Apply intrinsics
[u, v] = K @ [u_d, v_d, 1]
```

#### Noise Models
- **White Noise**: Gaussian with specified σ
- **Random Walk**: Integrated white noise
- **Bias Drift**: Exponentially correlated noise
- **Outliers**: Configurable percentage with uniform distribution

### Estimation Layer (`src/estimation/`)

#### Estimator Interface
```python
class BaseEstimator(ABC):
    @abstractmethod
    def initialize(self, initial_pose: Pose) -> None:
        """Initialize estimator with initial pose"""
        
    @abstractmethod
    def predict(self, imu_data) -> None:
        """Predict next state using IMU data"""
        
    @abstractmethod
    def update(self, frame: CameraFrame, landmarks: Map) -> None:
        """Update state with camera observations"""
        
    @abstractmethod
    def get_result(self) -> EstimatorResult:
        """Get current estimation result"""
```

#### EKF-SLAM
- **State**: Robot pose + landmark positions
- **Prediction**: IMU integration with error propagation
- **Update**: Sequential measurement updates
- **Complexity**: O(n²) for n landmarks

#### GTSAM-EKF (IMU Preintegration)
- **Backend**: GTSAM with iSAM2 incremental optimization
- **IMU Handling**: CombinedImuFactor with preintegration
- **Bias Estimation**: Joint estimation of IMU biases
- **Gravity**: Automatic compensation
- **Complexity**: O(1) per keyframe update

#### GTSAM-SWBA (Sliding Window Bundle Adjustment)
- **Window Size**: Configurable (default 10 keyframes)
- **Optimization**: Levenberg-Marquardt with GTSAM
- **Marginalization**: Proper marginalization of old states
- **Smart Factors**: Efficient projection factors
- **Complexity**: O(w³) for w window size

#### Square Root Information Filter (SRIF)
- **Representation**: R^T R form of information matrix
- **Updates**: QR factorization for numerical stability
- **Advantages**: Better conditioning than EKF
- **Complexity**: O(n²) with better constants

### Evaluation Layer (`src/evaluation/`)

#### Metrics

**Absolute Trajectory Error (ATE)**:
```python
ATE = sqrt(mean(||p_est - p_gt||²))  # After SE(3) alignment
```

**Relative Pose Error (RPE)**:
```python
RPE = sqrt(mean(||log(T_rel_est @ T_rel_gt^-1)||²))
```

**Normalized Estimation Error Squared (NEES)**:
```python
NEES = (x_est - x_gt)^T @ Σ^-1 @ (x_est - x_gt)
# Should follow χ² distribution if consistent
```

#### Statistical Tests
- **Two-sample t-test**: Compare estimator means
- **F-test**: Compare estimator variances
- **χ² test**: Consistency checking
- **Wilcoxon signed-rank**: Non-parametric comparison

### Visualization Layer (`src/plotting/`)

#### Plot Types
1. **3D Trajectory**: Interactive 3D visualization with Plotly
2. **Error Evolution**: Time series of errors
3. **Consistency Plots**: NEES with χ² bounds
4. **Performance Matrix**: Heatmap of metrics
5. **Timing Analysis**: Runtime comparisons
6. **Memory Usage**: Resource consumption

#### Dashboard Components
```javascript
Dashboard = {
    overview: SummaryTable,
    trajectory: Plot3D,
    errors: TimeSeries,
    metrics: Heatmap,
    performance: BarChart,
    consistency: NEESPlot
}
```

## Design Patterns

### 1. Factory Pattern
Used for creating estimators and trajectory generators:
```python
def create_estimator(config: Dict) -> SLAMEstimator:
    if config['type'] == 'ekf':
        return EKFSlam(config)
    elif config['type'] == 'swba':
        return SlidingWindowBA(config)
    # ...
```

### 2. Strategy Pattern
Different noise models and optimization algorithms:
```python
class NoiseModel(ABC):
    @abstractmethod
    def add_noise(self, measurement: np.ndarray) -> np.ndarray:
        pass

class GaussianNoise(NoiseModel):
    def add_noise(self, measurement):
        return measurement + np.random.normal(0, self.sigma)
```

### 3. Observer Pattern
Event-driven updates for real-time visualization:
```python
class EstimatorObserver:
    def on_state_update(self, state: State):
        self.visualizer.update(state)
```

### 4. Builder Pattern
Complex simulation configuration:
```python
simulation = (SimulationBuilder()
    .with_trajectory('circle', radius=5.0)
    .with_landmarks(count=200)
    .with_imu(rate=200, noise_level=0.1)
    .with_camera(rate=30, fov=90)
    .build())
```

## External Interfaces

### Command-Line Interface
```bash
# Main commands
./run.sh simulate [trajectory] [options]
./run.sh slam [estimator] --input [file] [options]
./run.sh evaluate [config] [options]
./run.sh convert [dataset] [input] [output]
./run.sh download [dataset] [sequence]
```

### Configuration Files (YAML)
```yaml
simulation:
  trajectory:
    type: circle
    params:
      radius: 10.0
  sensors:
    imu:
      rate: 200
      noise: standard
    camera:
      rate: 30
      resolution: [640, 480]
```

### Data Format (JSON)
```json
{
  "metadata": {
    "version": "1.0",
    "timestamp": "2024-01-01T00:00:00Z"
  },
  "calibration": {...},
  "groundtruth": {
    "trajectory": [...],
    "landmarks": [...]
  },
  "measurements": {
    "imu": [...],
    "camera_frames": [...]
  }
}
```

### Python API
```python
from src.simulation import generate_trajectory
from src.estimation import EKFSlam
from src.evaluation import compute_ate

# Generate data
traj = generate_trajectory('circle', duration=20)
# Run SLAM
slam = EKFSlam(config)
result = slam.process(data)
# Evaluate
ate = compute_ate(result, groundtruth)
```

## Performance Considerations

### Computational Complexity
| Component | Complexity | Bottleneck |
|-----------|-----------|------------|
| EKF-SLAM | O(n²) | Covariance update |
| GTSAM-EKF | O(1) per keyframe | iSAM2 update |
| GTSAM-SWBA | O(w³) | Bundle adjustment (w=window) |
| SRIF | O(n²) | QR factorization |
| IMU Preintegration | O(m) | m measurements between keyframes |
| ATE/RPE | O(n) | Alignment step |

### Memory Requirements
- **EKF**: O(n²) for covariance matrix
- **GTSAM-EKF**: O(k) for k active variables in iSAM2
- **GTSAM-SWBA**: O(w×n) for window Jacobians
- **SRIF**: O(n²) for R matrix
- **Preintegration**: O(1) per IMU segment (consolidated)
- **Trajectory**: O(t) for t timesteps

### Optimization Strategies
1. **Parallelization**
   - Multi-estimator evaluation (ProcessPoolExecutor)
   - Parallel landmark observations
   - Batch matrix operations (NumPy vectorization)

2. **Caching**
   - Precomputed sensor calibrations
   - Trajectory interpolation tables
   - Jacobian reuse in optimization

3. **Sparse Operations**
   - Sparse information matrix in SRIF
   - Sparse Jacobians in SWBA
   - Selective landmark updates

### Scalability Limits
- **Landmarks**: ~10,000 for real-time EKF
- **Trajectory Length**: ~100,000 poses
- **Window Size**: ~20 keyframes for SWBA
- **Parallel Jobs**: CPU cores - 1

## Future Extensions

### Planned Features
1. **Additional Estimators**
   - Graph-SLAM
   - Particle filters
   - Learning-based methods

2. **Sensor Modalities**
   - LiDAR integration
   - Event cameras (full TUM-VIE support)
   - Multi-camera rigs

3. **Advanced Evaluation**
   - Loop closure metrics
   - Map quality assessment
   - Robustness analysis

4. **Real-time Capabilities**
   - ROS integration
   - Hardware-in-the-loop testing
   - Live visualization

### Extension Points
- Custom trajectory generators via plugin system
- User-defined noise models
- External optimizer integration
- Custom visualization backends

## References

### Architecture Inspirations
- **GTSAM**: Factor graph architecture
- **ORB-SLAM**: Keyframe-based structure
- **OKVIS**: Multi-IMU handling
- **Kalibr**: Calibration pipeline design

### Standards and Conventions
- **ROS REP-103**: Coordinate frames
- **ROS REP-105**: Transformations
- **TUM RGB-D**: Evaluation metrics
- **EuRoC MAV**: Dataset format

---

## Recent Updates

### Version 2.0.0 (January 2025)
- **GTSAM Integration**: Full support for GTSAM-based estimators
- **IMU Preintegration**: CombinedImuFactor with bias estimation
- **Improved Testing**: Comprehensive test suite with GTSAM validation
- **Documentation**: Complete module documentation in `docs/`
- **Performance**: O(1) keyframe updates with iSAM2

### Version 1.0.0 (December 2024)
- Initial release with EKF, SWBA, SRIF estimators
- Basic simulation and evaluation pipeline
- Interactive visualization tools

---

*Last Updated: January 2025*
*Version: 2.0.0*