# Estimation Module Documentation

## Overview

The estimation module (`src/estimation/`) implements various SLAM (Simultaneous Localization and Mapping) algorithms for state estimation from sensor measurements. It provides multiple estimator implementations with different trade-offs between accuracy, computational cost, and implementation complexity.

## Table of Contents
- [Architecture](#architecture)
- [Available Estimators](#available-estimators)
- [Base Estimator Interface](#base-estimator-interface)
- [EKF SLAM](#ekf-slam)
- [GTSAM-Based Estimators](#gtsam-based-estimators)
- [Square Root Information Filter](#square-root-information-filter)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Performance Comparison](#performance-comparison)

## Architecture

```
src/estimation/
├── __init__.py
├── base_estimator.py         # Abstract base class
├── ekf_slam.py              # Extended Kalman Filter SLAM
├── gtsam_ekf_estimator.py   # GTSAM-based EKF with IMU preintegration
├── gtsam_swba_estimator.py  # GTSAM Sliding Window Bundle Adjustment
├── srif_slam.py             # Square Root Information Filter
├── gtsam_imu_preintegration.py  # IMU preintegration utilities
└── factor_graph_utils.py    # Factor graph construction helpers
```

## Available Estimators

| Estimator | Type | Features | Use Case |
|-----------|------|----------|----------|
| EKF-SLAM | Filter | Fast, simple | Real-time, small-scale |
| GTSAM-EKF | Hybrid | IMU preintegration, incremental | Real-time, IMU-centric |
| GTSAM-SWBA | Optimization | Sliding window, bundle adjustment | High accuracy, batch |
| SRIF-SLAM | Filter | Numerically stable, square root form | Medium-scale, stable |

## Base Estimator Interface

All estimators inherit from `BaseEstimator`:

```python
class BaseEstimator(ABC):
    @abstractmethod
    def initialize(self, initial_pose: Pose) -> None:
        """Initialize estimator with initial pose."""
        pass
    
    @abstractmethod
    def predict(self, imu_data) -> None:
        """Predict next state using IMU data."""
        pass
    
    @abstractmethod
    def update(self, frame: CameraFrame, landmarks: Map) -> None:
        """Update state with camera observations."""
        pass
    
    @abstractmethod
    def get_result(self) -> EstimatorResult:
        """Get current estimation result."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset estimator to initial state."""
        pass
```

## EKF SLAM

### Overview

The Extended Kalman Filter SLAM (`ekf_slam.py`) implements the classic EKF-SLAM algorithm with joint state-map estimation.

### State Representation

```python
# State vector: [position, orientation, velocity, landmarks...]
state = [
    rx, ry, rz,           # Robot position
    qw, qx, qy, qz,       # Robot orientation (quaternion)
    vx, vy, vz,           # Robot velocity
    l1x, l1y, l1z,        # Landmark 1 position
    l2x, l2y, l2z,        # Landmark 2 position
    ...
]

# Covariance matrix
P = [
    [P_robot,    P_robot_map],
    [P_map_robot, P_map]
]
```

### Implementation

```python
class EKFSlam(BaseEstimator):
    def __init__(self, config: EstimatorConfig):
        self.state = np.zeros(10)  # Initial robot state
        self.P = np.eye(10) * config.initial_covariance
        self.landmarks = {}
        
    def predict(self, imu_measurements: List[IMUMeasurement]):
        """EKF prediction step with IMU."""
        for imu in imu_measurements:
            # Predict state
            self.state = self.motion_model(self.state, imu, dt)
            
            # Predict covariance
            F = self.compute_jacobian_F(self.state, imu)
            Q = self.process_noise(dt)
            self.P = F @ self.P @ F.T + Q
    
    def update(self, observations: List[CameraObservation]):
        """EKF update step with camera."""
        for obs in observations:
            # Compute innovation
            z_pred = self.observation_model(self.state, landmark)
            innovation = obs.pixel - z_pred
            
            # Compute Kalman gain
            H = self.compute_jacobian_H(self.state, landmark)
            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S)
            
            # Update state and covariance
            self.state += K @ innovation
            self.P = (np.eye(len(self.state)) - K @ H) @ self.P
```

### Features

- **Joint state-map estimation**: Robot pose and landmarks in single state
- **Landmark initialization**: Inverse depth or bearing-only
- **Outlier rejection**: Chi-squared test on innovations
- **State augmentation**: Dynamic landmark addition

## GTSAM-Based Estimators

### GTSAM-EKF Estimator

The GTSAM-EKF (`gtsam_ekf_estimator.py`) combines GTSAM's factor graphs with incremental optimization.

#### Key Features

- **IMU Preintegration**: Efficient IMU handling with `CombinedImuFactor`
- **Incremental Optimization**: iSAM2 for real-time performance
- **Bias Estimation**: Joint estimation of IMU biases
- **Gravity Handling**: Automatic gravity compensation

#### Implementation

```python
class GTSAMEKFEstimatorV2(BaseEstimator):
    def __init__(self, config):
        # Initialize iSAM2
        isam2_params = gtsam.ISAM2Params()
        isam2_params.relinearizeThreshold = 0.01
        self.isam2 = gtsam.ISAM2(isam2_params)
        
        # Factor graph and values
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_values = gtsam.Values()
        
        # IMU preintegration
        self.preintegration_params = self.create_imu_params(config)
        
    def predict_with_imu(self, imu_measurements, to_timestamp):
        """Predict using IMU with preintegration."""
        # Create preintegration
        pim = GTSAMPreintegration(self.preintegration_params)
        
        # Integrate measurements
        for imu in imu_measurements:
            pim.add_measurement(imu, dt)
        
        # Create CombinedImuFactor
        factor = gtsam.CombinedImuFactor(
            X(i), V(i),  # Previous pose and velocity
            X(j), V(j),  # Current pose and velocity
            B(i), B(j),  # Biases
            pim.get_preintegrated_measurements()
        )
        
        self.graph.add(factor)
        self.isam2.update(self.graph, self.initial_values)
```

### GTSAM Sliding Window BA

The GTSAM-SWBA (`gtsam_swba_estimator.py`) implements sliding window bundle adjustment.

#### Features

- **Fixed window size**: Maintains computational bounds
- **Marginalization**: Proper marginalization of old states
- **Bundle adjustment**: Joint optimization over window
- **Smart factors**: Efficient projection factors

#### Implementation

```python
class GtsamSWBAEstimator(BaseEstimator):
    def __init__(self, config):
        self.window_size = config.window_size
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.smoother = gtsam.BatchFixedLagSmoother(window_size)
        
    def update_window(self):
        """Maintain sliding window."""
        if len(self.poses) > self.window_size:
            # Marginalize oldest pose
            self.marginalize_old_states()
            
        # Optimize within window
        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self.graph, self.values
        )
        self.values = optimizer.optimize()
    
    def add_visual_factors(self, observations):
        """Add smart projection factors."""
        for landmark_id, obs_list in observations.items():
            factor = gtsam.SmartProjectionPoseFactor(self.camera_model)
            
            for obs in obs_list:
                factor.add(obs.pixel, X(obs.pose_idx))
            
            self.graph.add(factor)
```

## Square Root Information Filter

The SRIF (`srif_slam.py`) provides numerically stable filtering.

### Key Concepts

- **Square root form**: Maintains R where RTR = information matrix
- **QR updates**: Numerically stable updates via QR decomposition
- **Givens rotations**: Efficient sparse updates

### Implementation

```python
class SRIFSlam(BaseEstimator):
    def __init__(self, config):
        # Information matrix in square root form
        self.R = np.eye(state_dim) * sqrt(initial_information)
        self.d = np.zeros(state_dim)  # Information vector
        
    def update(self, measurement, jacobian):
        """SRIF update using QR decomposition."""
        # Stack measurement
        augmented = np.vstack([
            self.R,
            sqrt(measurement_info) * jacobian
        ])
        
        # QR decomposition
        Q, R_new = np.linalg.qr(augmented)
        
        # Extract updated square root information
        self.R = R_new[:state_dim, :state_dim]
        
        # Solve for state
        self.state = np.linalg.solve(self.R, self.d)
```

## Configuration

### Estimator Configuration

```python
@dataclass
class EstimatorConfig:
    # Common parameters
    estimator_type: str = 'ekf'
    max_landmarks: int = 1000
    max_iterations: int = 100
    convergence_threshold: float = 1e-6
    
    # EKF specific
    initial_covariance: float = 0.1
    process_noise_position: float = 0.01
    process_noise_orientation: float = 0.001
    measurement_noise_camera: float = 1.0
    
    # GTSAM specific
    relinearize_threshold: float = 0.01
    relinearize_skip: int = 1
    
    # SWBA specific
    window_size: int = 10
    marginalization_threshold: float = 0.1
    
    # IMU parameters
    imu_rate: float = 200.0
    gravity_magnitude: float = 9.81
    accel_noise_density: float = 0.01
    gyro_noise_density: float = 0.001
```

### Loading Configuration

```yaml
# config/estimators/gtsam_ekf.yaml
estimator:
  type: gtsam-ekf
  max_landmarks: 500
  
  imu:
    rate: 200.0
    gravity: 9.81
    accel_noise: 0.01
    gyro_noise: 0.001
    
  optimization:
    relinearize_threshold: 0.01
    max_iterations: 10
```

## Usage Examples

### Basic Usage

```python
from src.estimation import create_estimator, EstimatorConfig

# Create estimator
config = EstimatorConfig(estimator_type='gtsam-ekf')
estimator = create_estimator(config)

# Initialize
initial_pose = Pose(position=[0, 0, 0])
estimator.initialize(initial_pose)

# Process measurements
for keyframe in keyframes:
    # Predict with IMU
    estimator.predict(keyframe.imu_measurements)
    
    # Update with camera
    estimator.update(keyframe.camera_frame, landmarks)

# Get result
result = estimator.get_result()
trajectory = result.trajectory
```

### Comparative Evaluation

```python
# Compare multiple estimators
estimators = {
    'ekf': create_estimator(EstimatorConfig(estimator_type='ekf')),
    'gtsam_ekf': create_estimator(EstimatorConfig(estimator_type='gtsam-ekf')),
    'swba': create_estimator(EstimatorConfig(estimator_type='gtsam-swba'))
}

results = {}
for name, estimator in estimators.items():
    estimator.initialize(initial_pose)
    
    for data in sensor_data:
        estimator.predict(data.imu)
        estimator.update(data.camera, landmarks)
    
    results[name] = estimator.get_result()

# Compare accuracy
for name, result in results.items():
    error = compute_trajectory_error(result.trajectory, ground_truth)
    print(f"{name}: RMSE = {error.rmse:.3f}m")
```

### Real-time Processing

```python
class RealtimeEstimator:
    def __init__(self, config):
        self.estimator = create_estimator(config)
        self.imu_buffer = []
        self.keyframe_selector = KeyframeSelector()
        
    def process_imu(self, imu: IMUMeasurement):
        """Process high-rate IMU."""
        self.imu_buffer.append(imu)
        
    def process_camera(self, frame: CameraFrame):
        """Process camera frame."""
        if self.keyframe_selector.is_keyframe(frame):
            # Predict with accumulated IMU
            self.estimator.predict(self.imu_buffer)
            self.imu_buffer.clear()
            
            # Update with camera
            self.estimator.update(frame, self.landmarks)
            
            # Get current pose
            result = self.estimator.get_result()
            return result.current_pose
```

## Performance Comparison

### Computational Complexity

| Estimator | Prediction | Update | Memory |
|-----------|-----------|---------|---------|
| EKF | O(n²) | O(n²m) | O(n²) |
| GTSAM-EKF | O(1) | O(k³) | O(nk) |
| SWBA | O(w³) | O(w³m) | O(w²) |
| SRIF | O(n²) | O(n²m) | O(n²) |

Where:
- n: Total state dimension
- m: Number of measurements
- k: Number of affected variables
- w: Window size

### Accuracy Comparison

Typical performance on circular trajectory (5m radius):

| Estimator | Position RMSE | Orientation RMSE | Runtime |
|-----------|--------------|------------------|---------|
| EKF | 0.15m | 2.5° | 50ms |
| GTSAM-EKF | 0.08m | 1.2° | 30ms |
| SWBA | 0.05m | 0.8° | 150ms |
| SRIF | 0.12m | 2.0° | 80ms |

## Advanced Features

### Multi-Robot SLAM

```python
class MultiRobotEstimator:
    def __init__(self, num_robots):
        self.estimators = [
            create_estimator(config) for _ in range(num_robots)
        ]
        self.shared_landmarks = Map()
        
    def process_robot_data(self, robot_id, data):
        # Local estimation
        self.estimators[robot_id].process(data)
        
        # Share landmark observations
        if self.is_communication_available():
            self.exchange_landmarks()
```

### Adaptive Configuration

```python
class AdaptiveEstimator:
    def __init__(self):
        self.estimator = create_estimator(initial_config)
        self.performance_monitor = PerformanceMonitor()
        
    def adapt_parameters(self):
        """Adapt parameters based on performance."""
        metrics = self.performance_monitor.get_metrics()
        
        if metrics.innovation_ratio > threshold:
            # Increase process noise
            self.config.process_noise *= 1.5
            
        if metrics.computation_time > time_budget:
            # Reduce window size or landmarks
            self.config.window_size = max(5, self.config.window_size - 1)
```

## Troubleshooting

### Common Issues

1. **Divergence**
   - Check initial covariance
   - Verify measurement associations
   - Increase process noise

2. **Slow Performance**
   - Reduce landmark count
   - Decrease window size
   - Use incremental methods

3. **Numerical Instability**
   - Switch to SRIF
   - Check conditioning
   - Add regularization

4. **IMU Integration Drift**
   - Estimate biases
   - Add vision updates
   - Check gravity alignment

## Best Practices

1. **Initialization**
   - Use accurate initial pose
   - Set conservative initial covariance
   - Initialize gravity direction properly

2. **Outlier Rejection**
   - Use robust cost functions
   - Implement chi-squared tests
   - Monitor innovation statistics

3. **Computational Efficiency**
   - Limit active landmarks
   - Use sliding windows
   - Employ incremental methods

4. **Debugging**
   - Log innovation statistics
   - Visualize covariance ellipses
   - Check Jacobian correctness

## References

- EKF-SLAM: "Simultaneous Localization and Mapping: Part I" (Durrant-Whyte & Bailey)
- GTSAM: "Factor Graphs and GTSAM" (Dellaert)
- IMU Preintegration: "On-Manifold Preintegration" (Forster et al.)
- SRIF: "Factorization Methods for Discrete Sequential Estimation" (Bierman)