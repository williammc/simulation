# IMU and Preintegration Documentation

## Overview

This document provides comprehensive documentation about IMU (Inertial Measurement Unit) modeling, simulation, and preintegration as used in the SLAM estimation system, particularly in the EKF and GTSAM-EKF estimators.

## Table of Contents
- [IMU Fundamentals](#imu-fundamentals)
- [IMU Simulation](#imu-simulation)
- [IMU Preintegration](#imu-preintegration)
- [GTSAM Integration](#gtsam-integration)
- [Usage in Estimators](#usage-in-estimators)
- [Common Issues and Solutions](#common-issues-and-solutions)

## IMU Fundamentals

### What is an IMU?

An IMU measures:
- **Accelerometer**: Linear acceleration (m/s²) in 3 axes
- **Gyroscope**: Angular velocity (rad/s) in 3 axes

### Specific Force

IMUs measure **specific force**, not true acceleration:
```
specific_force = acceleration - gravity
```

This is crucial for understanding IMU integration:
- The IMU feels gravity as an upward force when stationary
- Integration must account for gravity to get correct position

### IMU Noise Model

Our IMU model includes several noise sources:

```python
# From src/sensor_models/imu_model.py
class IMUCalibration:
    accelerometer_noise_density: float  # m/s²/√Hz
    gyroscope_noise_density: float      # rad/s/√Hz
    accelerometer_random_walk: float    # m/s³/√Hz
    gyroscope_random_walk: float        # rad/s²/√Hz
```

## IMU Simulation

### Generating IMU Measurements

The IMU simulator (`src/simulation/imu_simulator.py`) generates realistic measurements:

```python
def simulate_imu_measurement(true_state, imu_calib):
    # Get true acceleration and angular velocity
    true_accel = compute_acceleration(true_state)
    true_gyro = true_state.angular_velocity
    
    # Add gravity to get specific force
    specific_force = true_accel - gravity_in_body_frame
    
    # Add noise
    accel_noise = np.random.normal(0, imu_calib.accelerometer_noise_density)
    gyro_noise = np.random.normal(0, imu_calib.gyroscope_noise_density)
    
    return IMUMeasurement(
        accelerometer=specific_force + accel_noise,
        gyroscope=true_gyro + gyro_noise
    )
```

### Bias Modeling

IMU biases evolve as random walks:

```python
# Bias evolution
accel_bias += np.random.normal(0, accel_bias_rw * sqrt(dt))
gyro_bias += np.random.normal(0, gyro_bias_rw * sqrt(dt))
```

## IMU Preintegration

### Why Preintegration?

Preintegration combines multiple IMU measurements between keyframes into a single relative motion constraint:

1. **Computational Efficiency**: Avoids re-integrating measurements during optimization
2. **Linearization Point**: Preintegrated values don't depend on absolute pose
3. **Covariance Propagation**: Properly accounts for measurement uncertainty

### Preintegration Mathematics

Between two keyframes at times $t_i$ and $t_j$:

```
Δp_ij = ∫∫ R(t) * (a(t) - b_a) dt²
Δv_ij = ∫ R(t) * (a(t) - b_a) dt  
ΔR_ij = ∏ exp(ω(t) - b_g) dt
```

Where:
- `Δp_ij`: Relative position change
- `Δv_ij`: Relative velocity change
- `ΔR_ij`: Relative rotation change
- `b_a, b_g`: Accelerometer and gyroscope biases

### Implementation

Our preintegration implementation (`src/estimation/gtsam_imu_preintegration.py`):

```python
class GTSAMPreintegration:
    def __init__(self, params: GTSAMPreintegrationParams):
        # Create GTSAM preintegration with gravity compensation
        self.gtsam_params = gtsam.PreintegrationCombinedParams.MakeSharedU(
            params.gravity_magnitude
        )
        self.pim = gtsam.PreintegratedCombinedMeasurements(
            self.gtsam_params, initial_bias
        )
    
    def add_measurement(self, imu: IMUMeasurement, dt: float):
        # GTSAM handles gravity compensation internally
        self.pim.integrateMeasurement(
            imu.accelerometer,  # Specific force
            imu.gyroscope,
            dt
        )
    
    def get_preintegrated_data(self) -> PreintegratedIMUData:
        return PreintegratedIMUData(
            delta_position=self.pim.deltaPij(),
            delta_velocity=self.pim.deltaVij(),
            delta_rotation=self.pim.deltaRij().matrix(),
            covariance=self.pim.preintMeasCov()
        )
```

## GTSAM Integration

### CombinedImuFactor

GTSAM's `CombinedImuFactor` connects consecutive poses using preintegrated IMU:

```python
# Create factor between poses i and j
factor = gtsam.CombinedImuFactor(
    X(i), V(i),  # Pose and velocity at time i
    X(j), V(j),  # Pose and velocity at time j
    B(i), B(j),  # IMU biases at both times
    preintegrated_measurements
)
```

### Gravity Handling

GTSAM automatically handles gravity compensation:
1. **Input**: Specific force (what IMU measures)
2. **Internal**: Adds gravity during integration
3. **Output**: Position/velocity in world frame

### Bias Estimation

The `CombinedImuFactor` jointly estimates:
- Robot trajectory (poses)
- Velocities at each keyframe
- IMU biases (slowly varying)

## Usage in Estimators

### EKF Estimator

The standard EKF (`src/estimation/ekf_slam.py`) uses a simplified IMU model:

```python
def predict_with_imu(self, imu_measurements):
    # Simple integration without preintegration
    for imu in imu_measurements:
        dt = compute_dt(imu)
        
        # Update orientation
        self.state.R = self.state.R @ exp_so3(imu.gyroscope * dt)
        
        # Update velocity (with gravity)
        accel_world = self.state.R @ imu.accelerometer + gravity
        self.state.v += accel_world * dt
        
        # Update position
        self.state.p += self.state.v * dt
```

### GTSAM-EKF Estimator

The GTSAM-EKF (`src/estimation/gtsam_ekf_estimator.py`) uses full preintegration:

```python
class GTSAMEKFEstimatorV2:
    def predict_with_imu(self, imu_measurements, to_timestamp):
        # Create preintegration
        preintegration = GTSAMPreintegration(self.params)
        preintegration.reset(self.current_bias)
        
        # Add all measurements
        for imu in imu_measurements:
            preintegration.add_measurement(imu, dt)
        
        # Create CombinedImuFactor
        factor = preintegration.create_combined_imu_factor(
            pose_keys, velocity_keys, bias_keys
        )
        
        # Add to graph and optimize
        self.graph.add(factor)
        self.isam2.update(self.graph, initial_values)
```

### Key Differences

| Feature | EKF | GTSAM-EKF |
|---------|-----|-----------|
| Preintegration | No | Yes |
| Bias Estimation | Simple | Full random walk |
| Gravity Handling | Manual | Automatic |
| Optimization | Kalman Filter | iSAM2 |
| Computational Cost | O(n) | O(1) per keyframe |

## Common Issues and Solutions

### 1. Gravity Direction

**Issue**: Incorrect gravity direction causes drift

**Solution**: GTSAM assumes gravity along negative Z:
```python
# Correct: gravity = [0, 0, -9.81]
params = gtsam.PreintegrationParams.MakeSharedU(9.81)
```

### 2. Specific Force vs Acceleration

**Issue**: Confusing specific force with acceleration

**Solution**: Remember IMU measures:
```python
# What IMU measures (specific force)
imu_measurement = acceleration - gravity

# To get acceleration
acceleration = imu_measurement + gravity
```

### 3. Coordinate Frames

**Issue**: Mixing body and world frames

**Solution**: Track frames carefully:
```python
# IMU measures in body frame
accel_body = imu.accelerometer

# Convert to world frame
accel_world = R_world_body @ accel_body

# Add gravity (in world frame)
accel_world += gravity_world
```

### 4. Integration Drift

**Issue**: Position drifts without vision updates

**Solution**: This is expected! Pure IMU integration always drifts:
- Use vision/GPS for periodic corrections
- Estimate and remove biases
- Higher quality IMU reduces but doesn't eliminate drift

### 5. Numerical Issues

**Issue**: Covariance becomes non-positive definite

**Solution**: Add small regularization:
```python
# Add integration noise
params.setIntegrationCovariance(1e-8 * np.eye(3))
```

## Testing and Validation

### Unit Tests

Key test files:
- `tests/test_imu_simulation.py`: IMU measurement generation
- `tests/test_preintegration.py`: Preintegration correctness
- `tests/test_gtsam_ekf.py`: GTSAM integration
- `tests/test_imu_integration.py`: End-to-end IMU integration

### Validation Against GTSAM

We validate our implementation against GTSAM's reference:
```python
# tests/test_preintegration_vs_gtsam.py
def test_against_gtsam():
    # Create identical measurements
    measurements = generate_test_trajectory()
    
    # Our implementation
    our_result = our_preintegration(measurements)
    
    # GTSAM's implementation
    gtsam_result = gtsam_preintegration(measurements)
    
    # Should match within numerical precision
    assert np.allclose(our_result, gtsam_result, rtol=1e-6)
```

## Performance Considerations

### Preintegration Frequency

- **Keyframe Rate**: Typically 1-10 Hz
- **IMU Rate**: Typically 100-1000 Hz
- **Preintegration**: Combines 10-1000 measurements per keyframe

### Memory Usage

```python
# Per keyframe storage
PreintegratedIMUData:
    delta_position: 3 floats
    delta_velocity: 3 floats  
    delta_rotation: 9 floats
    covariance: 225 floats (15x15)
    Total: ~1 KB per keyframe
```

### Computational Complexity

- **Without Preintegration**: O(n*m) where n=poses, m=measurements
- **With Preintegration**: O(n) where n=poses
- **iSAM2 Update**: O(affected variables) ≈ O(1) for local changes

## References

1. **Forster et al.**: "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry"
2. **GTSAM Documentation**: PreintegratedCombinedMeasurements class
3. **Quaternion Kinematics**: Trawny & Roumeliotis technical report
4. **IMU Noise Models**: Woodman, "An introduction to inertial navigation"