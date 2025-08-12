# SLAM Simulation System

A comprehensive simulation system for evaluating SLAM estimators (EKF, Sliding Window BA, SRIF) against ground truth data.

## Phase 1 Complete ✓

Phase 1 features are fully implemented and tested.

## Phase 2 Complete ✓

### Features Implemented

#### 1. Project Structure
- ✓ Organized folder structure per requirements
- ✓ `run.sh` script for easy command execution
- ✓ Python package setup with `setup.py`

#### 2. Configuration System (Pydantic)
- ✓ Type-safe configuration models
- ✓ YAML loading/saving with validation
- ✓ Sample configs for simulation and estimators
- ✓ Automatic parameter validation and defaults

#### 3. Mathematical Utilities
- ✓ **SO3 Operations**: Rotation matrices, exponential/logarithm maps
- ✓ **SE3 Operations**: Transformation matrices, inverse, adjoint
- ✓ **Quaternion Operations**: Multiplication, SLERP, conversions
- ✓ **Frame Transformations**: Point/vector transforms, Euler angles

#### 4. CLI Tool (Typer)
- ✓ Commands: simulate, slam, dashboard, download
- ✓ Rich terminal output with colors and tables
- ✓ Help documentation and info display

#### 5. Testing
- ✓ Unit tests for configuration validation
- ✓ Unit tests for all math operations
- ✓ Numerical stability tests

## Quick Start

### Setup Environment
```bash
./run.sh setup
```

### Run Tests
```bash
./run.sh test
```

### View Help
```bash
./run.sh help
```

### CLI Commands
```bash
# Simulation (placeholder)
./run.sh simulate circle --duration 20

# SLAM Estimation (placeholder)
./run.sh slam ekf --input output/simulation.json

# Dashboard (placeholder)
./run.sh dashboard

# System Info
python -m tools.cli info
```

## Configuration

### Simulation Config
```yaml
trajectory:
  type: circle
  duration: 20.0
  params:
    radius: 2.0
    height: 1.5

cameras:
  - id: cam0
    model: pinhole-radtan
    rate: 30.0
    intrinsics:
      fx: 458.654
      # ...

imus:
  - id: imu0
    rate: 200.0
    noise_model: standard
```

### Estimator Config
```yaml
type: ekf  # or swba, srif

ekf:
  chi2_threshold: 5.991
  innovation_threshold: 3.0
  # ...
```

## Transformation Naming Convention

**IMPORTANT**: This project uses a consistent naming convention for all transformations to avoid confusion:

### Convention Rules
- **`param_a_R_b`**: Rotation matrix that transforms from frame `b` to frame `a`
- **`param_a_t_b`**: Translation vector from frame `b` to frame `a`
- **`param_a_T_b`**: Transformation matrix (SE3) from frame `b` to frame `a`

### Examples
```python
# Rotation from camera frame to world frame
W_R_C = world_from_camera_rotation

# Translation from body/IMU frame to camera frame
C_t_B = camera_from_body_translation

# Transformation from body frame to world frame
W_T_B = world_from_body_transform

# Chain transformations: W_T_C = W_T_B @ B_T_C
W_T_C = W_T_B @ B_T_C  # World from camera = World from body @ Body from camera

# Transform a point from camera frame to world frame
p_camera = np.array([1, 0, 0])  # Point in camera frame
p_world = transform_point(W_T_C, p_camera)  # Point in world frame
```

### Common Frame Abbreviations
- **W**: World/Map frame (fixed inertial reference)
- **B**: Body/IMU frame (vehicle center)
- **C**: Camera frame (optical center)
- **L**: Left camera
- **R**: Right camera
- **E**: Event camera

## Math Utilities API

```python
from src.utils.math_utils import *

# SO3 operations (using scipy.spatial.transform.Rotation)
W_R_B = so3_exp(omega)  # Axis-angle to rotation
omega = so3_log(W_R_B)  # Rotation to axis-angle

# SE3 operations
W_T_B = se3_exp(xi)     # Twist to transformation
xi = se3_log(W_T_B)     # Transformation to twist

# Quaternions (convention: [w, x, y, z])
q = quaternion_multiply(q1, q2)
q_interp = quaternion_slerp(q1, q2, t)
W_R_B = quaternion_to_rotation_matrix(q)

# Apply transformations
p_world = transform_point(W_T_C, p_camera)  # Transform point
v_world = transform_vector(W_R_C, v_camera)  # Rotate vector (no translation)

# Utility functions
R_random = random_rotation_matrix()  # Generate random rotation
q_random = random_quaternion()       # Generate random quaternion
R = rotation_matrix_from_vectors(v1, v2)  # Find R such that R @ v1 ≈ v2
```

#### 6. Data Structures (Phase 2)
- ✓ **IMU Data**: IMUMeasurement, IMUData with chronological ordering
- ✓ **Camera Data**: ImagePoint, CameraObservation, CameraFrame, CameraData
- ✓ **Trajectory**: Pose (SE3), TrajectoryState with velocities, Trajectory with interpolation
- ✓ **Landmarks**: Landmark with covariance, Map collection
- ✓ **Calibration**: Camera intrinsics/extrinsics, IMU calibration

#### 7. JSON I/O
- ✓ **SimulationData class**: Complete container matching JSON schema
- ✓ **Serialization**: to_dict/from_dict for all data structures
- ✓ **Save/Load functions**: Convenience functions for complete simulation data
- ✓ **JSON Schema compliance**: Follows technical specifications

#### 8. Trajectory Generators
- ✓ **Circle trajectory**: Constant angular velocity, configurable radius/height
- ✓ **Figure-8 trajectory**: Lemniscate parametrization
- ✓ **Spiral trajectory**: Expanding radius with vertical motion
- ✓ **Line trajectory**: Constant velocity motion
- ✓ **Velocity computation**: Analytical velocities and angular velocities
- ✓ **Uniform sampling**: Configurable rate (Hz) with exact timestamps

#### 9. TUM-VIE Dataset Reader
- ✓ **Calibration loading**: Support for calibration_a.json and calibration_b.json
- ✓ **IMU data loading**: Parse CSV with nanosecond timestamps
- ✓ **Camera data loading**: Frame timestamps and image paths
- ✓ **Ground truth loading**: Mocap trajectory data
- ✓ **Export to SimulationData**: Convert to common format

## Quick Start

### Generate Trajectory
```bash
# Generate circle trajectory (5 seconds, 100Hz)
./run.sh simulate circle --duration 5

# Generate figure-8 trajectory
./run.sh simulate figure8 --duration 10

# Generate spiral trajectory
./run.sh simulate spiral --duration 15

# Generate line trajectory
./run.sh simulate line --duration 8
```

### Output Format
Generated files are saved to `output/` directory as JSON with:
- Complete trajectory with poses and velocities
- Timestamps at specified sampling rate
- Metadata including trajectory type and parameters

## Next Steps (Phase 3)

- [ ] Implement landmark generation with visibility checking
- [ ] Add camera projection models
- [ ] Implement IMU measurement generation
- [ ] Add configurable noise models

## Project Structure
```
slam_simulation/
├── run.sh                  # Main runner script
├── requirements.txt        # Dependencies
├── setup.py               # Package setup
├── config/                # YAML configurations
│   ├── simulation_circle.yaml
│   ├── ekf.yaml
│   ├── swba.yaml
│   └── srif.yaml
├── src/
│   ├── common/
│   │   └── config.py      # Pydantic models
│   ├── utils/
│   │   └── math_utils.py  # SO3, SE3, quaternions
│   └── ...
├── tools/
│   └── cli.py            # Typer CLI
└── tests/
    ├── test_config.py    # Config validation tests
    └── test_math_utils.py # Math operation tests
```

## Dependencies

Core: numpy, scipy, pydantic, pyyaml, typer, plotly
Dev: pytest, black, mypy
Optional: numba, joblib (performance)