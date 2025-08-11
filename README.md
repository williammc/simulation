# SLAM Simulation System

A comprehensive simulation system for evaluating SLAM estimators (EKF, Sliding Window BA, SRIF) against ground truth data.

## Phase 1 Complete ✓

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

## Math Utilities API

```python
from src.utils.math_utils import *

# SO3 operations
R = so3_exp(omega)  # Axis-angle to rotation
omega = so3_log(R)  # Rotation to axis-angle

# SE3 operations
T = se3_exp(xi)     # Twist to transformation
xi = se3_log(T)     # Transformation to twist

# Quaternions
q = quaternion_multiply(q1, q2)
q_interp = quaternion_slerp(q1, q2, t)
R = quaternion_to_rotation_matrix(q)

# Transformations
p_new = transform_point(T, p)
v_new = transform_vector(R, v)
```

## Next Steps (Phase 2)

- [ ] Implement data structures (IMU, Camera, Trajectory)
- [ ] Create JSON I/O with schema validation
- [ ] Build simple trajectory generator
- [ ] Add TUM-VIE dataset reader

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