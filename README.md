# SLAM Simulation System

A comprehensive Visual-Inertial SLAM simulation and evaluation framework supporting multiple estimation algorithms (EKF, Sliding Window Bundle Adjustment, Square Root Information Filter).

## Features

- **Simulation**: Generate synthetic sensor data with configurable noise models
- **Multiple Estimators**: EKF, SWBA (optimization-based), SRIF (numerically stable)
- **Sensor Support**: Monocular/stereo cameras, single/multi-IMU configurations
- **Evaluation**: ATE, RPE, NEES metrics with statistical comparison
- **Visualization**: Interactive 3D trajectories and performance dashboards

## Quick Start

### Setup
```bash
# Install dependencies
./run.sh setup

# Run tests to verify installation
./run.sh test
```

### Example Workflows

#### 1. Generate and Process Synthetic Data
```bash
# Generate circle trajectory with landmarks and sensor measurements
./run.sh simulate circle --duration 20 --output output/circle_sim.json

# Run EKF SLAM on the generated data
./run.sh slam ekf --input output/circle_sim.json --output output/ekf_result.json

# Compare multiple estimators
./run.sh compare --input output/circle_sim.json --output output/comparison.json

# Visualize results
./run.sh dashboard --input output/ekf_result.json
```

#### 2. Process TUM-VIE Dataset
```bash
# Download TUM-VIE dataset
./run.sh download tumvie --sequence room1

# Convert to simulation format
./run.sh convert tumvie data/tumvie/room1 --output output/room1.json

# Run SLAM and evaluate
./run.sh slam swba --input output/room1.json --output output/swba_result.json
./run.sh evaluate --result output/swba_result.json --ground-truth output/room1.json
```

#### 3. Batch Evaluation
```bash
# Run all estimators on multiple trajectories
for traj in circle figure8 spiral; do
    ./run.sh simulate $traj --duration 30 --output output/${traj}_sim.json
    ./run.sh compare --input output/${traj}_sim.json --output output/${traj}_comparison.json
done

# Generate comparison report
./run.sh report output/*_comparison.json --output report.html
```

#### 4. Custom Configuration
```bash
# Use custom simulation config
./run.sh simulate --config config/custom_sim.yaml --output output/custom.json

# Run estimator with tuned parameters
./run.sh slam ekf --input output/custom.json --config config/ekf_tuned.yaml
```

## Available Commands

### Simulation
```bash
./run.sh simulate [circle|figure8|spiral|line] [options]
  --duration FLOAT     Trajectory duration in seconds (default: 20)
  --rate FLOAT        IMU rate in Hz (default: 200)
  --camera-rate FLOAT Camera rate in Hz (default: 30)
  --num-landmarks INT Number of landmarks (default: 100)
  --add-noise        Add noise to measurements (keeps ground truth separate)
  --noise-level FLOAT Noise scale factor (default: 1.0)
```

#### Understanding Simulation Output

When you run simulation with `--add-noise`, the output JSON contains:

1. **Ground Truth** (perfect, noise-free):
   - `groundtruth.trajectory`: True robot poses
   - `groundtruth.landmarks`: True 3D landmark positions

2. **Noisy Measurements** (what sensors observe):
   - `measurements.imu`: IMU data with noise
   - `measurements.camera_frames`: Camera observations with pixel noise

Example structure:
```json
{
  "groundtruth": {
    "trajectory": [...],  // True poses without noise
    "landmarks": [...]    // True 3D positions
  },
  "measurements": {
    "imu": [...],         // Noisy accelerometer/gyroscope
    "camera_frames": [...]  // Noisy pixel observations
  }
}
```

#### Inspecting Ground Truth

```bash
# View ground truth information
python tools/inspect_ground_truth.py inspect output/simulation.json

# Extract only ground truth to separate file
python tools/inspect_ground_truth.py extract-gt output/simulation.json

# Compare noise levels
python tools/inspect_ground_truth.py compare-noise output/simulation.json
```

### SLAM Estimation
```bash
./run.sh slam [ekf|swba|srif] --input FILE [options]
  --config FILE       Estimator config YAML
  --output FILE       Output trajectory JSON
  --visualize         Show live visualization
```

### Comparison
```bash
./run.sh compare --input FILE [options]
  --estimators LIST   Estimators to compare (default: all)
  --metrics LIST      Metrics to compute (ate,rpe,nees)
  --output FILE       Comparison results JSON
```

### Visualization

#### Interactive Plots
```bash
./run.sh plot INPUT_FILE [options]
  --compare FILE      Add comparison data (e.g., SLAM output)
  --output FILE       Output HTML file (default: auto-generated)
  --trajectory        Show 3D trajectory and landmarks
  --measurements      Show 2D camera measurements with keyframe selection
  --imu              Show IMU data (accelerometer & gyroscope)
  --keyframes NUM    Limit number of keyframes to display
  --no-browser       Don't auto-open in browser

# Examples:
# Plot simulation data
./run.sh plot output/circle_sim.json

# Compare ground truth with SLAM output
./run.sh plot output/circle_sim.json --compare output/ekf_result.json

# Generate specific plots only
./run.sh plot output/data.json --no-measurements --no-imu
```

#### Live Dashboard
```bash
./run.sh dashboard [options]
  --input FILE        Result or comparison JSON
  --port INT          Dashboard port (default: 8050)
  --no-browser        Don't open browser automatically
```

## Configuration Files

### Simulation Config
```yaml
# config/simulation.yaml
trajectory:
  type: circle
  params:
    radius: 2.0
    height: 1.5

sensors:
  cameras:
    - id: cam0
      model: pinhole-radtan
      rate: 30.0
      noise:
        pixel_std: 1.0
  
  imus:
    - id: imu0
      rate: 200.0
      noise:
        accel_noise_density: 0.01
        gyro_noise_density: 0.001

landmarks:
  count: 100
  distribution: uniform
  bounds: [-5, 5, -5, 5, 0, 3]
```

### Estimator Config
```yaml
# config/ekf.yaml
type: ekf
chi2_threshold: 5.991
innovation_threshold: 3.0
initial_covariance:
  position: 0.1
  orientation: 0.01
  velocity: 0.1
```

## Python API

```python
from src.simulation import generate_trajectory, add_sensor_noise
from src.estimation import EKFSlam, SlidingWindowBA, SRIFSlam
from src.evaluation import compute_ate, compute_rpe

# Generate trajectory
traj = generate_trajectory("circle", duration=20.0)

# Add measurements
sim_data = add_sensor_noise(traj, noise_level=1.0)

# Run SLAM
ekf = EKFSlam(config)
result = ekf.process(sim_data)

# Evaluate
ate, metrics = compute_ate(result.trajectory, sim_data.ground_truth)
print(f"ATE RMSE: {metrics.ate_rmse:.3f} meters")
```

## Performance Tips

- Use `--parallel` flag for multi-core processing
- Reduce trajectory duration for quick tests
- Use binary format (`--format binary`) for large datasets
- Enable numba JIT with `export SLAM_USE_NUMBA=1`

## Project Structure

```
slam_simulation/
├── src/
│   ├── simulation/      # Data generation
│   ├── estimation/      # SLAM algorithms
│   ├── evaluation/      # Metrics and comparison
│   └── visualization/   # Plotting and dashboard
├── config/             # YAML configurations
├── tools/              # CLI and utilities
└── tests/              # Unit tests
```

## Testing

```bash
# Run all tests
./run.sh test

# Run specific test suite
./run.sh test tests/test_ekf_slam.py

# Run with coverage
./run.sh test --coverage
```

## Development

### Important: Python Command Execution

**Always use `./run.sh cmd` wrapper for Python commands in this project.**

```bash
# ✅ CORRECT - Use run.sh wrapper
./run.sh cmd "python3 -m pytest tests/test_multi_imu.py -v"
./run.sh cmd "python3 tools/inspect_data.py --file output/sim.json"
./run.sh cmd "python3 -c 'import src.estimation; print(\"Success\")'"

# ❌ WRONG - Direct python calls may fail due to environment/path issues
python3 -m pytest tests/test_multi_imu.py -v
python3 tools/inspect_data.py --file output/sim.json
```

The `./run.sh cmd` wrapper ensures:
- Proper Python environment activation
- Correct PYTHONPATH configuration  
- Consistent dependency loading
- Cross-platform compatibility

### Running Individual Tests
```bash
# Single test method
./run.sh cmd "python3 -m pytest tests/test_multi_imu.py::TestIMUFusion::test_voting_fusion -v"

# Test class
./run.sh cmd "python3 -m pytest tests/test_multi_imu.py::TestIMUFusion -v"

# Multiple specific tests
./run.sh cmd "python3 -m pytest tests/test_multi_imu.py tests/test_stereo_camera.py -v"
```

## Citation

If you use this simulator in your research, please cite:
```bibtex
@software{slam_simulation,
  title = {SLAM Simulation System},
  year = {2025},
  url = {https://github.com/williammc/slam-simulation}
}
```

## License

MIT License - See LICENSE file for details