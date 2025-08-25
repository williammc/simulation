# SLAM Simulation System

A comprehensive Visual-Inertial SLAM simulation and evaluation framework supporting multiple estimation algorithms including GTSAM-based implementations with IMU preintegration.

## Features

- **Simulation**: Generate synthetic sensor data with realistic noise models and trajectories
- **Multiple Estimators**: 
  - EKF-SLAM: Classic Extended Kalman Filter
  - GTSAM-EKF: GTSAM with IMU preintegration and iSAM2
  - GTSAM-SWBA: Sliding Window Bundle Adjustment with GTSAM
  - SRIF: Square Root Information Filter (numerically stable)
- **IMU Preintegration**: Full support for GTSAM's CombinedImuFactor with bias estimation
- **Sensor Support**: Monocular/stereo cameras, IMU with realistic noise models
- **Evaluation**: Comprehensive metrics (ATE, RPE, runtime, memory) with statistical analysis
- **Visualization**: Interactive 3D trajectories, error plots, and performance dashboards
- **Benchmarking**: Automated evaluation pipeline with parallel execution

## Sensor Conventions

### IMU Model
The IMU is modeled as a single integrated sensor with co-located accelerometer and gyroscope:
- **Origin**: IMU center (unified for both accelerometer and gyroscope)
- **No lever arm**: Accelerometer and gyroscope share the same physical origin
- **Typical for MEMS**: Reflects modern MEMS IMUs (e.g., BMI055, ICM-20649) where both sensors are on the same chip
- **Frame**: Body frame (B) typically coincides with IMU frame, with `B_T_S = I` (identity) by default

This simplification is valid for most applications as the distance between accelerometer and gyroscope in MEMS chips is negligible (<1mm).

### IMU Preintegration (GTSAM)
The GTSAM-based estimators use IMU preintegration for efficient and accurate state estimation:

- **Specific Force**: IMU measures acceleration minus gravity (f = a - g)
- **Preintegration**: Combines multiple IMU measurements between keyframes into a single constraint
- **CombinedImuFactor**: GTSAM factor that handles:
  - Gravity compensation automatically
  - IMU bias estimation (accelerometer and gyroscope biases)
  - Proper covariance propagation
- **Benefits**:
  - O(1) computational complexity per keyframe (vs O(n) for raw integration)
  - Bias estimation improves long-term accuracy
  - Numerically stable through manifold-based integration

For detailed IMU preintegration documentation, see [docs/imu.md](docs/imu.md).

## Quick Start

### Setup
```bash
# Install dependencies
./run.sh setup

# Run tests to verify installation
./run.sh test
```

### Quick Command Reference
```bash
# Generate simulation data
./run.sh simulate circle --duration 10 --output sim.json

# Run GTSAM-EKF (recommended for IMU-heavy scenarios)
./run.sh slam gtsam-ekf --input sim.json --output result.json

# Evaluate results
./run.sh evaluate --result result.json --ground-truth sim.json

# Generate interactive plots
./run.sh plot sim.json --compare result.json

# Run complete benchmark
./run.sh e2e-simple --trajectory circle --estimator gtsam-ekf
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
./run.sh download tum-vie --sequence room1

# Convert to simulation format
./run.sh convert tumvie data/TUM-VIE/room1 --output output/room1.json

# Run SLAM and evaluate
./run.sh slam swba --input output/room1.json --output output/swba_result.json
./run.sh evaluate --result output/swba_result.json --ground-truth output/room1.json
```

#### 3. Comprehensive Evaluation Pipeline
```bash
# Run full evaluation across all datasets and estimators
./run.sh evaluation

# Use custom evaluation config
./run.sh evaluation config/evaluation_test.yaml

# Run on specific datasets/estimators
./run.sh evaluation --datasets circle,figure8 --estimators ekf,swba

# Dry run to preview execution plan
./run.sh evaluation --dry-run
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

#### Inspecting Simulation Data

```bash
# View simulation data structure
python -c "import json; d=json.load(open('output/simulation.json')); print(list(d.keys()))"

# Extract ground truth trajectory info
python -c "import json; d=json.load(open('output/simulation.json')); print(f\"Trajectory: {len(d['groundtruth']['trajectory'])} states\")"

# View trajectory and landmarks directly
./run.sh plot output/simulation.json --trajectory --landmarks
```

### Dataset Conversion

#### TUM-VIE Dataset Conversion
The TUM-VIE converter generates synthetic visual observations from mocap ground truth:

```bash
./run.sh convert tumvie INPUT_PATH OUTPUT_FILE [options]
  --num-landmarks INT     Number of synthetic 3D landmarks (default: 200)
  --keyframe-interval FLOAT  Time between keyframes in seconds (default: 0.1)
  --pixel-noise FLOAT     Standard deviation of pixel noise (default: 1.0)

# Examples:
# Convert TUM-VIE dataset with default settings
./run.sh convert tumvie data/TUM-VIE/room1 output/room1.json

# Generate dense observations (more landmarks, frequent keyframes)
./run.sh convert tumvie data/TUM-VIE/room1 output/room1_dense.json \
  --num-landmarks 500 --keyframe-interval 0.05

# Generate sparse, noisy observations for challenging scenarios
./run.sh convert tumvie data/TUM-VIE/room1 output/room1_sparse.json \
  --num-landmarks 100 --keyframe-interval 0.2 --pixel-noise 3.0
```

The converter:
1. Loads mocap ground truth trajectory and IMU measurements
2. Generates synthetic 3D landmarks around the trajectory
3. Creates keyframes at regular intervals
4. Projects landmarks to camera frames using calibration
5. Adds realistic pixel noise to observations

This approach ensures consistent, reproducible benchmarks without requiring actual images.

### SLAM Estimation
```bash
./run.sh slam [ekf|gtsam-ekf|gtsam-swba|srif] --input FILE [options]
  --config FILE       Estimator config YAML
  --output FILE       Output trajectory JSON
  --visualize         Show live visualization
  --verbose          Verbose output

# Available estimators:
# - ekf: Classic Extended Kalman Filter
# - gtsam-ekf: GTSAM with IMU preintegration (CombinedImuFactor)
# - gtsam-swba: GTSAM Sliding Window Bundle Adjustment
# - srif: Square Root Information Filter

# Examples:
./run.sh slam gtsam-ekf --input output/circle_sim.json --output output/gtsam_result.json
./run.sh slam gtsam-swba --input output/complex_sim.json --config config/estimators/swba.yaml
```

### Comparison
```bash
./run.sh compare --input FILE [options]
  --estimators LIST   Estimators to compare (default: all)
  --metrics LIST      Metrics to compute (ate,rpe,nees)
  --output FILE       Comparison results JSON
```

### Evaluation

#### Comprehensive Benchmarking
```bash
./run.sh evaluation [CONFIG_FILE] [options]
  --datasets LIST     Comma-separated list of datasets (default: all)
  --estimators LIST   Comma-separated list of estimators (default: all enabled)
  --parallel INT      Number of parallel jobs (default: from config)
  --output DIR        Override output directory from config
  --skip-generation   Skip dataset generation even if missing
  --skip-dashboard    Skip dashboard generation
  --dry-run          Show execution plan without running

# Examples:
# Full evaluation with default config
./run.sh evaluation

# Quick test with minimal config
./run.sh evaluation config/evaluation_test.yaml

# Benchmark specific combinations
./run.sh evaluation --datasets circle,spiral --estimators ekf,swba --parallel 4
```

The evaluation system:
- Automatically generates missing simulated datasets
- Downloads and converts TUM-VIE real-world datasets
- Runs all estimators in parallel
- Computes comprehensive metrics (ATE, RPE, runtime, memory)
- Generates interactive dashboard with KPI comparisons

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

### Evaluation Config
```yaml
# config/e2e_config.yaml
evaluation:
  name: "SLAM Evaluation Pipeline"
  output_dir: "output/evaluation"
  parallel_jobs: 4
  kpis: [ate_rmse, rpe_translation_rmse, runtime_ms, memory_mb]

datasets:
  simulated:
    generate_if_missing: true
    types:
      - name: "circle"
        config:
          trajectory:
            type: "circle"
            radius: 10.0
            duration: 60.0
          noise:
            add_noise: true
            imu_noise_level: 0.1
          landmarks:
            num_landmarks: 200
            
  tum_vi:
    download_if_missing: true
    sequences:
      - name: "mocap-desk"
        has_ground_truth: true

estimators:
  ekf:
    enabled: true
    config:
      process_noise:
        position: 0.01
        
  swba:
    enabled: true
    config:
      window_size: 10
      
  srif:
    enabled: true
    config:
      qr_threshold: 1e-10

dashboard:
  title: "SLAM Evaluation Dashboard"
  sections:
    - name: "Performance Matrix"
      type: "heatmap"
      metric: "ate_rmse"
    - name: "Timing Analysis"
      type: "bar_chart"
      metric: "runtime_ms"
```

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
│   ├── common/          # Shared data structures and utilities
│   ├── simulation/      # Trajectory and sensor data generation
│   ├── estimation/      # SLAM algorithms (EKF, GTSAM, SRIF)
│   ├── evaluation/      # Metrics, benchmarking, and analysis
│   ├── plotting/        # Visualization and dashboards
│   ├── utils/           # Math utilities and helpers
│   └── io/              # Data I/O and serialization
├── config/              # YAML configurations
│   ├── simulation/      # Trajectory and sensor configs
│   ├── estimators/      # Algorithm-specific configs
│   └── evaluation/      # Benchmark configurations
├── tools/               # CLI and utilities
│   ├── cli.py          # Main command interface
│   ├── simulate.py     # Simulation command
│   ├── slam.py         # SLAM estimation command
│   └── evaluate.py     # Evaluation command
├── tests/               # Unit and integration tests
│   ├── test_gtsam_ekf.py # GTSAM-EKF tests
│   ├── test_preintegration.py # IMU preintegration tests
│   └── gtsam-comparison/ # GTSAM validation tests
├── docs/                # Documentation
│   ├── architecture.md # System architecture
│   ├── imu.md          # IMU and preintegration guide
│   ├── simulation.md   # Simulation module docs
│   ├── estimation.md   # Estimator documentation
│   ├── plotting.md     # Visualization guide
│   ├── evaluation.md   # Metrics and benchmarking
│   └── tooling.md      # CLI and tools guide
└── notebooks/           # Jupyter notebooks for analysis
```

## Testing

```bash
# Run all tests
./run.sh test

# Run specific test suite
./run.sh test tests/test_ekf_slam.py

# Run GTSAM-specific tests
./run.sh test tests/test_gtsam_ekf.py
./run.sh test-gtsam  # Run GTSAM comparison tests

# Run with coverage
./run.sh test --coverage

# Run C++ tests (if built)
./run.sh test --cpp
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[architecture.md](docs/architecture.md)**: System architecture, design patterns, and data flow
- **[imu.md](docs/imu.md)**: IMU fundamentals, preintegration theory, and GTSAM integration
- **[simulation.md](docs/simulation.md)**: Trajectory generation, sensor simulation, and configuration
- **[estimation.md](docs/estimation.md)**: SLAM algorithms, implementation details, and usage
- **[evaluation.md](docs/evaluation.md)**: Metrics (ATE/RPE), benchmarking, and statistical analysis
- **[plotting.md](docs/plotting.md)**: Visualization tools, interactive plots, and dashboards
- **[tooling.md](docs/tooling.md)**: CLI commands, batch processing, and automation

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

## Requirements

### Software Dependencies
- Python 3.8+
- NumPy, SciPy, Matplotlib
- GTSAM Python bindings (`pip install gtsam`)
- Plotly for interactive visualization
- PyYAML for configuration
- Click for CLI

### Optional Dependencies
- C++ compiler (for C++ tests and binaries)
- Jupyter for notebooks
- pytest for testing

### Installation
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Or use the setup script
./run.sh setup
```

## Contributing

Contributions are welcome! Please ensure:
1. All tests pass (`./run.sh test`)
2. Code follows existing style conventions
3. New features include tests and documentation
4. IMU-related changes validate against GTSAM

## License

MIT License - See LICENSE file for details
