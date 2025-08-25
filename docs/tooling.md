# Tooling Documentation

## Overview

The tools module (`tools/`) provides command-line interfaces and utilities for running simulations, SLAM estimation, evaluation, and analysis. It includes both Python CLI tools and shell scripts for streamlined workflows.

## Table of Contents
- [Architecture](#architecture)
- [Command Line Interface](#command-line-interface)
- [Main Commands](#main-commands)
- [Shell Scripts](#shell-scripts)
- [Configuration Management](#configuration-management)
- [Batch Processing](#batch-processing)
- [Usage Examples](#usage-examples)

## Architecture

```
tools/
├── cli.py              # Main CLI entry point
├── simulate.py         # Simulation command
├── slam.py            # SLAM estimation command
├── evaluate.py        # Evaluation command
├── plot.py           # Plotting command
├── dashboard.py      # Dashboard generation
├── batch_runner.py   # Batch processing
└── utils.py          # CLI utilities

Scripts:
├── run.sh            # Main runner script
├── setup.sh          # Environment setup
└── test.sh          # Test runner
```

## Command Line Interface

### Main CLI Entry Point

The main CLI is accessed through `run.sh` or directly via Python:

```bash
# Using shell script
./run.sh [command] [options]

# Direct Python
python -m tools.cli [command] [options]
```

### Available Commands

```bash
./run.sh --help

Commands:
  simulate     Run simulation to generate synthetic SLAM data
  slam         Run SLAM estimator on simulation data
  evaluate     Evaluate SLAM results against ground truth
  dashboard    Generate dashboard from SLAM KPIs
  test         Run unit tests
  plot         Generate plots from results
  clean        Clean generated files
  info         Show system information
  e2e          Run end-to-end pipeline
  e2e-simple   Run simple end-to-end test
```

## Main Commands

### Simulate Command

Generate synthetic SLAM data:

```bash
# Basic simulation
./run.sh simulate --trajectory circle --duration 10

# With custom configuration
./run.sh simulate \
    --config config/simulation/circle.yaml \
    --output output/simulation.json

# Advanced options
./run.sh simulate \
    --trajectory figure8 \
    --duration 20 \
    --imu-rate 200 \
    --camera-rate 30 \
    --num-landmarks 500 \
    --seed 42 \
    --output output/custom_sim.json
```

**Options:**
- `--trajectory`: Trajectory type (circle, figure8, line, random)
- `--duration`: Simulation duration in seconds
- `--config`: Configuration file path
- `--output`: Output file path
- `--imu-rate`: IMU measurement rate (Hz)
- `--camera-rate`: Camera frame rate (Hz)
- `--num-landmarks`: Number of landmarks to generate
- `--seed`: Random seed for reproducibility

### SLAM Command

Run SLAM estimation:

```bash
# Run with default estimator
./run.sh slam --input output/simulation.json

# Specify estimator
./run.sh slam \
    --input output/simulation.json \
    --estimator gtsam-ekf \
    --output output/result.json

# With custom configuration
./run.sh slam \
    --input output/simulation.json \
    --estimator gtsam-swba \
    --config config/estimators/swba.yaml \
    --verbose
```

**Available Estimators:**
- `ekf`: Extended Kalman Filter
- `gtsam-ekf`: GTSAM-based EKF with IMU preintegration
- `gtsam-swba`: GTSAM Sliding Window Bundle Adjustment
- `srif`: Square Root Information Filter

**Options:**
- `--input`: Input simulation data
- `--estimator`: Estimator type
- `--config`: Estimator configuration
- `--output`: Output result file
- `--verbose`: Verbose output
- `--profile`: Enable profiling

### Evaluate Command

Evaluate estimation results:

```bash
# Basic evaluation
./run.sh evaluate \
    --result output/result.json \
    --ground-truth output/simulation.json

# With specific metrics
./run.sh evaluate \
    --result output/result.json \
    --ground-truth output/simulation.json \
    --metrics ate rpe runtime \
    --output output/evaluation.json

# Generate report
./run.sh evaluate \
    --result output/result.json \
    --ground-truth output/simulation.json \
    --report evaluation_report.html
```

**Options:**
- `--result`: Estimation result file
- `--ground-truth`: Ground truth data
- `--metrics`: Metrics to compute (ate, rpe, landmarks, runtime)
- `--output`: Output evaluation file
- `--report`: Generate HTML/PDF report

### Dashboard Command

Generate visualization dashboard:

```bash
# Generate dashboard from results
./run.sh dashboard \
    --simulation output/simulation.json \
    --result output/result.json \
    --evaluation output/evaluation.json \
    --output dashboard.html

# Interactive dashboard
./run.sh dashboard \
    --simulation output/simulation.json \
    --result output/result.json \
    --interactive \
    --port 8080
```

**Options:**
- `--simulation`: Simulation data
- `--result`: Estimation result
- `--evaluation`: Evaluation metrics
- `--output`: Output file (HTML/PDF)
- `--interactive`: Launch interactive server
- `--port`: Server port for interactive mode

### Plot Command

Generate specific plots:

```bash
# Plot trajectory
./run.sh plot trajectory \
    --result output/result.json \
    --ground-truth output/simulation.json \
    --output plots/trajectory.png

# Plot errors
./run.sh plot errors \
    --evaluation output/evaluation.json \
    --type ate rpe \
    --output plots/errors.png

# Plot IMU data
./run.sh plot imu \
    --simulation output/simulation.json \
    --output plots/imu.png
```

**Plot Types:**
- `trajectory`: 3D trajectory comparison
- `errors`: Error metrics over time
- `imu`: IMU measurements
- `camera`: Camera observations
- `landmarks`: Landmark positions
- `timeline`: Sensor timeline

## Shell Scripts

### Main Runner (run.sh)

```bash
#!/bin/bash
# Main runner script

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run CLI
python -m tools.cli "$@"
```

### Setup Script (setup.sh)

```bash
# Install dependencies
./setup.sh

# With specific Python version
./setup.sh --python python3.9

# Development setup
./setup.sh --dev
```

### Test Runner (test.sh)

```bash
# Run all tests
./test.sh

# Run specific test module
./test.sh tests/test_estimation.py

# With coverage
./test.sh --coverage

# Include C++ tests
./test.sh --cpp
```

## Configuration Management

### Configuration Files

Configuration files are organized by type:

```
config/
├── simulation/
│   ├── default.yaml
│   ├── circle.yaml
│   └── complex.yaml
├── estimators/
│   ├── ekf.yaml
│   ├── gtsam_ekf.yaml
│   └── swba.yaml
└── evaluation/
    └── metrics.yaml
```

### Loading Configuration

```python
# In CLI commands
from tools.utils import load_config

config = load_config('config/simulation/circle.yaml')

# With overrides
config = load_config(
    'config/simulation/default.yaml',
    overrides={
        'trajectory.radius': 3.0,
        'sensors.imu.rate': 500
    }
)
```

### Environment Variables

```bash
# Set default paths
export SLAM_SIM_CONFIG_DIR="/path/to/configs"
export SLAM_SIM_OUTPUT_DIR="/path/to/output"
export SLAM_SIM_CACHE_DIR="/path/to/cache"

# Enable debug mode
export SLAM_SIM_DEBUG=1
export SLAM_SIM_VERBOSE=1
```

## Batch Processing

### Batch Runner

Run multiple experiments:

```bash
# Run batch from configuration
./run.sh batch --config batch_config.yaml

# Parallel execution
./run.sh batch \
    --config batch_config.yaml \
    --parallel 4 \
    --output batch_results/
```

### Batch Configuration

```yaml
# batch_config.yaml
experiments:
  - name: "noise_study"
    base_config: "config/simulation/default.yaml"
    parameter_sweep:
      sensors.imu.noise: [0.001, 0.01, 0.1]
      sensors.camera.pixel_noise: [0.5, 1.0, 2.0]
    estimators: ["ekf", "gtsam-ekf"]
    
  - name: "trajectory_comparison"
    trajectories: ["circle", "figure8", "random"]
    estimators: ["gtsam-ekf", "gtsam-swba"]
    metrics: ["ate", "rpe", "runtime"]

output:
  format: "json"
  save_plots: true
  generate_report: true
```

### Grid Search

```python
# Grid search for parameter tuning
from tools.batch_runner import GridSearch

grid_search = GridSearch({
    'process_noise': [0.001, 0.01, 0.1],
    'measurement_noise': [0.1, 1.0, 10.0],
    'window_size': [5, 10, 20]
})

results = grid_search.run(
    estimator='gtsam-swba',
    dataset='simulation.json',
    metric='ate'
)

best_params = grid_search.get_best_parameters()
```

## Usage Examples

### End-to-End Pipeline

```bash
# Simple end-to-end test
./run.sh e2e-simple \
    --trajectory circle \
    --estimator gtsam-ekf

# Full pipeline with custom parameters
./run.sh e2e \
    --simulation-config config/simulation/complex.yaml \
    --estimator-config config/estimators/gtsam_ekf.yaml \
    --evaluation-metrics ate rpe runtime \
    --output-dir results/e2e_test/
```

### Comparative Analysis

```bash
# Compare multiple estimators
for estimator in ekf gtsam-ekf gtsam-swba; do
    ./run.sh slam \
        --input simulation.json \
        --estimator $estimator \
        --output results/${estimator}_result.json
    
    ./run.sh evaluate \
        --result results/${estimator}_result.json \
        --ground-truth simulation.json \
        --output results/${estimator}_eval.json
done

# Generate comparison dashboard
./run.sh dashboard \
    --results results/*_result.json \
    --evaluations results/*_eval.json \
    --comparison-mode \
    --output comparison_dashboard.html
```

### Parameter Sensitivity Study

```bash
# Study noise sensitivity
for noise in 0.001 0.01 0.1; do
    # Generate simulation with noise level
    ./run.sh simulate \
        --trajectory circle \
        --imu-noise $noise \
        --output sim_noise_${noise}.json
    
    # Run estimation
    ./run.sh slam \
        --input sim_noise_${noise}.json \
        --estimator gtsam-ekf \
        --output result_noise_${noise}.json
    
    # Evaluate
    ./run.sh evaluate \
        --result result_noise_${noise}.json \
        --ground-truth sim_noise_${noise}.json \
        --output eval_noise_${noise}.json
done

# Plot sensitivity analysis
./run.sh plot sensitivity \
    --evaluations eval_noise_*.json \
    --parameter noise \
    --output plots/noise_sensitivity.png
```

### Custom Pipeline

```python
# Create custom CLI command
# tools/custom_command.py

import click
from tools.cli import cli

@cli.command()
@click.option('--input', required=True)
@click.option('--output', default='output.json')
def custom_analysis(input, output):
    """Run custom analysis pipeline."""
    # Load data
    data = load_simulation(input)
    
    # Custom processing
    result = run_custom_algorithm(data)
    
    # Save result
    save_result(result, output)
    
    click.echo(f"Analysis complete: {output}")
```

## Advanced Features

### Profiling Support

```bash
# Enable profiling
./run.sh slam \
    --input simulation.json \
    --estimator gtsam-ekf \
    --profile \
    --profile-output profile.stats

# Analyze profile
python -m pstats profile.stats
```

### Memory Monitoring

```bash
# Monitor memory usage
./run.sh slam \
    --input simulation.json \
    --monitor-memory \
    --memory-interval 0.1
```

### Debug Mode

```bash
# Enable debug output
export SLAM_SIM_DEBUG=1
./run.sh slam \
    --input simulation.json \
    --debug \
    --breakpoint predict  # Break at prediction step
```

### Caching

```bash
# Enable caching for faster reruns
./run.sh simulate \
    --trajectory circle \
    --cache \
    --cache-dir .cache/

# Clear cache
./run.sh clean --cache
```

## Troubleshooting

### Common Issues

1. **Module Not Found**
   ```bash
   # Ensure PYTHONPATH is set
   export PYTHONPATH="$(pwd):${PYTHONPATH}"
   ```

2. **Permission Denied**
   ```bash
   # Make scripts executable
   chmod +x run.sh setup.sh test.sh
   ```

3. **Virtual Environment Issues**
   ```bash
   # Recreate virtual environment
   rm -rf .venv
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Configuration Errors**
   ```bash
   # Validate configuration
   ./run.sh validate-config config/simulation/custom.yaml
   ```

## Development

### Adding New Commands

1. Create command module in `tools/`
2. Register with CLI in `cli.py`
3. Add tests in `tests/test_tools/`
4. Update documentation

### Testing Tools

```python
# Test CLI commands
def test_simulate_command():
    from click.testing import CliRunner
    from tools.cli import cli
    
    runner = CliRunner()
    result = runner.invoke(cli, [
        'simulate',
        '--trajectory', 'circle',
        '--duration', '5'
    ])
    
    assert result.exit_code == 0
```

## References

- Click Documentation: https://click.palletsprojects.com/
- Python CLI Best Practices: https://docs.python-guide.org/scenarios/cli/
- Batch Processing: "GNU Parallel" documentation