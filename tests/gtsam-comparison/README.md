# GTSAM Comparison Tests

This directory contains comprehensive tests comparing our IMU preintegration implementation with GTSAM (the gold standard library for IMU preintegration).

## Key Finding

**Our implementation is correct and matches GTSAM exactly** (differences < 1e-14 meters, which is machine precision).

## Test Files

1. **test_speed_comparison.py** - Compares error at different rotation speeds
2. **test_estimator_comparison.py** - Quick 2-second trajectory comparison
3. **test_detailed_comparison.py** - 10-second trajectory with 3D visualization
4. **test_error_analysis.py** - Per-axis error breakdown
5. **test_all_comparisons.py** - Master test runner that generates all plots

## Running the Tests

### Run all tests:
```bash
python -m pytest tests/gtsam-comparison/test_all_comparisons.py -v
```

### Run individual tests:
```bash
python tests/gtsam-comparison/test_speed_comparison.py
python tests/gtsam-comparison/test_estimator_comparison.py
python tests/gtsam-comparison/test_detailed_comparison.py
python tests/gtsam-comparison/test_error_analysis.py
```

## Generated Outputs

All tests generate interactive Plotly HTML plots in `tests/gtsam-comparison/outputs/`:

- **speed_comparison.html** - Shows how error increases with rotation speed
- **estimator_comparison.html** - 2D trajectory and error plots
- **detailed_comparison.html** - 3D trajectory visualization
- **error_analysis.html** - Per-axis error analysis
- **master_dashboard.html** - Summary dashboard with key findings

## Key Results

### Error vs Rotation Speed (Perfect IMU, r=2m)

| Period (s) | Angular Velocity (rad/s) | Centripetal Accel (m/s²) | Final Error (m) |
|------------|---------------------------|---------------------------|-----------------|
| 10.0       | 0.628                     | 0.8                       | 0.039           |
| 5.0        | 1.257                     | 3.2                       | 0.079           |
| 2.0        | 3.142                     | 19.7                      | 0.196           |
| 1.0        | 6.283                     | 79.0                      | 0.392           |

### Why Errors Occur with Perfect IMU

Even with noiseless IMU measurements, errors accumulate due to:

1. **Discrete Integration** - IMU at 200 Hz must approximate continuous motion
2. **High Centripetal Acceleration** - At 2π rad/s, reaches 79 m/s²
3. **Rotation-Translation Coupling** - Fast rotation creates complex interactions
4. **Finite Sampling** - Cannot capture instantaneous changes between samples

## Verification Tests

All tests include assertions to verify:
1. GTSAM and our implementation produce identical results
2. Z-axis error remains near zero for planar motion
3. Error increases monotonically with rotation speed
4. Implementation differences are at machine precision level

## Dependencies

- numpy
- gtsam (python bindings)
- plotly
- pytest