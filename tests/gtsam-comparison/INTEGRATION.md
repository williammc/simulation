# GTSAM Comparison Tests - Integration Guide

## Overview

The GTSAM comparison tests verify that our IMU preintegration implementation matches GTSAM (the gold standard library) exactly. These tests are now fully integrated into the project's test suite.

## Running the Tests

### 1. Run All Tests (Including GTSAM)
```bash
./run.sh test
```
This is the default behavior - all tests including GTSAM comparisons will run.

### 2. Run Tests Without GTSAM Comparisons
```bash
./run.sh test --no-gtsam
```
Use this if you want to skip GTSAM tests (e.g., if GTSAM is not installed).

### 3. Run Only GTSAM Comparison Tests
```bash
./run.sh test-gtsam
```
This runs only the GTSAM comparison tests and generates interactive plots.

### 4. Verbose Output
```bash
./run.sh test-gtsam --verbose
```
Shows detailed test output including all print statements.

## Test Coverage

The GTSAM comparison tests include:

1. **Speed Comparison** (`test_speed_comparison.py`)
   - Tests error at different rotation speeds (0.63 to 6.28 rad/s)
   - Verifies quadratic error growth with angular velocity

2. **Estimator Comparison** (`test_estimator_comparison.py`)
   - Quick 2-second trajectory comparison
   - Verifies GTSAM and our EKF produce identical results

3. **Detailed Comparison** (`test_detailed_comparison.py`)
   - 10-second trajectory with 3D visualization
   - Comprehensive error analysis over longer duration

4. **Error Analysis** (`test_error_analysis.py`)
   - Per-axis (X, Y, Z) error breakdown
   - Cumulative error tracking

5. **All Comparisons** (`test_all_comparisons.py`)
   - Master test that runs all comparisons
   - Generates summary dashboard

## Interactive Visualizations

After running tests, interactive Plotly HTML plots are generated in:
```
tests/gtsam-comparison/outputs/
```

Key visualizations:
- `master_dashboard.html` - Summary dashboard with all findings
- `speed_comparison.html` - Error vs rotation speed
- `estimator_comparison.html` - Side-by-side trajectories
- `detailed_comparison.html` - 3D trajectory visualization
- `error_analysis.html` - Per-axis error breakdown

## Continuous Integration

These tests are automatically run as part of:
- `./run.sh test` - Default test suite
- CI/CD pipelines (if configured)
- Pre-commit hooks (if configured)

## Test Assertions

All tests include assertions to verify:
- GTSAM and our implementation match within 1e-10m
- Error increases monotonically with rotation speed
- Z-axis error remains near zero for planar motion
- All keyframe predictions are computed correctly

## Dependencies

Required packages (automatically installed):
- `gtsam` (Python bindings)
- `plotly` (for interactive visualizations)
- `pytest` (test framework)
- `numpy` (numerical computations)

## Troubleshooting

### GTSAM Not Installed
If GTSAM is not installed, you can:
1. Skip GTSAM tests: `./run.sh test --no-gtsam`
2. Install GTSAM: `pip install gtsam`

### Tests Failing
If tests fail, check:
1. GTSAM version compatibility
2. NumPy version compatibility
3. Review error messages for specific failures

### Plots Not Generated
Ensure the output directory exists:
```bash
mkdir -p tests/gtsam-comparison/outputs
```

## Summary

✅ GTSAM comparison tests are fully integrated into `./run.sh test`
✅ Dedicated command `./run.sh test-gtsam` for running only these tests
✅ Option to exclude with `./run.sh test --no-gtsam`
✅ Interactive Plotly visualizations generated automatically
✅ All tests pass, confirming our implementation matches GTSAM exactly