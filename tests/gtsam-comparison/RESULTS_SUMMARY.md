# GTSAM Comparison Test Results

## Executive Summary

✅ **Our IMU preintegration implementation is correct and matches GTSAM exactly**
- Maximum position difference: < 1e-14 meters (machine precision)
- Both implementations produce identical errors with perfect IMU

## Test Results

### 1. Speed Comparison
Shows how integration error increases with rotation speed:

| Period (s) | Angular Velocity (rad/s) | Centripetal Accel (m/s²) | Final Error (m) |
|------------|---------------------------|---------------------------|-----------------|
| 10.0       | 0.628                     | 0.8                       | 0.0392          |
| 5.0        | 1.257                     | 3.2                       | 0.0785          |
| 2.0        | 3.142                     | 19.7                      | 0.1962          |
| 1.0        | 6.283                     | 79.0                      | 0.3923          |

**Key Finding**: Error grows quadratically with angular velocity

### 2. Estimator Comparison (2s trajectory)
- GTSAM mean error: 0.1035m
- Our EKF mean error: 0.1035m
- Maximum difference: 4.52e-15m ✅

### 3. Detailed Comparison (10s trajectory)
- GTSAM final error: 0.0592m
- Our EKF final error: 0.0592m
- Mean error difference: 2.05e-14m ✅

### 4. Error Analysis
- X-axis max error: 0.0592m (identical for both)
- Y-axis max error: 0.0189m (identical for both)
- Z-axis max error: 4.57e-13m (near zero for planar motion)

## Interactive Visualizations

All tests generate interactive Plotly HTML files in `outputs/`:

1. **speed_comparison.html** - Error vs rotation speed analysis
2. **estimator_comparison.html** - Side-by-side trajectory comparison
3. **detailed_comparison.html** - 3D trajectory visualization
4. **error_analysis.html** - Per-axis error breakdown
5. **master_dashboard.html** - Summary dashboard with all key findings

## Conclusions

1. **Implementation Correctness**: Our preintegration matches GTSAM at machine precision level
2. **Error Source**: Discretization errors at high rotation rates (not implementation bugs)
3. **Error Pattern**: Quadratic growth with angular velocity
4. **Practical Impact**: At 6.28 rad/s (1 Hz rotation), expect ~40cm error after one rotation with perfect IMU

## Recommendations

For high-speed circular motion:
- Increase IMU rate (e.g., 1000 Hz instead of 200 Hz)
- Add visual constraints to bound drift
- Use motion priors when trajectory is known
- Consider higher-order integration methods