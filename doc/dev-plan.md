# Development Plan - SLAM Simulation System

## Overview
Phased implementation focusing on incremental, testable components. Each phase produces working functionality that can be validated before proceeding.

---

## Phase 1: Foundation & Infrastructure (Week 1)

### 1.1 Project Setup
- [ ] Initialize project structure per requirements
- [ ] Create `run.sh` with basic commands
- [ ] Setup `requirements.txt` with core dependencies (numpy, scipy, pydantic, plotly, typer)
- [ ] Create base Typer CLI in `tools/cli.py`

### 1.2 Configuration System
- [ ] Implement Pydantic models for all configs
- [ ] Create YAML loaders with validation
- [ ] Add sample config files in `config/`
- [ ] Unit tests for config validation

### 1.3 Math Utilities
- [ ] SO3/SE3 operations (rotation, transformation matrices)
- [ ] Quaternion operations and conversions
- [ ] Coordinate frame transformations
- [ ] Unit tests for all math operations

**Deliverable:** Working CLI with config system and math utilities

---

## Phase 2: Data Structures & I/O (Week 2)

### 2.1 Core Data Models
- [ ] Implement sensor data structures (IMU, Camera measurements)
- [ ] Trajectory representation (pose, velocity, timestamps)
- [ ] Landmark/feature point structures
- [ ] Calibration data structures

### 2.2 JSON I/O
- [ ] Implement JSON schema from specs
- [ ] Data serialization/deserialization
- [ ] TUM-VIE dataset reader (basic)
- [ ] Unit tests for I/O operations

### 2.3 Simple Trajectory Generator
- [ ] Circle trajectory only (simplest case)
- [ ] Uniform timestamp sampling
- [ ] Velocity/acceleration computation
- [ ] Export to JSON format

**Deliverable:** Can generate and save simple circle trajectory

---

## Phase 3: Basic Simulation (Week 3)

### 3.1 Landmark Generation
- [ ] Random 3D point cloud in bounding box
- [ ] Visibility checking (frustum culling)
- [ ] Store landmark ground truth

### 3.2 Ideal Measurements
- [ ] Camera projection (pinhole model, no distortion)
- [ ] Perfect IMU measurements (no noise)
- [ ] Data association (known correspondences)
- [ ] Generate measurement JSON

### 3.3 Noise Models
- [ ] IMU noise generation (white noise + bias)
- [ ] Camera measurement noise (Gaussian pixel noise)
- [ ] Add noise to ideal measurements
- [ ] Configurable noise levels

**Deliverable:** Complete simulation with noisy measurements

---

## Phase 4: Visualization Tools (Week 4)

### 4.1 Trajectory Plotting
- [ ] 3D trajectory with Plotly
- [ ] Ground truth vs estimated (placeholder)
- [ ] Save as HTML

### 4.2 Measurement Visualization
- [ ] 2D feature tracks in image plane
- [ ] IMU data time series plots
- [ ] Landmark 3D scatter plot

### 4.3 Basic Dashboard
- [ ] Static HTML generation
- [ ] Layout with multiple plots
- [ ] Placeholder for KPIs

**Deliverable:** Visualization of simulation data

---

## Phase 5: Estimator Base Class (Week 5)

### 5.1 Abstract Estimator
- [ ] Define state vector interface
- [ ] Abstract methods (predict, update, optimize)
- [ ] Error metrics computation
- [ ] Result storage format

### 5.2 IMU Integration
- [ ] IMU preintegration class
- [ ] Discrete integration (Euler/RK4)
- [ ] Jacobian computation
- [ ] Unit tests with analytical solutions

### 5.3 Camera Measurement Model
- [ ] Projection/unprojection functions
- [ ] Jacobians for optimization
- [ ] Reprojection error computation

**Deliverable:** Base classes ready for estimator implementation

---

## Phase 6: EKF Implementation (Week 6)

### 6.1 EKF Core
- [ ] State and covariance initialization
- [ ] IMU prediction step
- [ ] Camera update step
- [ ] Simple outlier rejection (chi-squared test)

### 6.2 EKF Testing
- [ ] Test on circle trajectory
- [ ] Verify covariance consistency
- [ ] Compare with ground truth
- [ ] Save results to JSON

**Deliverable:** Working EKF on simple trajectory

---

## Phase 7: Sliding Window BA (Week 7-8)

### 7.1 Optimization Framework
- [ ] Cost function formulation
- [ ] Jacobian computation (analytical)
- [ ] Gauss-Newton solver
- [ ] Huber robust cost

### 7.2 Sliding Window
- [ ] Keyframe management
- [ ] Marginalization (Schur complement)
- [ ] Window size control
- [ ] Prior factors from marginalization

### 7.3 SWBA Testing
- [ ] Test on same scenarios as EKF
- [ ] Convergence analysis
- [ ] Performance comparison

**Deliverable:** Working SWBA implementation

---

## Phase 8: SRIF Implementation (Week 8-9)

### 8.1 Square Root Form
- [ ] QR factorization updates
- [ ] Information matrix representation
- [ ] Measurement updates via QR

### 8.2 SRIF Testing
- [ ] Numerical stability tests
- [ ] Comparison with EKF (should be equivalent)
- [ ] Performance benchmarking

**Deliverable:** Working SRIF implementation

---

## Phase 9: Evaluation Framework (Week 9)

### 9.1 Metrics Implementation
- [ ] ATE/RPE computation
- [ ] NEES consistency test
- [ ] Timing and memory profiling

### 9.2 Comparison Tools
- [ ] Run all estimators on same data
- [ ] Generate comparison tables
- [ ] Statistical significance tests

### 9.3 KPI Dashboard
- [ ] Load multiple run results
- [ ] Interactive comparison plots
- [ ] Export report generation

**Deliverable:** Complete evaluation pipeline

---

## Phase 10: Advanced Features (Week 10)

### 10.1 Additional Trajectories
- [ ] Figure-8, spiral, line trajectories
- [ ] Trajectory interpolation (splines)

### 10.2 Camera Distortion
- [ ] Radial-tangential model
- [ ] Distortion in projection
- [ ] Undistortion for measurements

### 10.3 Multi-Sensor Support
- [ ] Stereo camera setup
- [ ] Multiple IMU handling
- [ ] Sensor synchronization

### 10.4 TUM-VIE Integration
- [ ] Complete dataset reader
- [ ] Calibration file parsing
- [ ] Download automation

**Deliverable:** Full feature set

---

## Testing Strategy

### Unit Tests (Continuous)
- Math operations correctness
- Data I/O integrity
- Individual component validation

### Integration Tests (Per Phase)
- End-to-end simulation runs
- Estimator convergence
- Performance benchmarks

### Validation Tests (Final)
- Against TUM-VIE ground truth
- Cross-validation between estimators
- Monte Carlo consistency tests

---

## Risk Mitigation

1. **Numerical Issues**: Start with simple, well-conditioned scenarios
2. **Performance**: Profile early, optimize only proven bottlenecks
3. **Debugging**: Extensive logging, intermediate result saving
4. **Scope Creep**: Defer advanced features to Phase 10

---

## Success Criteria

- [ ] All three estimators produce reasonable trajectories
- [ ] ATE < 5% of trajectory scale on simple scenarios
- [ ] Consistent covariance estimates (NEES test passes)
- [ ] Complete pipeline from simulation to evaluation
- [ ] Documentation and examples for each component