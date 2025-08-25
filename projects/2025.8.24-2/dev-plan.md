# GTSAM Integration Development Plan

## Overview
Phased development plan for migrating from custom SLAM estimators to GTSAM-based implementations.

---

## Phase 1: Environment Setup & Foundation (Day 1-2)

### Day 1: Dependencies & Project Structure

**Afternoon (4 hours)**
- [ ] Create new file structure:
  ```
  src/estimation/
  ├── gtsam_base.py (create)
  ├── gtsam_ekf_estimator.py (create skeleton)
  └── gtsam_swba_estimator.py (create skeleton)
  ```
- [ ] Copy reference code from `tests/gtsam-comparison/` to workspace
- [ ] Set up basic logging for GTSAM estimators

### Day 2: Base Infrastructure
**Morning (4 hours)**
- [ ] Implement `GtsamBaseEstimator` class:
  - [ ] `__init__` with config parsing
  - [ ] Basic factor graph initialization
  - [ ] Symbol management (X for poses, L for landmarks, V for velocities)
  - [ ] Noise model creation utilities

**Afternoon (4 hours)**
- [ ] Implement data conversion utilities in `GtsamBaseEstimator`:
  - [ ] `pose_to_gtsam()`: Convert Pose to gtsam.Pose3
  - [ ] `gtsam_to_pose()`: Convert gtsam.Pose3 to Pose
  - [ ] `extract_trajectory()`: Convert GTSAM values to Trajectory
  - [ ] `extract_landmarks()`: Convert GTSAM values to Map
- [ ] Create unit tests for conversions

---

## Phase 2: GtsamEkfEstimator Implementation (Day 3-5)

### Day 3: Core EKF Structure
**Morning (4 hours)**
- [ ] Implement `GtsamEkfEstimator.__init__`:
  - [ ] Initialize ISAM2 solver
  - [ ] Set up ISAM2 parameters
  - [ ] Initialize pose and landmark counters
- [ ] Implement `initialize()` method:
  - [ ] Add prior factor for initial pose
  - [ ] Add prior factors for velocity and biases
  - [ ] First ISAM2 update

**Afternoon (4 hours)**
- [ ] Create test file `tests/test_gtsam_ekf.py`
- [ ] Write test: `test_gtsam_ekf_initialization()`
- [ ] Write test: `test_gtsam_ekf_prior_factors()`
- [ ] Ensure initialization tests pass

### Day 4: IMU Integration
**Morning (4 hours)**
- [ ] Study reference implementation in `tests/gtsam-comparison/test_preintegration.py`
- [ ] Implement `predict()` method for IMU:
  - [ ] Create preintegrated IMU factor
  - [ ] Add between factor for poses
  - [ ] Handle velocity and bias variables

**Afternoon (4 hours)**
- [ ] Test IMU prediction:
  - [ ] Write test: `test_single_imu_prediction()`
  - [ ] Write test: `test_multiple_imu_predictions()`
  - [ ] Verify state propagation is correct
- [ ] Debug and fix IMU integration issues

### Day 5: Vision Integration & Testing
**Morning (4 hours)**
- [ ] Implement `update()` method for camera:
  - [ ] Add projection factors for landmarks
  - [ ] Handle new landmark initialization
  - [ ] Implement outlier rejection (if needed)

**Afternoon (4 hours)**
- [ ] Complete EKF testing:
  - [ ] Write test: `test_camera_update()`
  - [ ] Write test: `test_full_ekf_trajectory()`
  - [ ] Compare with reference implementation
- [ ] Implement `get_result()` method

---

## Phase 3: GtsamSWBAEstimator Implementation (Day 6-8)

### Day 6: SWBA Structure
**Morning (4 hours)**
- [ ] Implement `GtsamSWBAEstimator.__init__`:
  - [ ] Set up sliding window parameters
  - [ ] Initialize window management structures
  - [ ] Choose between ISAM2 or batch optimizer

**Afternoon (4 hours)**
- [ ] Implement window management:
  - [ ] `add_to_window()`: Add new keyframe
  - [ ] `is_window_full()`: Check window size
  - [ ] `get_oldest_frame()`: For marginalization
- [ ] Create test file `tests/test_gtsam_swba.py`

### Day 7: Marginalization
**Morning (4 hours)**
- [ ] Implement marginalization strategy:
  - [ ] Study GTSAM marginalization examples
  - [ ] Implement `marginalize_oldest_frame()`
  - [ ] Convert marginalized states to priors

**Afternoon (4 hours)**
- [ ] Test marginalization:
  - [ ] Write test: `test_window_size_maintained()`
  - [ ] Write test: `test_marginalization_consistency()`
  - [ ] Verify no information loss

### Day 8: SWBA Optimization
**Morning (4 hours)**
- [ ] Implement batch optimization within window:
  - [ ] Build factor graph for current window
  - [ ] Run Levenberg-Marquardt optimization
  - [ ] Extract and cache results

**Afternoon (4 hours)**
- [ ] Complete SWBA testing:
  - [ ] Write test: `test_swba_full_trajectory()`
  - [ ] Compare accuracy with EKF
  - [ ] Benchmark performance

---

## Phase 4: Integration & CLI (Day 9-10)

### Day 9: CLI Integration
**Morning (4 hours)**
- [ ] Update `tools/cli.py`:
  - [ ] Add 'gtsam-ekf' to estimator choices
  - [ ] Add 'gtsam-swba' to estimator choices
  - [ ] Update help documentation

**Afternoon (4 hours)**
- [ ] Update `tools/slam.py`:
  - [ ] Import GTSAM estimators
  - [ ] Add to estimator factory
  - [ ] Handle GTSAM-specific config

### Day 10: E2E Integration
**Morning (4 hours)**
- [ ] Create GTSAM config files:
  - [ ] `config/estimators/gtsam_ekf.yaml`
  - [ ] `config/estimators/gtsam_swba.yaml`
  - [ ] Set reasonable default parameters

**Afternoon (4 hours)**
- [ ] Test e2e integration:
  - [ ] Run: `./run.sh slam gtsam-ekf --input test_data.json`
  - [ ] Run: `./run.sh e2e/e2e-simple --estimator gtsam-ekf`
  - [ ] Verify output format compatibility


---

## Phase 5: Documentation & Polish (Day 13-14)

### Day 13: Documentation
**Morning (4 hours)**
- [ ] Write `docs/gtsam_integration.md`:
  - [ ] Architecture overview
  - [ ] Usage examples
  - [ ] Configuration guide

**Afternoon (4 hours)**
- [ ] Update existing docs:
  - [ ] Update README.md
  - [ ] Update architecture.md
  - [ ] Create migration guide

### Day 14: Final Testing & Optimization
**Morning (4 hours)**
- [ ] Run full test suite:
  - [ ] All unit tests pass
  - [ ] All integration tests pass
  - [ ] E2E tests complete successfully

**Afternoon (4 hours)**
- [ ] Performance optimization:
  - [ ] Profile code for bottlenecks
  - [ ] Optimize factor graph structure
  - [ ] Final benchmarks

---

## Daily Verification Checklist

### End of Each Day:
- [ ] All new code has unit tests
- [ ] Tests are passing
- [ ] Code is committed to version control
- [ ] Progress logged in daily notes

### Key Milestones:
- **Day 2**: Base infrastructure complete
- **Day 5**: GtsamEkfEstimator fully functional
- **Day 8**: GtsamSWBAEstimator fully functional
- **Day 10**: CLI integration complete
- **Day 12**: Legacy code removed
- **Day 14**: Project complete

---

## Risk Management

### Potential Blockers & Mitigation:

1. **GTSAM Installation Issues**
   - Mitigation: Have Docker fallback ready
   - Time buffer: 0.5 days

2. **IMU Preintegration Complexity**
   - Mitigation: Use reference implementation
   - Time buffer: 1 day

3. **Marginalization Bugs**
   - Mitigation: Start with simple fixed-lag smoother
   - Time buffer: 0.5 days

4. **Performance Issues**
   - Mitigation: Profile early and often
   - Time buffer: 1 day

---

## Testing Strategy

### Unit Tests (Run after each component):
```bash
pytest tests/test_gtsam_base.py -xvs
pytest tests/test_gtsam_ekf.py -xvs
pytest tests/test_gtsam_swba.py -xvs
```

### Integration Tests (Run daily):
```bash
./run.sh test --gtsam
```

### E2E Tests (Run after Day 10):
```bash
./run.sh e2e/e2e-simple --estimator gtsam-ekf
./run.sh e2e/e2e-simple --estimator gtsam-swba
```

### Performance Tests (Run on Days 11 & 14):
```bash
python tools/benchmark_estimators.py
```

---

## Success Metrics

### Code Quality:
- [ ] Zero failing tests
- [ ] >90% code coverage on new code
- [ ] All functions have docstrings
- [ ] Type hints on all methods

### Performance:
- [ ] Runtime within 2x of legacy
- [ ] Memory usage <1GB for test cases
- [ ] ATE accuracy within 10% of legacy

### Integration:
- [ ] Works with all existing data formats
- [ ] CLI commands functional
- [ ] E2E tests passing

---

## Notes for Developers

### GTSAM Quick References:
- Factor Graph: `gtsam.NonlinearFactorGraph()`
- Values: `gtsam.Values()`
- Symbols: `gtsam.symbol('x', 0)` for x0
- Pose: `gtsam.Pose3(rotation, translation)`
- Prior: `gtsam.PriorFactorPose3(key, pose, noise)`

### Common Issues:
1. **Symbol conflicts**: Use consistent naming (X=poses, L=landmarks, V=velocities, B=biases)
2. **Noise models**: GTSAM uses information form (inverse covariance)
3. **Optimization**: May need multiple iterations for convergence
4. **Memory**: Clear factor graph after ISAM2 updates

### Useful Commands:
```bash
# Check GTSAM version
python -c "import gtsam; print(gtsam.__version__)"

# Run specific test
pytest tests/test_gtsam_ekf.py::test_initialization -xvs

# Profile memory usage
mprof run python tools/slam.py gtsam-ekf --input data.json
mprof plot
```