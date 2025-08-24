# Development Plan - Simplified Educational Trajectory Estimation

## Project Vision
Transform the codebase into a **lean, educational demonstration** of trajectory estimation algorithms using preintegrated IMU and keyframe-based processing. Focus on clarity, minimalism, and pedagogical value.

### Core Principles
1. **Simplicity First**: Remove complex SLAM features, focus on trajectory estimation
2. **Educational Value**: Clear code with explanations, suitable for learning
3. **Minimal Dependencies**: Use only essential libraries
4. **Clean Pipeline**: Simple flow from simulation → estimation → evaluation
5. **Comparison Focus**: Easy comparison between EKF, SWBA, and SRIF approaches

### Simplified Architecture
```
Simulation (generates keyframes + preintegrated IMU)
    ↓
Estimation (EKF / SWBA / SRIF - trajectory only)
    ↓
Evaluation (ATE, RPE, NEES metrics)
    ↓
Visualization (plots, KPI dashboard, HTML reports)
```

### What Was Removed (Simplification)
- ❌ Raw IMU processing (only preintegrated)
- ❌ Visual SLAM features (no landmark mapping)
- ❌ Complex visual updates (minimal keyframe handling)
- ❌ Loop closure detection
- ❌ Map management and optimization
- ❌ Multi-camera support complexity

### What Remains (Core Educational Value)
- ✅ Three estimation algorithms (EKF, SWBA, SRIF)
- ✅ Preintegrated IMU for efficiency
- ✅ Keyframe-based trajectory estimation
- ✅ Comprehensive metrics (ATE, RPE, NEES)
- ✅ Clean visualization and reporting
- ✅ Side-by-side algorithm comparison

## Phase 1: Config Refactoring (Priority: High) ✅ COMPLETED
**Actual Time: 2.5 hours**

### Completed Steps:
- [x] Created branch `refactor-configs`
- [x] Removed duplicate configs from all estimator files
- [x] Updated imports to use `src.common.config`
- [x] Created `BaseEstimatorConfig` with inheritance hierarchy
- [x] Migrated all config fields with proper validation
- [x] Fixed field naming (robust_cost_type → robust_kernel)
- [x] Unified EstimatorType enum
- [x] Updated all tests and config files
- [x] Fixed SRIF quaternion dimension issue (SO3 vs quaternion)
- [x] **Result: All 297 tests passing**

---

## Phase 2: Preintegrated IMU Support (Priority: Medium) ✅
**Estimated Time: 4-6 hours** *(Completed)*

### Step 2.1: Data Structure Implementation ✅
- [x] Create branch `preintegrated-imu-support`
- [x] Add `PreintegratedIMUData` class to `src/common/data_structures.py`
  - [x] Include `from_keyframe_id` and `to_keyframe_id` fields
  - [x] Add keyframe association validation
- [x] Add necessary imports (numpy, typing)
- [x] Implement validation methods

### Step 2.2: Config Updates ✅
- [x] Add `use_preintegrated_imu: bool = False` to `EKFConfig`
- [x] Add `use_preintegrated_imu: bool = False` to `SWBAConfig`
- [x] Add `use_preintegrated_imu: bool = False` to `SRIFConfig`
- [x] Add preintegration parameters (if needed)

### Step 2.3: Estimator Refactoring - EKF ✅
- [x] Split `predict()` method into `_predict_raw()` and `_predict_preintegrated()`
- [x] Update method signature to accept `Union[List[IMUMeasurement], PreintegratedIMUData]`
- [x] Implement preintegrated prediction logic
- [x] Update state covariance propagation

### Step 2.4: Estimator Refactoring - SWBA ✅
- [x] Split optimization to handle preintegrated factors
- [x] Create IMU factor from preintegrated data
- [x] Update residual computation
- [x] Modify Jacobian calculations

### Step 2.5: Estimator Refactoring - SRIF ✅
- [x] Adapt information matrix updates for preintegrated data
- [x] Modify QR factorization approach
- [x] Update measurement incorporation

### Step 2.6: Preintegration Pipeline ✅
- [x] Add method to `IMUPreintegrator` to batch process measurements between keyframes
- [x] Create utility function to convert raw IMU to preintegrated between keyframe pairs
- [x] Link preintegrated data to target keyframe (`CameraFrame.preintegrated_imu`)
- [x] Add caching mechanism for preintegrated results
- [x] Ensure keyframe IDs are properly tracked in preintegrated data

### Step 2.7: Simulation Integration ✅
- [x] Add config option in `SimulationConfig` for preintegration
- [x] Modify data generation pipeline
- [x] Create preintegration step in simulation workflow
- [x] Store both raw and preintegrated data

### Step 2.8: Testing Infrastructure ✅
- [x] Create test fixtures with preintegrated data
- [x] Add unit tests for `PreintegratedIMUData`
- [x] Test estimator behavior with both data types
- [x] Validate numerical consistency

### Step 2.9: Integration Tests ✅
- [x] Create comparison tests (raw vs preintegrated)
- [x] Verify state estimation accuracy
- [x] Check computational performance
- [x] Validate covariance consistency

---

## Phase 3: Keyframe Selection Implementation (Priority: Medium) ✅
**Estimated Time: 3-4 hours**
**Status: COMPLETED**

### Step 3.1: Configuration Classes ✅
- [x] Create branch `keyframe-selection`
- [x] Add `KeyframeSelectionConfig` to `src/common/config.py`
- [x] Add validation for strategy-specific fields
- [x] Update `SimulationConfig` to include keyframe selection

### Step 3.2: Data Structure Updates ✅
- [x] Add `is_keyframe` field to `CameraFrame` *(Completed in Phase 2)*
- [x] Add `keyframe_id` field to `CameraFrame` *(Completed in Phase 2)*
- [x] Add `preintegrated_imu` field to `CameraFrame` for IMU between keyframes *(Completed in Phase 2)*
- [x] Update serialization methods for new fields *(Completed in Phase 2)*
- [x] Ensure backward compatibility with existing data

### Step 3.3: Keyframe Selector Implementation ✅
- [x] Create `KeyframeSelector` class in `src/simulation/keyframe_selector.py`
- [x] Implement fixed interval strategy
- [x] Implement motion-based strategy
- [x] Implement hybrid strategy
- [x] Add unit tests for each strategy

### Step 3.4: Simulation Pipeline Integration ✅
- [x] Integrate `KeyframeSelector` into data generation
- [x] Mark frames as keyframes during simulation *(Partially done in Phase 2)*
- [x] Assign sequential keyframe IDs *(Partially done in Phase 2)*
- [x] Trigger IMU preintegration between consecutive keyframes *(Completed in Phase 2)*
- [x] Store preintegrated IMU data in corresponding keyframes *(Completed in Phase 2)*
- [x] Update simulation output to include keyframe and preintegration info *(Completed in Phase 2)*

### Step 3.5: Estimator Updates ✅
- [x] Add `use_keyframes_only` to `EKFConfig`
- [x] Add `use_simulation_keyframes` to `SWBAConfig`
- [x] Add `use_keyframes_only` to `SRIFConfig`
- [x] Update estimator update methods to check keyframe flag
- [x] Modify SWBA to optionally use pre-selected keyframes

### Step 3.6: Visualization Updates ✅
- [x] Update trajectory plots to highlight keyframes
- [x] Add keyframe markers in 3D plots
- [x] Create keyframe statistics visualization
- [x] Update plot legends

### Step 3.7: Testing ✅
- [x] Test fixed interval selection
- [x] Test motion-based selection
- [x] Test hybrid selection
- [x] Verify keyframe consistency across runs
- [x] Test estimator behavior with keyframes

---

## Phase 4: Keyframe-Preintegration Integration (Priority: High) ✅ COMPLETED
**Estimated Time: 2 hours**
**Actual Time: 3 hours**

### Step 4.1: Data Flow Integration ✅
- [x] Ensure keyframes are selected before IMU preintegration
- [x] Verify preintegrated data references correct keyframe IDs
- [x] Test data consistency between visual and inertial constraints

### Step 4.2: Estimator Integration ✅
- [x] Update SWBA to use keyframe-associated preintegrated IMU
- [x] Modify EKF to optionally skip non-keyframe updates
- [x] Update SRIF to handle keyframe-based processing
- [x] **Note: Simplified to preintegrated-only processing**

### Step 4.3: Factor Graph Construction ✅
- [x] Create IMU factors between consecutive keyframes
- [x] Ensure proper factor connectivity in optimization
- [x] Validate Jacobian computations with keyframe states

### Step 4.4: Integration Testing ✅
- [x] Test complete pipeline: keyframe selection → IMU preintegration → estimation
- [x] Verify consistency of results across different keyframe intervals
- [x] Validate that preintegrated IMU data matches raw integration  
- [x] Check memory efficiency with sparse keyframes
- [x] **Result: 336 tests passing with simplified estimators**

---

## Phase 5: Educational Documentation & Validation (Priority: High)
**Purpose: Create a lean, educational demonstration of trajectory estimation**
**Estimated Time: 3-4 hours**
**Status: PARTIALLY COMPLETED**

### Step 5.1: Simplified Performance Metrics
- [ ] Create single benchmark script comparing EKF vs SWBA vs SRIF
- [ ] Generate clear runtime comparison table
- [ ] Plot memory usage for each estimator
- [ ] Document computational complexity (O notation)
- [x] Create performance summary report (via EstimatorResultStorage.create_kpi_summary)

### Step 5.2: Educational Notebooks & Tutorials
- [ ] Create `notebooks/` directory for interactive tutorials
- [ ] `00_quickstart.ipynb` - Getting started in 5 minutes
  - Load sample data
  - Run simple estimation
  - Visualize results
  - Key concepts overview
- [ ] `01_ekf_basics.ipynb` - Extended Kalman Filter deep dive
  - **Learning Objectives**: Understand predict-update cycle, covariance propagation
  - Predict step mathematics with LaTeX
  - Update step derivation
  - Interactive covariance ellipse visualization
  - Compare predicted vs corrected trajectories
  - Effect of noise parameters
- [ ] `02_swba_optimization.ipynb` - Sliding Window Bundle Adjustment
  - **Learning Objectives**: Graph optimization, window management, factor graphs
  - Factor graph visualization with networkx
  - Cost function exploration
  - Interactive window size effects
  - Convergence animation
  - Comparison with batch optimization
- [ ] `03_srif_information.ipynb` - Square Root Information Filter
  - **Learning Objectives**: Information form, numerical stability, QR factorization
  - Information matrix vs covariance matrix
  - QR factorization step-by-step
  - Numerical stability comparison with EKF
  - Sparse matrix visualization
  - Computational complexity analysis
- [ ] `04_algorithm_comparison.ipynb` - Comprehensive comparison
  - **Learning Objectives**: Trade-offs, use cases, performance metrics
  - Run all three estimators on same data
  - Interactive parameter tuning with sliders
  - Real-time performance metrics
  - Error distribution analysis
  - Memory and computation profiling
- [ ] `05_preintegration_tutorial.ipynb` - IMU preintegration explained
  - **Learning Objectives**: Preintegration theory, bias correction, manifold operations
  - Mathematical derivation with SymPy
  - Numerical vs analytical integration comparison
  - Bias correction effects visualization
  - Covariance propagation
  - SO(3) manifold operations

### Step 5.3: KPI Reports & Visualization
- [x] Create unified KPI dashboard generator (EstimatorResultStorage.create_kpi_summary)
- [~] Generate trajectory error plots (ATE, RPE) (partial - metrics computed but no dedicated plots)
- [~] Create consistency metrics visualization (NEES) (partial - metrics defined, dashboard has method)
- [ ] Build comparison matrix (accuracy vs speed)
- [x] Export results as clean HTML report (multiple dashboard generators exist)
- [ ] Add LaTeX equations in docstrings for key algorithms

### Step 5.4: Minimalistic Documentation
- [ ] Write `ALGORITHMS.md` - concise explanation of EKF, SWBA, SRIF
- [ ] Create `QUICKSTART.md` - run simulation in 5 minutes
- [ ] Document simplified pipeline: simulate → estimate → evaluate
- [ ] Add flowchart diagrams for each estimator
- [ ] Create config templates with educational comments

### Step 5.5: Clean Code & Testing
- [ ] Remove any remaining SLAM/landmark code
- [ ] Simplify interfaces to essential parameters only
- [ ] Add type hints for educational clarity
- [ ] Create unit tests that demonstrate concepts
- [ ] Ensure all examples run without errors

### Step 5.6: C++ Integration (ADDED - COMPLETED)
- [x] Design C++ header-only EstimatorResult I/O library
- [x] Create `cpp_estimation/include/simulation_io/estimator_result_io.hpp`
- [x] Implement EstimatorResultIO::save() and load() methods
- [x] Ensure compatibility with Python EstimatorResultStorage
- [x] Create example demonstrating C++ result generation
- [x] Verify Python can load and evaluate C++ results
- [x] Add cpp_implementation flag for C++ origin identification

### Step 5.7: Result Storage Refactoring (ADDED - COMPLETED)
- [x] Refactor tools/slam.py to use EstimatorResultStorage.save_result()
- [x] Refactor tools/evaluate.py to use EstimatorResultStorage.load_result()
- [x] Add comprehensive unit tests for EstimatorResultStorage
- [x] Fix consistency metrics serialization format
- [x] Test end-to-end with refactored tools

### Step 5.8: Example Usage Patterns
```bash
# Launch interactive notebooks
jupyter notebook notebooks/01_ekf_basics.ipynb

# Run all notebooks in sequence
jupyter nbconvert --execute notebooks/*.ipynb

# Generate static HTML from notebooks
jupyter nbconvert --to html notebooks/04_algorithm_comparison.ipynb

# Quick command-line comparison
python -m examples.quick_compare --trajectory circle --noise low

# Generate comprehensive report
python -m reports.generate_kpi --input results/ --output kpi_report.html
```

### Step 5.7: Notebook Features
- [ ] Interactive widgets for parameter tuning (ipywidgets)
- [ ] Animated plots showing algorithm progression (matplotlib.animation)
- [ ] LaTeX equations for all mathematical concepts
- [ ] Code cells with step-by-step execution
- [ ] Markdown explanations between code blocks
- [ ] Export to PDF for offline reading
- [ ] Binder-ready for cloud execution

---

## Testing Checklist

### Unit Tests ✅
- [x] `test_config.py` - Config validation and serialization
- [x] `test_data_structures.py` - PreintegratedIMUData
- [x] `test_ekf_slam.py` - EKF with preintegrated IMU only (simplified)
- [x] `test_swba_slam.py` - SWBA with preintegrated IMU only (simplified)
- [x] `test_srif_slam.py` - SRIF with preintegrated IMU only (simplified)
- [x] `test_imu_integration.py` - Preintegration logic

### Integration Tests ✅
- [x] End-to-end simulation with preintegrated data
- [x] Config loading and validation
- [x] Estimator switching between modes (simplified to preintegrated only)
- [x] Results comparison

### Regression Tests
- [x] Ensure existing functionality unchanged (adapted for simplified version)
- [x] Verify default behavior (now preintegrated IMU only)
- [x] Check backwards compatibility (tests adapted)

---

## Risk Mitigation

### Potential Issues
1. **Config Migration**: Existing config files may break
   - Solution: Add migration script or compatibility layer

2. **Test Failures**: Tests may depend on dataclass configs
   - Solution: Systematic test update with clear error messages

3. **Performance Regression**: Preintegration overhead
   - Solution: Lazy evaluation, caching, optional features

4. **Numerical Instability**: Accumulated errors in preintegration
   - Solution: Careful numerical implementation, regular resets

### Rollback Plan
- Keep feature branches separate
- Tag stable version before changes
- Document all API changes
- Maintain backwards compatibility flags

---

## Dependencies

### External Libraries
- `pydantic>=2.0` (already in use)
- `numpy` (already in use)
- No new dependencies required

### Internal Dependencies
- `src.common.config` (central config module)
- `src.common.data_structures` (data types)
- `src.estimation.imu_integration` (preintegration logic)
- All estimator modules

### Phase Dependencies
- **Phase 1** (Config Refactoring) - Independent, can be done first
- **Phase 2** (Preintegrated IMU) - Depends on Phase 1 for configs
- **Phase 3** (Keyframe Selection) - Can run parallel to Phase 2
- **Phase 4** (Integration) - Depends on both Phase 2 and Phase 3
- **Phase 5** (Validation) - Depends on all previous phases

**Recommended Order**: Phase 1 → (Phase 2 & 3 in parallel) → Phase 4 → Phase 5

---

## Success Criteria

### Phase 1 Success ✅
- [x] All tests pass with Pydantic configs
- [x] No duplicate config definitions
- [x] Config validation working
- [x] Serialization/deserialization functional

### Phase 2 Success ✅
- [x] Estimators accept both IMU data types (simplified to preintegrated only)
- [x] Preintegrated mode produces correct results
- [x] Preintegrated data correctly linked to keyframes
- [x] Performance improvement demonstrated
- [x] All tests pass

### Phase 3 Success ✅
- [x] Keyframe selection strategies working
- [x] Frames correctly marked as keyframes
- [x] Keyframe IDs properly assigned
- [x] Integration with estimators functional

### Phase 4 Success ✅
- [x] Keyframes and preintegrated IMU properly associated
- [x] Complete pipeline working end-to-end
- [x] Factor graph correctly constructed with keyframe constraints
- [x] Consistent results across different configurations

### Overall Success (Educational Focus)
- [ ] Clean, understandable code suitable for teaching
- [ ] Complete educational documentation with examples
- [ ] Working comparison between 3 estimation algorithms
- [ ] Professional KPI reports and visualizations
- [ ] All examples run successfully
- [ ] Minimal dependencies and simple setup
- [ ] Clear algorithm explanations with math
- [ ] Performance benchmarks documented
- [ ] HTML report generation working
- [ ] < 1000 lines per estimator (simplicity goal)