# GTSAM Integration & Codebase Simplification Specifications

## Overview
Migrate from custom SLAM estimators (EKF, SWBA, SRIF) to GTSAM-based implementations to improve robustness, performance, and maintainability while focusing on simulation capabilities.

## 1. Architecture Design

### 1.1 Core Components

```
src/estimation/
├── base_estimator.py          # Keep abstract interface
├── gtsam_base.py              # NEW: GTSAM base class
├── gtsam_ekf_estimator.py    # NEW: GTSAM EKF
├── gtsam_swba_estimator.py   # NEW: GTSAM SWBA  
├── cpp_binary_estimator.py   # Keep for external binaries
├── imu_integration.py         # Keep, useful for GTSAM
└── result_io.py              # Keep for I/O

TO BE REMOVED:
- ekf_slam.py
- swba_slam.py  
- srif_slam.py
```

### 1.2 Class Hierarchy

```python
BaseEstimator (base_estimator.py)
    ├── GtsamBaseEstimator (gtsam_base.py)
    │   ├── GtsamEkfEstimator (gtsam_ekf_estimator.py)
    │   └── GtsamSWBAEstimator (gtsam_swba_estimator.py)
    └── CppBinaryEstimator (cpp_binary_estimator.py)
```

## 2. Implementation Specifications

### 2.1 GtsamBaseEstimator

**File**: `src/estimation/gtsam_base.py`

**Purpose**: Common functionality for all GTSAM-based estimators

**Key Methods**:
```python
class GtsamBaseEstimator(BaseEstimator):
    def __init__(self, config: BaseConfig):
        """Initialize factor graph and solver"""
        
    def initialize(self, initial_pose: Pose):
        """Add prior factor for initial pose"""
        
    def add_imu_factor(self, preintegrated: PreintegratedIMUData):
        """Add preintegrated IMU factor between poses"""
        
    def add_vision_factor(self, observation: CameraObservation, landmark: Landmark):
        """Add projection factor for landmark observation"""
        
    def optimize(self) -> EstimatorResult:
        """Run optimization and extract results"""
        
    def get_result(self) -> EstimatorResult:
        """Convert GTSAM values to EstimatorResult format"""
```

**Required GTSAM Components**:
- `gtsam.NonlinearFactorGraph`: Factor graph container
- `gtsam.Values`: Variable values container
- `gtsam.Symbol`: Variable naming (X for poses, L for landmarks, V for velocities, B for biases)

### 2.2 GtsamEkfEstimator

**File**: `src/estimation/gtsam_ekf_estimator.py`

**Purpose**: Extended Kalman Filter using GTSAM's incremental solver

**Key Features**:
- Use `gtsam.ISAM2` for incremental updates
- Process one keyframe at a time
- Maintain fixed computation per update

**Implementation**:
```python
class GtsamEkfEstimator(GtsamBaseEstimator):
    def __init__(self, config: EKFConfig):
        super().__init__(config)
        self.isam2 = gtsam.ISAM2(gtsam.ISAM2Params())
        self.current_pose_id = 0
        
    def predict(self, preintegrated_imu: PreintegratedIMUData):
        """Add IMU factor and predict next pose"""
        # Create IMU factor between X(i) and X(i+1)
        # Update ISAM2 incrementally
        
    def update(self, frame: CameraFrame, landmarks: Map):
        """Add vision factors for current pose"""
        # Add projection factors for observed landmarks
        # Update ISAM2 with new measurements
```

**GTSAM Factors to Use**:
- `PriorFactorPose3`: Initial pose constraint
- `ImuFactor` or custom preintegrated factor
- `GenericProjectionFactor`: Camera measurements
- `PriorFactorVector`: Bias priors

### 2.3 GtsamSWBAEstimator

**File**: `src/estimation/gtsam_swba_estimator.py`

**Purpose**: Sliding Window Bundle Adjustment using GTSAM

**Key Features**:
- Maintain fixed window of keyframes
- Use `gtsam.LevenbergMarquardtOptimizer` for batch optimization
- Marginalize old states using `gtsam.ISAM2` or custom marginalization

**Implementation**:
```python
class GtsamSWBAEstimator(GtsamBaseEstimator):
    def __init__(self, config: SWBAConfig):
        super().__init__(config)
        self.window_size = config.window_size
        self.active_poses = []  # Poses in current window
        
    def predict(self, preintegrated_imu: PreintegratedIMUData):
        """Add new pose to window"""
        # Add pose to window
        # If window full, marginalize oldest
        
    def update(self, frame: CameraFrame, landmarks: Map):
        """Optimize current window"""
        # Build factor graph for window
        # Run batch optimization
        # Extract and store results
        
    def marginalize_old_states(self):
        """Remove old poses from window"""
        # Convert old states to prior factors
        # Remove from active window
```

**GTSAM Components**:
- `LevenbergMarquardtOptimizer`: Batch optimization
- `MarginalizeLinearFactorGraph`: For marginalization
- `FixedLagSmoother`: Alternative sliding window approach

## 3. Data Interface Specifications

### 3.1 Input Data Compatibility

Maintain compatibility with existing data structures:
- `PreintegratedIMUData`: Direct use in GTSAM IMU factors
- `CameraFrame`: Convert observations to GTSAM projection factors
- `Map`/`Landmark`: Use for 3D point positions

### 3.2 Output Format

Ensure `EstimatorResult` format remains unchanged:
```python
EstimatorResult:
    - trajectory: Trajectory with poses
    - landmarks: Estimated landmark positions
    - states: Internal estimator states
    - runtime_ms: Execution time
    - metadata: Additional information
```

## 4. Integration Specifications

### 4.1 CLI Integration

Update `tools/cli.py` to support new estimators:

```python
ESTIMATOR_MAP = {
    'gtsam-ekf': GtsamEkfEstimator,
    'gtsam-swba': GtsamSWBAEstimator,
    'cpp': CppBinaryEstimator  # Keep for compatibility
}
```

### 4.2 E2E Command Support

Modify `run.sh` to handle:
```bash
# Single estimator run
./run.sh slam gtsam-ekf --input simulation.json --output results/

# E2E simple test
./run.sh e2e/e2e-simple --estimator gtsam-ekf

# E2E comparison
./run.sh e2e/e2e-simple --estimator gtsam-swba
```

### 4.3 Configuration Files

Create new config files:
- `config/estimators/gtsam_ekf.yaml`
- `config/estimators/gtsam_swba.yaml`

Example configuration:
```yaml
# config/estimators/gtsam_ekf.yaml
type: gtsam-ekf
parameters:
  relinearize_threshold: 0.1
  relinearize_skip: 10
  cache_linearized_factors: true
  enable_partial_relinearization: false
  
noise_models:
  prior_pose: [0.1, 0.1, 0.1, 0.05, 0.05, 0.05]  # x,y,z,roll,pitch,yaw
  prior_velocity: [0.1, 0.1, 0.1]
  prior_bias: [0.01, 0.01, 0.01, 0.001, 0.001, 0.001]
  
imu:
  preintegration_type: "combined"  # or "sequential"
  coriolis_effect: false
  
vision:
  robust_kernel: "huber"
  huber_parameter: 1.345
  projection_noise: [1.0, 1.0]  # pixels
```

## 5. Testing Specifications

### 5.1 Test Structure

```
tests/
├── test_gtsam_estimators.py      # NEW: Main GTSAM tests
├── test_gtsam_ekf.py             # NEW: EKF-specific tests
├── test_gtsam_swba.py            # NEW: SWBA-specific tests
├── gtsam-comparison/             # Reference implementation
│   ├── test_estimator_comparison.py
│   └── test_preintegration.py
└── test_complete_pipeline.py     # Update for GTSAM
```

### 5.2 Test Cases

**Basic Functionality**:
```python
def test_gtsam_ekf_initialization():
    """Test GTSAM EKF can be initialized"""
    
def test_gtsam_ekf_single_prediction():
    """Test single IMU prediction step"""
    
def test_gtsam_ekf_single_update():
    """Test single camera update"""
```

**Integration Tests**:
```python
def test_gtsam_ekf_full_trajectory():
    """Test complete trajectory estimation"""
    
def test_gtsam_swba_window_management():
    """Test sliding window maintains size"""
    
def test_gtsam_marginalization():
    """Test old states are properly marginalized"""
```

**Comparison Tests** (before removing legacy):
```python
def test_gtsam_vs_legacy_ekf():
    """Compare GTSAM EKF with legacy implementation"""
    # Run same data through both
    # Assert trajectories are similar (not identical due to numerics)
```

### 5.3 Performance Benchmarks

Track key metrics:
- Runtime per keyframe
- Memory usage over time
- Accuracy (ATE, RPE)
- Convergence rate

## 6. Migration Plan

### Phase 1: Setup (Days 1-2)
- [ ] Install GTSAM dependencies
- [ ] Create project structure
- [ ] Implement GtsamBaseEstimator skeleton
- [ ] Set up basic tests

### Phase 2: GtsamEkfEstimator (Days 3-5)
- [ ] Implement core EKF functionality
- [ ] Add IMU factor support
- [ ] Add vision factor support
- [ ] Pass basic tests

### Phase 3: GtsamSWBAEstimator (Days 6-8)
- [ ] Implement sliding window logic
- [ ] Add marginalization
- [ ] Optimize window management
- [ ] Pass window tests

### Phase 4: Integration (Days 9-10)
- [ ] Update CLI commands
- [ ] Integrate with e2e-simple
- [ ] Update configuration files
- [ ] Run full pipeline tests

### Phase 5: Cleanup (Days 11-12)
- [ ] Remove legacy estimators
- [ ] Clean up old tests
- [ ] Update documentation
- [ ] Performance optimization

### Phase 6: Validation (Days 13-14)
- [ ] Run comparison tests
- [ ] Benchmark performance
- [ ] Fix any regressions
- [ ] Final documentation

## 7. Dependencies

### Required Packages
```txt
# Add to requirements.txt
gtsam>=4.2.0
```

### Installation Instructions
```bash
# Ubuntu/Debian
sudo apt-get install libboost-all-dev
pip install gtsam

# macOS
brew install boost
pip install gtsam

# Alternative: build from source
git clone https://github.com/borglab/gtsam.git
cd gtsam
mkdir build && cd build
cmake .. -DGTSAM_BUILD_PYTHON=ON
make -j4
make install
```

## 8. Risk Mitigation

| Risk | Mitigation Strategy |
|------|-------------------|
| GTSAM installation issues | Provide Docker container with pre-installed GTSAM |
| Performance regression | Profile early, optimize factor graph structure |
| Numerical differences | Set tolerances appropriately in tests |
| API breaking changes | Keep adapter layer in GtsamBaseEstimator |
| Memory leaks | Use GTSAM's built-in memory management |

## 9. Success Criteria

### Functional Requirements
- [ ] GtsamEkfEstimator processes all test trajectories
- [ ] GtsamSWBAEstimator maintains fixed window size
- [ ] Both estimators integrate with e2e-simple
- [ ] All existing tests pass or have GTSAM equivalents

### Performance Requirements
- [ ] Runtime within 2x of legacy implementations
- [ ] Memory usage < 1GB for standard test cases
- [ ] ATE accuracy comparable to legacy (< 10% difference)

### Code Quality
- [ ] 90%+ test coverage for new code
- [ ] No memory leaks (valgrind clean)
- [ ] Documentation for all public methods
- [ ] Type hints for all functions

## 10. Example Implementation

### Minimal Working Example
```python
import gtsam
import numpy as np
from src.estimation.base_estimator import BaseEstimator, EstimatorResult

class GtsamEkfEstimator(BaseEstimator):
    def __init__(self, config):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_values = gtsam.Values()
        self.isam2 = gtsam.ISAM2()
        self.pose_count = 0
        
    def initialize(self, initial_pose):
        # Create pose
        pose = gtsam.Pose3(
            gtsam.Rot3(initial_pose.rotation_matrix),
            initial_pose.position
        )
        
        # Add prior
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05])
        )
        self.graph.add(gtsam.PriorFactorPose3(
            gtsam.symbol('x', 0), pose, prior_noise
        ))
        
        # Initialize values
        self.initial_values.insert(gtsam.symbol('x', 0), pose)
        
        # Update ISAM2
        self.isam2.update(self.graph, self.initial_values)
        self.graph.resize(0)
        self.initial_values.clear()
        
    def predict(self, preintegrated_imu):
        # Create IMU factor
        # ... implementation ...
        self.pose_count += 1
        
    def get_result(self):
        # Extract current estimate
        values = self.isam2.calculateEstimate()
        
        # Convert to EstimatorResult
        trajectory = self._extract_trajectory(values)
        landmarks = self._extract_landmarks(values)
        
        return EstimatorResult(
            trajectory=trajectory,
            landmarks=landmarks,
            runtime_ms=0.0,
            metadata={}
        )
```

## 11. Documentation Requirements

### Code Documentation
- Docstrings for all classes and methods
- Type hints for all parameters and returns
- Example usage in docstrings

### User Documentation
- Migration guide from legacy to GTSAM
- Configuration parameter explanations
- Troubleshooting guide

### Developer Documentation
- GTSAM factor graph design decisions
- Performance optimization tips
- Extension points for new factors

## 12. Deliverables

1. **Code**:
   - `gtsam_base.py`
   - `gtsam_ekf_estimator.py`
   - `gtsam_swba_estimator.py`
   - Updated `tools/cli.py`

2. **Tests**:
   - `test_gtsam_estimators.py`
   - `test_gtsam_ekf.py`
   - `test_gtsam_swba.py`

3. **Configuration**:
   - `config/estimators/gtsam_ekf.yaml`
   - `config/estimators/gtsam_swba.yaml`

4. **Documentation**:
   - `docs/gtsam_integration.md`
   - Updated `README.md`
   - Migration guide

5. **Scripts**:
   - Updated `run.sh` with GTSAM commands
   - Performance benchmark script
   - Comparison script (legacy vs GTSAM)