
## 1. Duplicate Config Declaration Refactoring

### Issue
- Duplicate config declaration for EKFConfig, SWBAConfig, SRIFConfig
    - Pydantic models in `src/common/config.py` (better validation, serialization)
    - Dataclasses in `src/estimation/*_slam.py` (local, redundant)

### Solution
=> Refactor to use only Pydantic configs from `src/common/config.py`

### Implementation Steps
1. **Remove dataclass configs** from:
   - `src/estimation/ekf_slam.py` (lines 140-160)
   - `src/estimation/swba_slam.py` (lines 101-125)
   - `src/estimation/srif_slam.py` (lines 148-165)

2. **Update imports** in estimator files:
   ```python
   from src.common.config import EKFConfig, SWBAConfig, SRIFConfig
   ```

3. **Modify estimator constructors** to work with Pydantic models
   - Access fields with dot notation (same as dataclass)
   - Benefit from Pydantic's validation

4. **Update any config instantiation** to use Pydantic constructors


## 2. Preintegrated IMU Data Support

### Issue
- Need to test estimators with preintegrated IMU data as optional input
- Current: Estimators only accept raw IMU measurements
- Impacts: EKF, SWBA, SRIF prediction/propagation steps

### Solution
=> Add support for both raw and preintegrated IMU data

### Implementation Design

1. **Create data structure** in `src/common/data_structures.py`:
   ```python
   @dataclass
   class PreintegratedIMUData:
       delta_position: np.ndarray      # Relative position change
       delta_velocity: np.ndarray      # Relative velocity change  
       delta_rotation: np.ndarray      # Relative rotation (quaternion/matrix)
       covariance: np.ndarray          # Uncertainty (15x15)
       jacobian: Optional[np.ndarray]  # Jacobian w.r.t biases
       dt: float                       # Total time interval
       from_keyframe_id: int           # Source keyframe ID
       to_keyframe_id: int             # Target keyframe ID
       source_measurements: Optional[List[IMUMeasurement]] = None
   ```

2. **Update Pydantic configs** to include mode selection:
   ```python
   class EKFConfig(BaseModel):
       use_preintegrated_imu: bool = False
       # ... existing fields
   
   class SWBAConfig(BaseModel):
       use_preintegrated_imu: bool = False
       # ... existing fields
   
   class SRIFConfig(BaseModel):
       use_preintegrated_imu: bool = False
       # ... existing fields
   ```

3. **Modify estimator prediction methods**:
   ```python
   def predict(self, 
              imu_data: Union[List[IMUMeasurement], PreintegratedIMUData],
              dt: float = None) -> None:
       if isinstance(imu_data, PreintegratedIMUData):
           self._predict_preintegrated(imu_data)
       else:
           self._predict_raw(imu_data, dt)
   ```

4. **Add preintegration support** to simulation:
   - Option to precompute IMU integration between keyframes
   - Store preintegrated results alongside raw measurements
   - Configure via simulation config

### Keyframe-IMU Preintegration Relationship

**Key Concept**: Preintegrated IMU data represents the accumulated IMU measurements between consecutive keyframes.

```
Keyframe_i --> [IMU measurements] --> Keyframe_j
           \                          /
            ---> PreintegratedIMUData
```

- Each `PreintegratedIMUData` connects two keyframes (from_keyframe_id, to_keyframe_id)
- Stored in the target keyframe's `preintegrated_imu` field
- Contains all IMU integration between the keyframe pair
- Enables efficient factor graph optimization in SLAM

### Benefits
- **Testing flexibility**: Can test with precomputed values for validation
- **Performance**: Reduced computational cost when using preintegrated data
- **Modularity**: Clean separation between raw and preintegrated processing
- **Backward compatibility**: Existing code continues to work with raw measurements
- **Keyframe association**: Clear linkage between visual and inertial constraints

### Impact on Estimators

| Estimator | Raw IMU Processing | Preintegrated Processing |
|-----------|-------------------|-------------------------|
| **EKF** | Integrate each measurement | Apply delta directly to state |
| **SWBA** | Integrate within window | Use as factor constraints |
| **SRIF** | Sequential integration | Information matrix update |

### Testing Strategy
1. Generate ground truth trajectory
2. Simulate raw IMU measurements
3. Preintegrate measurements offline
4. Compare estimator outputs using both modes
5. Validate consistency of results


## 3. Keyframe Selection Configuration

### Issue
- Need systematic keyframe selection during simulation
- Current: SWBA has runtime keyframe selection based on motion thresholds
- Missing: Configuration for image-based keyframe selection (e.g., every N frames)
- Need to mark frames as keyframes in simulation data

### Solution
=> Add keyframe selection strategy to configuration and data structures

### Implementation Design

1. **Add keyframe selection config** in `src/common/config.py`:
   ```python
   class KeyframeSelectionConfig(BaseModel):
       """Configuration for keyframe selection strategies."""
       strategy: Literal["fixed_interval", "motion_based", "hybrid"] = "fixed_interval"
       
       # Fixed interval strategy
       frame_interval: int = Field(20, ge=1, description="Select keyframe every N frames")
       
       # Motion-based strategy (for runtime selection)
       translation_threshold: float = Field(0.5, gt=0, description="Translation threshold (m)")
       rotation_threshold: float = Field(0.3, gt=0, description="Rotation threshold (rad)")
       time_threshold: float = Field(0.5, gt=0, description="Time threshold (s)")
       
       # Hybrid strategy uses both fixed interval and motion thresholds
       force_keyframe_interval: int = Field(50, ge=1, description="Force keyframe after N frames")
   ```

2. **Update CameraFrame** in `src/common/data_structures.py`:
   ```python
   @dataclass
   class CameraFrame:
       timestamp: float
       camera_id: str
       observations: List[CameraObservation]
       image_path: Optional[str] = None
       is_keyframe: bool = False  # NEW: Mark as keyframe
       keyframe_id: Optional[int] = None  # NEW: Keyframe index
       preintegrated_imu: Optional[PreintegratedIMUData] = None  # NEW: IMU between keyframes
   ```

3. **Update SimulationConfig** to include keyframe selection:
   ```python
   class SimulationConfig(BaseModel):
       # ... existing fields
       keyframe_selection: KeyframeSelectionConfig = Field(
           default_factory=KeyframeSelectionConfig,
           description="Keyframe selection configuration"
       )
   ```

4. **Update estimator configs** to use keyframe settings:
   ```python
   class EKFConfig(BaseModel):
       use_keyframes_only: bool = False  # Process only keyframes
       # ... existing fields
   
   class SWBAConfig(BaseModel):
       use_simulation_keyframes: bool = True  # Use pre-selected keyframes
       override_keyframe_selection: Optional[KeyframeSelectionConfig] = None
       # ... existing fields
   
   class SRIFConfig(BaseModel):
       use_keyframes_only: bool = False
       # ... existing fields
   ```

5. **Add keyframe selector** in simulation pipeline:
   ```python
   class KeyframeSelector:
       def __init__(self, config: KeyframeSelectionConfig):
           self.config = config
           self.frame_count = 0
           self.last_keyframe_pose = None
           self.last_keyframe_time = None
       
       def is_keyframe(self, frame: CameraFrame, pose: Pose) -> bool:
           """Determine if frame should be a keyframe."""
           if self.config.strategy == "fixed_interval":
               return self.frame_count % self.config.frame_interval == 0
           elif self.config.strategy == "motion_based":
               return self._check_motion_thresholds(pose)
           elif self.config.strategy == "hybrid":
               return self._check_hybrid_criteria(pose)
   ```

### Benefits
- **Simulation control**: Pre-select keyframes during data generation
- **Consistency**: All estimators can use same keyframe selection
- **Flexibility**: Multiple selection strategies available
- **Testing**: Can compare different keyframe selection approaches
- **Reproducibility**: Keyframe selection is deterministic and configurable

### Impact on Components

| Component | Changes Required |
|-----------|-----------------|
| **Simulation** | Generate and mark keyframes during data creation |
| **Data Structures** | Add keyframe flags to CameraFrame |
| **Estimators** | Option to use pre-selected keyframes |
| **Visualization** | Highlight keyframes in plots |
| **Evaluation** | Metrics for keyframe-only trajectories |

### Use Cases
1. **Dense reconstruction**: Every frame is keyframe (interval=1)
2. **Efficient SLAM**: Sparse keyframes (interval=20-30)
3. **Motion-adaptive**: More keyframes during fast motion
4. **Benchmark testing**: Fixed keyframes for fair comparison
