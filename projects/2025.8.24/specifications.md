# Configuration System Modularization - Technical Specifications

## Architecture Overview

### System Components
```
┌─────────────────────────────────────────────────────┐
│                  Main Config File                    │
│                 (scenario.yaml)                      │
└──────────────────┬──────────────────────────────────┘
                   │ includes/references
                   ▼
┌──────────────────────────────────────────────────────┐
│            ConfigLoader (Enhanced)                   │
│  - YAML parsing with custom tags                     │
│  - File inclusion resolution                         │
│  - Configuration merging                             │
│  - Schema validation                                 │
└──────────┬───────────────────────────┬───────────────┘
           │                           │
           ▼                           ▼
┌──────────────────┐        ┌──────────────────────────┐
│ Component Configs│        │   Configuration Cache    │
│ - trajectories/  │        │  - Loaded configs        │
│ - estimators/    │        │  - Resolved paths        │
│ - noises/        │        │  - Validation results   │
│ - imus/          │        └──────────────────────────┘
│ - cameras/       │
└──────────────────┘
```

## Implementation Details

### Current Architecture Analysis

The codebase uses Pydantic models in `src/common/config.py` for configuration:
- **SimulationConfig**: Main simulation configuration with trajectory, cameras, IMUs, environment
- **EstimatorConfig**: Estimator-specific configs (EKF, SWBA, SRIF)
- Simple YAML loading via `yaml.safe_load()` - no include mechanism exists
- Tools like `simulate.py` and `slam.py` load configs directly

### 1. Configuration File Structure

#### Main Scenario Configuration
```yaml
# config/scenarios/vio_scenario.yaml
name: "Visual-Inertial Odometry Scenario"
version: "1.0"

components:
  trajectory: !include ../trajectories/figure8.yaml
  imu: !include ../imus/mpu6050.yaml
  camera: !include ../cameras/pinhole_640x480.yaml
  noise: !include ../noises/realistic.yaml
  estimator: !include ../estimators/ekf.yaml

# Component-specific overrides
overrides:
  trajectory:
    duration: 120.0
  camera:
    fps: 30
```

#### Component Configuration Examples

**Trajectory** (`config/trajectories/figure8.yaml`):
```yaml
type: "figure8"
parameters:
  center: [0, 0, 0]
  amplitude_x: 10.0
  amplitude_y: 5.0
  frequency: 0.1
  duration: 60.0
```

**Estimator** (`config/estimators/cpp_binary.yaml`):
```yaml
type: "cpp_binary"
parameters:
  executable: "cpp_estimation/build/estimator"
  input_format: "json"
  output_format: "json"
  args:
    - "--config"
    - "cpp_config.yaml"
  timeout: 300  # seconds
```

### 2. ConfigLoader Enhancement

**Note**: Currently there is no `config_loader.py` file. We need to create it in `src/utils/`.

#### Class Structure
```python
# src/utils/config_loader.py (NEW FILE)

class ConfigLoader:
    def __init__(self, base_path: Path = None):
        self.base_path = base_path or Path.cwd()
        self.cache = {}
        self.yaml_loader = self._create_yaml_loader()
    
    def load_config(self, config_path: Union[str, Path]) -> Dict:
        """Load configuration with includes resolved"""
        
    def _resolve_includes(self, config: Dict, current_path: Path) -> Dict:
        """Recursively resolve !include tags"""
        
    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Deep merge configurations with override precedence"""
        
    def validate_config(self, config: Dict, schema: Dict = None) -> bool:
        """Validate configuration against schema"""
```

#### Custom YAML Tags
```python
def include_constructor(loader, node):
    """Handle !include tag for file inclusion"""
    file_path = loader.construct_scalar(node)
    resolved_path = resolve_path(file_path, loader.current_file)
    return load_yaml_file(resolved_path)

def env_constructor(loader, node):
    """Handle !env tag for environment variables"""
    env_var = loader.construct_scalar(node)
    return os.environ.get(env_var, "")

yaml.add_constructor('!include', include_constructor)
yaml.add_constructor('!env', env_constructor)
```

### 3. Configuration Merging Strategy

#### Merge Rules
1. **Scalar values**: Override replaces base
2. **Lists**: Override replaces base (no concatenation by default)
3. **Dictionaries**: Deep merge with override precedence
4. **Special markers**:
   - `~append`: Append to existing list
   - `~merge`: Deep merge dictionaries
   - `~replace`: Full replacement (default)

#### Example
```python
def deep_merge(base: Dict, override: Dict) -> Dict:
    result = base.copy()
    for key, value in override.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        else:
            result[key] = value
    return result
```

### 4. Schema Validation

#### Schema Definition
```yaml
# config/schemas/scenario_schema.yaml
type: object
required: [name, components]
properties:
  name:
    type: string
  version:
    type: string
    pattern: "^\\d+\\.\\d+$"
  components:
    type: object
    required: [trajectory, estimator]
    properties:
      trajectory:
        $ref: "#/definitions/trajectory"
      imu:
        $ref: "#/definitions/imu"
      camera:
        $ref: "#/definitions/camera"
      noise:
        $ref: "#/definitions/noise"
      estimator:
        $ref: "#/definitions/estimator"
```

### 5. External Binary Integration

**Note**: Currently no external binary support exists. The estimators (EKF, SWBA, SRIF) are all Python implementations.

#### Process Management
```python
class CppBinaryEstimator:
    def __init__(self, config: Dict):
        self.executable = Path(config['executable'])
        self.timeout = config.get('timeout', 300)
        
    def run(self, input_data: Dict) -> Dict:
        # Write input JSON
        input_file = self._write_input(input_data)
        
        # Execute binary
        result = subprocess.run(
            [self.executable, '--input', input_file],
            capture_output=True,
            timeout=self.timeout
        )
        
        # Read output JSON
        return self._read_output(result.stdout)
```

### 6. Backwards Compatibility

#### Legacy Config Detection
```python
def is_legacy_config(config: Dict) -> bool:
    """Detect monolithic configuration format"""
    return not 'components' in config and 'trajectory' in config

def convert_legacy_config(config: Dict) -> Dict:
    """Convert legacy format to new modular format"""
    return {
        'name': config.get('name', 'Legacy Scenario'),
        'components': {
            'trajectory': config.get('trajectory'),
            'imu': config.get('imu'),
            'camera': config.get('camera'),
            'noise': config.get('noise'),
            'estimator': config.get('estimator')
        }
    }
```

## Integration with Existing Code

### Affected Files
1. **src/common/config.py**: Modify `load_simulation_config()` and `load_estimator_config()` functions (lines 536-549)
2. **tools/simulate.py**: Update lines 83-94 to use new ConfigLoader
3. **tools/slam.py**: Update lines 59-72 to use new ConfigLoader  
4. **tools/e2e_pipeline.py**: Update config loading sections
5. **tools/e2e_orchestrator.py**: Update config loading sections

### New Files
1. **src/utils/config_loader.py**: Main ConfigLoader implementation
2. **config/trajectories/*.yaml**: Trajectory component configs
3. **config/estimators/*.yaml**: Estimator component configs
4. **config/noises/*.yaml**: Noise model configs
5. **config/imus/*.yaml**: IMU sensor configs
6. **config/cameras/*.yaml**: Camera sensor configs
7. **config/scenarios/*.yaml**: Main scenario configs

## API Specifications

### Public Interface
```python
# Primary API
config_loader = ConfigLoader(base_path="config/")
config = config_loader.load_config("scenarios/vio_scenario.yaml")

# Validation
is_valid = config_loader.validate_config(config)

# Component access
trajectory_config = config['components']['trajectory']
estimator_config = config['components']['estimator']
```

### CLI Integration
```bash
# Load with modular config
python tools/simulate.py --config config/scenarios/vio_scenario.yaml

# Override specific component
python tools/simulate.py --config config/scenarios/vio_scenario.yaml \
                        --trajectory config/trajectories/circle.yaml
```

## Error Handling

### Error Types
1. **FileNotFoundError**: Component file doesn't exist
2. **CircularIncludeError**: Circular dependency in includes
3. **SchemaValidationError**: Configuration doesn't match schema
4. **MergeConflictError**: Incompatible configuration merge

### Error Messages
```python
class ConfigError(Exception):
    """Base configuration error"""
    
class IncludeError(ConfigError):
    """Error resolving include"""
    def __init__(self, path: Path, parent: Path):
        super().__init__(
            f"Cannot include '{path}' from '{parent}': File not found"
        )
```

## Testing Requirements

### Unit Tests
- Configuration loading with includes
- Circular dependency detection
- Schema validation
- Legacy format conversion
- Merge operations

### Integration Tests
- End-to-end scenario loading
- Tool compatibility
- External binary execution
- Performance benchmarks

## Performance Specifications

### Metrics
- Configuration load time: < 100ms
- Include resolution: < 10ms per file
- Validation overhead: < 20ms
- Cache hit rate: > 90% for repeated loads

### Optimization Strategies
1. **Lazy loading**: Load components only when accessed
2. **Caching**: Cache resolved configurations
3. **Parallel loading**: Load independent components concurrently
4. **Pre-compilation**: Pre-process frequently used configs