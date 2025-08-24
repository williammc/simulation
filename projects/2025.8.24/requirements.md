# Configuration System Modularization - Requirements Document

## Project Overview
Enhance the simulation framework's configuration system to support modular, reusable configuration components without modifying existing simulation, sensor, or estimation code.

## Business Objectives
- **Reduce configuration duplication** across simulation scenarios
- **Accelerate scenario creation** through reusable components
- **Simplify maintenance** by centralizing common configurations
- **Enable rapid prototyping** of different sensor/estimator combinations

## Functional Requirements

### 1. Component Modularization
The system shall support separate configuration files for:
- **Trajectories** (`config/trajectories/`): Define motion paths (circle, figure8, spiral, etc.)
- **Estimators** (`config/estimators/`): Configure estimation algorithms (EKF, SWBA, SRIF, external binaries)
- **Noise Models** (`config/noises/`): Define sensor noise characteristics
- **IMU Sensors** (`config/imus/`): Specify IMU hardware configurations
- **Camera Sensors** (`config/cameras/`): Define camera intrinsics and properties

### 2. Configuration Composition
- Support referencing external configuration files from main scenario configs
- Enable configuration inheritance and overrides
- Maintain backward compatibility with existing monolithic configs

### 3. External Estimator Support
- Support `cpp_binary` estimator type that:
  - Accepts simulation data via JSON input file
  - Executes external binary (e.g., `cpp_estimation/examples/read_python_data.cpp`)
  - Reads estimation results from JSON output file
  - Integrates seamlessly with Python analysis pipeline

### 4. Tool Integration
- Configuration loader must work with existing tools in `tools/` directory
- Support both CLI and programmatic configuration loading

## Non-Functional Requirements

### Performance
- Configuration parsing overhead < 100ms for typical scenarios
- No runtime performance impact on simulation execution

### Maintainability
- Clear error messages for configuration issues
- Validation of cross-component dependencies
- Documentation for configuration schema

### Compatibility
- Python 3.8+ support
- YAML 1.2 compliance
- Cross-platform (Linux, macOS, Windows)

## Constraints
- **No modifications** to existing sensor models (camera, IMU)
- **No modifications** to simulation core logic
- **No modifications** to estimator implementations
- Changes **limited to** configuration loading mechanism only
- Must maintain compatibility with existing configuration files

## Success Criteria
1. Ability to compose a simulation from modular config components
2. Reduction in configuration duplication by >70%
3. Support for at least 3 different trajectory types
4. Support for at least 4 different estimator configurations
5. Successful integration of external C++ estimator binary
6. All existing tools continue to function without modification

## Out of Scope
- GUI configuration editor
- Runtime configuration changes
- Configuration versioning/migration tools
- Performance optimizations beyond configuration loading