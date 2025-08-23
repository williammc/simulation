# SimulationIO - C++ JSON I/O Library

A header-only C++ library for reading and writing simulation data in JSON format, compatible with the Python `SimulationData` structure.

## Features

- Header-only implementation for easy integration
- Full compatibility with Python JSON schema
- Support for:
  - Trajectory states with position and quaternion
  - IMU measurements (accelerometer and gyroscope)
  - Camera observations and frames
  - Landmarks with optional descriptors
  - Camera and IMU calibration data
  - Metadata and configuration

## Requirements

- C++17 or later
- CMake 3.14 or later
- nlohmann/json (automatically fetched by CMake)

## Building

```bash
mkdir build
cd build
cmake ..
make
```

To run tests:
```bash
make test
# or
ctest
```

## Usage

### Basic Example

```cpp
#include <simulation_io/json_io.hpp>

using namespace simulation_io;

// Create simulation data
SimulationData data;
data.metadata.version = "1.0";
data.metadata.trajectory_type = "circular";

// Add trajectory states
TrajectoryState state;
state.timestamp = 0.0;
state.position = Vector3(1.0, 2.0, 3.0);
state.quaternion = Quaternion(1.0, 0.0, 0.0, 0.0);
data.trajectory.push_back(state);

// Save to JSON
JsonIO::save(data, "output.json");

// Load from JSON
SimulationData loaded = JsonIO::load("output.json");
```

### CMake Integration

Add to your CMakeLists.txt:

```cmake
add_subdirectory(path/to/cpp_io)
target_link_libraries(your_target PRIVATE simulation_io)
```

Or use FetchContent:

```cmake
include(FetchContent)
FetchContent_Declare(
    simulation_io
    SOURCE_DIR /path/to/cpp_io
)
FetchContent_MakeAvailable(simulation_io)

target_link_libraries(your_target PRIVATE simulation_io)
```

## Data Structures

### Core Types
- `Vector3`: 3D vector
- `Quaternion`: Unit quaternion (w, x, y, z)
- `Matrix3x3`: 3x3 rotation matrix
- `Matrix4x4`: 4x4 transformation matrix

### Simulation Data
- `TrajectoryState`: Timestamped pose with optional velocity
- `Landmark`: 3D point with ID and optional descriptor
- `IMUMeasurement`: Accelerometer and gyroscope readings
- `CameraObservation`: 2D pixel observation of a landmark
- `CameraFrame`: Collection of observations at a timestamp

### Calibration
- `CameraCalibration`: Camera intrinsics and extrinsics
- `IMUCalibration`: IMU noise parameters and sampling rate

## Testing

Run the test executable to verify JSON I/O functionality:

```bash
./build/test/test_json_io
```

This will:
1. Create sample simulation data
2. Save it to JSON
3. Load it back
4. Verify data integrity
5. Display sample values

## License

Same as the parent project.