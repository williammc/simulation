# Simulation Module Documentation

## Overview

The simulation module (`src/simulation/`) provides comprehensive tools for generating synthetic SLAM data including trajectories, sensor measurements, and environmental features. This enables controlled testing and validation of SLAM algorithms.

## Table of Contents
- [Architecture](#architecture)
- [Trajectory Generation](#trajectory-generation)
- [Sensor Simulation](#sensor-simulation)
- [Landmark Generation](#landmark-generation)
- [Simulation Pipeline](#simulation-pipeline)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)

## Architecture

```
src/simulation/
├── __init__.py
├── simulator.py              # Main simulation orchestrator
├── trajectory_generator.py   # Generate robot trajectories
├── imu_simulator.py         # Simulate IMU measurements
├── camera_simulator.py      # Simulate camera observations
├── landmark_generator.py    # Generate 3D landmarks
└── sensor_sync.py          # Multi-sensor synchronization
```

## Trajectory Generation

### Available Trajectories

The system supports multiple trajectory types (`trajectory_generator.py`):

#### 1. Circle Trajectory
```python
params = {
    'radius': 2.0,      # Circle radius (m)
    'height': 1.5,      # Constant height (m)
    'duration': 5.0,    # Time for full circle (s)
    'rate': 200.0       # Output rate (Hz)
}
trajectory = generate_trajectory('circle', params)
```

#### 2. Figure-8 Trajectory
```python
params = {
    'scale': 3.0,       # Overall size scaling
    'duration': 10.0,   # Total duration (s)
    'rate': 200.0       # Output rate (Hz)
}
trajectory = generate_trajectory('figure8', params)
```

#### 3. Line Trajectory
```python
params = {
    'start': [0, 0, 0],     # Start position
    'end': [10, 0, 0],      # End position
    'duration': 5.0,        # Travel time (s)
    'rate': 200.0           # Output rate (Hz)
}
trajectory = generate_trajectory('line', params)
```

#### 4. Random Walk
```python
params = {
    'num_waypoints': 10,    # Number of random waypoints
    'bounds': [-5, 5],      # Spatial bounds for x,y,z
    'duration': 20.0,       # Total duration (s)
    'rate': 200.0           # Output rate (Hz)
}
trajectory = generate_trajectory('random', params)
```

### Trajectory State

Each trajectory point contains:
```python
class TrajectoryState:
    pose: Pose                    # Position and orientation
    linear_velocity: np.ndarray   # Linear velocity (m/s)
    angular_velocity: np.ndarray  # Angular velocity (rad/s)
    linear_acceleration: np.ndarray  # Linear acceleration (m/s²)
```

### Interpolation

Trajectories support interpolation for sensor synchronization:
```python
interpolator = TrajectoryInterpolator(trajectory)

# Get state at any timestamp
state = interpolator.get_state_at_time(t=2.5)

# Get states at multiple times
times = np.linspace(0, 10, 100)
states = interpolator.get_states_at_times(times)
```

## Sensor Simulation

### IMU Simulator

The IMU simulator (`imu_simulator.py`) generates realistic inertial measurements:

```python
class IMUSimulator:
    def __init__(self, calibration: IMUCalibration):
        self.calibration = calibration
        self.bias = self.initialize_bias()
    
    def simulate_measurement(self, true_state: State) -> IMUMeasurement:
        # Compute specific force (acceleration - gravity)
        specific_force = self.compute_specific_force(true_state)
        
        # Add noise and bias
        accel = specific_force + self.bias.accel + noise
        gyro = true_state.angular_velocity + self.bias.gyro + noise
        
        return IMUMeasurement(accel, gyro, timestamp)
```

**Key Features:**
- Specific force computation (handles gravity)
- Bias modeling (constant + random walk)
- Realistic noise (white noise + random walk)
- Multiple IMU support

### Camera Simulator

The camera simulator (`camera_simulator.py`) generates visual observations:

```python
class CameraSimulator:
    def __init__(self, calibration: CameraCalibration):
        self.calibration = calibration
        self.K = self.get_intrinsic_matrix()
    
    def simulate_frame(self, pose: Pose, landmarks: Map) -> CameraFrame:
        observations = []
        
        for landmark in landmarks:
            # Project to camera frame
            pixel = self.project(landmark.position, pose)
            
            # Check visibility
            if self.is_visible(pixel):
                # Add measurement noise
                pixel += np.random.normal(0, self.pixel_noise, 2)
                observations.append(CameraObservation(
                    landmark_id=landmark.id,
                    pixel=pixel
                ))
        
        return CameraFrame(observations, timestamp)
```

**Features:**
- Pinhole camera model
- Visibility checking (FOV, occlusion)
- Pixel noise simulation
- Multi-camera support
- Stereo camera simulation

### Sensor Synchronization

The `SensorSync` class handles multi-rate sensors:

```python
sync = SensorSync()

# Register sensors with their rates
sync.register_sensor('imu', rate=200.0)
sync.register_sensor('camera', rate=30.0)
sync.register_sensor('gps', rate=1.0)

# Generate synchronized timeline
timeline = sync.generate_timeline(duration=10.0)

for event in timeline:
    if event.sensor == 'imu':
        imu_data = imu_sim.simulate(event.timestamp)
    elif event.sensor == 'camera':
        camera_data = camera_sim.simulate(event.timestamp)
```

## Landmark Generation

### Landmark Patterns

The landmark generator (`landmark_generator.py`) creates various 3D patterns:

#### 1. Cubic Grid
```python
landmarks = generate_landmarks('cube', {
    'density': 10,          # Landmarks per cubic meter
    'bounds': [-5, 5],      # Spatial bounds
    'height_range': [0, 3]  # Height range
})
```

#### 2. Cylindrical Pattern
```python
landmarks = generate_landmarks('cylinder', {
    'radius': 5.0,          # Cylinder radius
    'height': 3.0,          # Cylinder height
    'num_landmarks': 100    # Total landmarks
})
```

#### 3. Random Distribution
```python
landmarks = generate_landmarks('random', {
    'num_landmarks': 200,
    'bounds': [[-10, 10], [-10, 10], [0, 5]]
})
```

#### 4. Structured Environments
```python
# Generate room with walls
landmarks = generate_room_landmarks({
    'room_size': [10, 8, 3],  # Width, depth, height
    'wall_density': 20,       # Points per square meter
    'include_ceiling': True
})
```

## Simulation Pipeline

### Main Simulator

The `Simulator` class orchestrates the complete simulation:

```python
class Simulator:
    def __init__(self, config: SimulationConfig):
        self.trajectory_gen = TrajectoryGenerator(config)
        self.imu_sim = IMUSimulator(config.imu_calib)
        self.camera_sim = CameraSimulator(config.camera_calib)
        self.landmark_gen = LandmarkGenerator(config)
    
    def run(self) -> SimulationData:
        # 1. Generate trajectory
        trajectory = self.trajectory_gen.generate()
        
        # 2. Generate landmarks
        landmarks = self.landmark_gen.generate()
        
        # 3. Simulate sensors along trajectory
        imu_data = self.simulate_imu(trajectory)
        camera_data = self.simulate_cameras(trajectory, landmarks)
        
        # 4. Create keyframes
        keyframes = self.select_keyframes(trajectory, camera_data)
        
        return SimulationData(
            trajectory=trajectory,
            landmarks=landmarks,
            imu_measurements=imu_data,
            camera_frames=camera_data,
            keyframes=keyframes
        )
```

### Keyframe Selection

Keyframes are selected based on:
- Fixed time intervals
- Traveled distance
- Rotation change
- Number of observed landmarks

```python
selector = KeyframeSelector(config)
keyframes = selector.select(trajectory, observations)
```

## Configuration

### Simulation Configuration

Configuration via YAML files (`config/simulation/`):

```yaml
# config/simulation/default.yaml
simulation:
  duration: 10.0
  seed: 42

trajectory:
  type: circle
  params:
    radius: 2.0
    height: 1.5
    rate: 200.0

sensors:
  imu:
    rate: 200.0
    calibration:
      accelerometer_noise_density: 0.01
      gyroscope_noise_density: 0.001
      accelerometer_random_walk: 0.001
      gyroscope_random_walk: 0.0001
  
  camera:
    rate: 30.0
    calibration:
      width: 640
      height: 480
      fx: 500.0
      fy: 500.0
      cx: 320.0
      cy: 240.0
      pixel_noise: 1.0

landmarks:
  type: cube
  params:
    num_landmarks: 500
    bounds: [-5, 5]

keyframes:
  selection_method: hybrid
  time_threshold: 0.5
  distance_threshold: 0.5
  rotation_threshold: 0.1
```

### Loading Configuration

```python
from src.utils.config_loader import load_simulation_config

# Load default config
config = load_simulation_config()

# Load specific config
config = load_simulation_config('config/simulation/circle.yaml')

# Override parameters
config = load_simulation_config(
    'default.yaml',
    overrides={'trajectory.params.radius': 3.0}
)
```

## Usage Examples

### Basic Simulation

```python
from src.simulation import Simulator, SimulationConfig

# Create configuration
config = SimulationConfig(
    trajectory_type='circle',
    trajectory_params={'radius': 2.0, 'duration': 5.0},
    num_landmarks=100
)

# Run simulation
simulator = Simulator(config)
data = simulator.run()

# Access results
print(f"Generated {len(data.trajectory.states)} trajectory points")
print(f"Generated {len(data.landmarks.landmarks)} landmarks")
print(f"Generated {len(data.imu_measurements)} IMU measurements")
print(f"Generated {len(data.camera_frames)} camera frames")
```

### Multi-Sensor Simulation

```python
# Configure multiple cameras
config = SimulationConfig()
config.add_camera('front', fov=90, rate=30)
config.add_camera('left', fov=120, rate=20)
config.add_camera('right', fov=120, rate=20)

# Configure multiple IMUs
config.add_imu('imu_high', rate=1000, noise=0.001)
config.add_imu('imu_low', rate=100, noise=0.01)

# Run simulation
simulator = Simulator(config)
data = simulator.run()

# Access multi-sensor data
front_frames = data.camera_data['front']
high_rate_imu = data.imu_data['imu_high']
```

### Custom Trajectory

```python
# Define custom trajectory function
def custom_trajectory(t: float) -> TrajectoryState:
    # Spiral trajectory
    r = 1.0 + 0.1 * t
    theta = 2 * np.pi * t / 5.0
    
    position = np.array([
        r * np.cos(theta),
        r * np.sin(theta),
        0.5 * t
    ])
    
    # Compute derivatives for velocity/acceleration
    velocity = compute_velocity(position, dt=0.01)
    
    return TrajectoryState(
        pose=Pose(position=position),
        linear_velocity=velocity
    )

# Use custom trajectory
trajectory = Trajectory()
for t in np.linspace(0, 10, 1000):
    trajectory.add_state(custom_trajectory(t))

# Simulate sensors
simulator = Simulator(config)
data = simulator.simulate_sensors(trajectory)
```

### Batch Simulation

```python
# Generate multiple simulations with different parameters
results = []

for radius in [1.0, 2.0, 3.0]:
    for noise in [0.001, 0.01, 0.1]:
        config = SimulationConfig(
            trajectory_params={'radius': radius},
            imu_noise=noise
        )
        
        simulator = Simulator(config)
        data = simulator.run()
        
        results.append({
            'radius': radius,
            'noise': noise,
            'data': data
        })

# Analyze results
for result in results:
    analyze_trajectory_smoothness(result['data'])
```

## Output Format

### Simulation Data Structure

```python
@dataclass
class SimulationData:
    # Ground truth
    trajectory: Trajectory
    landmarks: Map
    
    # Sensor data
    imu_measurements: List[IMUMeasurement]
    camera_frames: List[CameraFrame]
    
    # Keyframes for SLAM
    keyframes: List[Keyframe]
    
    # Metadata
    config: SimulationConfig
    timestamp: str
    duration: float
```

### Saving and Loading

```python
from src.io import save_simulation, load_simulation

# Save simulation data
save_simulation(data, 'output/simulation_001.json')

# Load simulation data
loaded_data = load_simulation('output/simulation_001.json')

# Save in binary format for efficiency
save_simulation(data, 'output/simulation_001.pkl', format='pickle')
```

## Performance Considerations

### Memory Optimization

For long simulations:
```python
# Stream processing for large datasets
simulator = Simulator(config)
with simulator.stream_mode() as stream:
    for chunk in stream.generate(chunk_size=1000):
        process_chunk(chunk)
        save_chunk(chunk)
```

### Parallel Simulation

```python
from multiprocessing import Pool

def run_single_simulation(params):
    config = SimulationConfig(**params)
    simulator = Simulator(config)
    return simulator.run()

# Run parallel simulations
params_list = [{'seed': i, **base_params} for i in range(100)]
with Pool(processes=8) as pool:
    results = pool.map(run_single_simulation, params_list)
```

## Validation

### Consistency Checks

The simulator includes validation:
- Trajectory continuity
- Sensor timestamp monotonicity
- Landmark visibility consistency
- IMU measurement physical constraints

```python
# Enable validation
config.enable_validation = True
simulator = Simulator(config)
data = simulator.run()  # Raises ValidationError if issues found
```

### Ground Truth Comparison

```python
# Compare simulated vs theoretical values
validator = SimulationValidator()
report = validator.validate(data)

print(f"Position continuity: {report.position_smooth}")
print(f"IMU bias stability: {report.bias_stable}")
print(f"Camera projection accuracy: {report.projection_error}")
```

## Troubleshooting

### Common Issues

1. **Trajectory Discontinuities**
   - Check interpolation settings
   - Increase trajectory generation rate

2. **No Camera Observations**
   - Verify landmark positions
   - Check camera FOV settings
   - Ensure proper coordinate frames

3. **IMU Drift**
   - This is realistic! IMU always drifts
   - Adjust noise parameters for less drift
   - Add bias estimation in SLAM

4. **Memory Issues**
   - Use streaming mode for long simulations
   - Reduce sensor rates
   - Save data incrementally

## References

- IMU simulation: "A Calibration Algorithm for Low-Cost IMUs" (Tedaldi et al.)
- Camera models: "Multiple View Geometry" (Hartley & Zisserman)
- Trajectory generation: "Minimum Snap Trajectory Generation" (Mellinger et al.)