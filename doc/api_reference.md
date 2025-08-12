# SLAM Simulation System - API Reference

## Table of Contents
1. [Core Data Structures](#core-data-structures)
2. [Simulation API](#simulation-api)
3. [Estimation API](#estimation-api)
4. [Evaluation API](#evaluation-api)
5. [Utility Functions](#utility-functions)
6. [CLI Commands](#cli-commands)

## Core Data Structures

### SimulationData
Main container for all simulation data.

```python
from src.common.json_io import SimulationData

class SimulationData:
    """Container for simulation data with JSON serialization support."""
    
    def __init__(self):
        self.metadata: Dict[str, Any]
        self.calibration: CalibrationData
        self.groundtruth: GroundTruthData
        self.measurements: MeasurementData
        
    def to_json(self, filepath: Path) -> None:
        """Save to JSON file."""
        
    @classmethod
    def from_json(cls, filepath: Path) -> 'SimulationData':
        """Load from JSON file."""
```

### State
Robot state representation.

```python
@dataclass
class State:
    timestamp: float
    position: np.ndarray  # [3,]
    orientation: np.ndarray  # [4,] quaternion [w,x,y,z]
    velocity: np.ndarray  # [3,]
    bias_accelerometer: np.ndarray  # [3,]
    bias_gyroscope: np.ndarray  # [3,]
    covariance: Optional[np.ndarray] = None  # [15,15]
```

### Trajectory
Sequence of states over time.

```python
class Trajectory:
    def __init__(self, states: List[State]):
        self.states = states
        
    def get_state_at_time(self, timestamp: float) -> State:
        """Interpolate state at given timestamp."""
        
    def to_matrix(self) -> np.ndarray:
        """Convert to Nx7 matrix [t,x,y,z,qw,qx,qy,qz]."""
```

### Landmark
3D point feature.

```python
@dataclass
class Landmark:
    id: int
    position: np.ndarray  # [3,] world coordinates
    descriptor: Optional[np.ndarray] = None
    covariance: Optional[np.ndarray] = None  # [3,3]
```

### CameraObservation
2D image measurement.

```python
@dataclass
class CameraObservation:
    landmark_id: int
    pixel: np.ndarray  # [2,] [u, v]
    camera_id: str = "cam0"
    descriptor: Optional[np.ndarray] = None
```

## Simulation API

### Trajectory Generation

```python
from src.simulation.trajectory import generate_trajectory

def generate_trajectory(
    trajectory_type: str,
    duration: float = 20.0,
    rate: float = 200.0,
    **params
) -> Trajectory:
    """
    Generate a trajectory.
    
    Args:
        trajectory_type: One of ['circle', 'figure8', 'spiral', 'line', 'random_walk']
        duration: Total duration in seconds
        rate: Sampling rate in Hz
        **params: Type-specific parameters
        
    Returns:
        Trajectory object with states at specified rate
        
    Examples:
        # Circle trajectory
        traj = generate_trajectory('circle', radius=5.0, height=1.5)
        
        # Figure-8 trajectory
        traj = generate_trajectory('figure8', scale_x=3.0, scale_y=2.0)
    """
```

### Landmark Generation

```python
from src.simulation.landmarks import generate_landmarks

def generate_landmarks(
    num_landmarks: int,
    bounds: List[float],
    distribution: str = 'uniform',
    seed: Optional[int] = None
) -> List[Landmark]:
    """
    Generate 3D landmarks.
    
    Args:
        num_landmarks: Number of landmarks to generate
        bounds: [x_min, x_max, y_min, y_max, z_min, z_max]
        distribution: 'uniform', 'gaussian', or 'clustered'
        seed: Random seed for reproducibility
        
    Returns:
        List of Landmark objects
    """
```

### Sensor Simulation

```python
from src.simulation.sensors import IMUSensor, CameraSensor

class IMUSensor:
    def __init__(self, config: Dict):
        """
        Initialize IMU sensor.
        
        Config keys:
            rate: Sampling rate in Hz
            noise_acc: Accelerometer noise density
            noise_gyro: Gyroscope noise density
            bias_acc_stability: Accelerometer bias stability
            bias_gyro_stability: Gyroscope bias stability
        """
        
    def generate_measurements(
        self,
        trajectory: Trajectory,
        add_noise: bool = True
    ) -> IMUData:
        """Generate IMU measurements from trajectory."""

class CameraSensor:
    def __init__(self, calibration: CameraCalibration):
        """Initialize camera with calibration."""
        
    def project_landmarks(
        self,
        state: State,
        landmarks: List[Landmark],
        add_noise: bool = True,
        pixel_noise_std: float = 1.0
    ) -> List[CameraObservation]:
        """Project landmarks to camera frame."""
```

### Complete Simulation

```python
from src.simulation import simulate_slam_scenario

def simulate_slam_scenario(
    config: Dict
) -> SimulationData:
    """
    Run complete simulation pipeline.
    
    Config structure:
        {
            'trajectory': {...},
            'landmarks': {...},
            'sensors': {
                'imu': {...},
                'camera': {...}
            },
            'noise': {...}
        }
    
    Returns:
        SimulationData with groundtruth and measurements
    """
```

## Estimation API

### Base Estimator Interface

```python
from src.estimation.base import SLAMEstimator

class SLAMEstimator(ABC):
    """Abstract base class for SLAM estimators."""
    
    @abstractmethod
    def process_imu(
        self,
        timestamp: float,
        accelerometer: np.ndarray,
        gyroscope: np.ndarray
    ) -> None:
        """Process IMU measurement."""
        
    @abstractmethod
    def process_camera(
        self,
        timestamp: float,
        observations: List[CameraObservation]
    ) -> None:
        """Process camera observations."""
        
    @abstractmethod
    def get_current_state(self) -> State:
        """Get current estimated state."""
        
    @abstractmethod
    def get_map(self) -> List[Landmark]:
        """Get estimated landmark map."""
```

### EKF-SLAM

```python
from src.estimation.ekf_slam import EKFSlam

class EKFSlam(SLAMEstimator):
    def __init__(self, config: Dict):
        """
        Initialize EKF-SLAM.
        
        Config keys:
            process_noise: Process noise covariance
            measurement_noise: Measurement noise
            chi2_threshold: Chi-squared test threshold
            max_landmarks: Maximum number of landmarks
        """
        
    def process(self, data: SimulationData) -> EstimationResult:
        """Process complete dataset."""
```

### Sliding Window Bundle Adjustment

```python
from src.estimation.swba_slam import SlidingWindowBA

class SlidingWindowBA(SLAMEstimator):
    def __init__(self, config: Dict):
        """
        Initialize SWBA.
        
        Config keys:
            window_size: Number of keyframes in window
            optimization_config: {
                'max_iterations': 10,
                'convergence_threshold': 1e-6,
                'robust_kernel': 'huber'
            }
        """
        
    def marginalize_old_states(self) -> None:
        """Marginalize states outside window."""
```

### Square Root Information Filter

```python
from src.estimation.srif_slam import SRIFSlam

class SRIFSlam(SLAMEstimator):
    def __init__(self, config: Dict):
        """
        Initialize SRIF.
        
        Config keys:
            qr_threshold: QR decomposition threshold
            measurement_chunking: Process measurements in chunks
            chunk_size: Size of measurement chunks
        """
        
    def update_information_matrix(
        self,
        jacobian: np.ndarray,
        residual: np.ndarray
    ) -> None:
        """Update R matrix using QR factorization."""
```

## Evaluation API

### Trajectory Metrics

```python
from src.evaluation.metrics import TrajectoryMetrics

class TrajectoryMetrics:
    @staticmethod
    def compute_ate(
        estimated: Trajectory,
        groundtruth: Trajectory,
        align: bool = True
    ) -> Dict[str, float]:
        """
        Compute Absolute Trajectory Error.
        
        Returns:
            {
                'rmse': float,
                'mean': float,
                'median': float,
                'std': float,
                'min': float,
                'max': float
            }
        """
        
    @staticmethod
    def compute_rpe(
        estimated: Trajectory,
        groundtruth: Trajectory,
        delta: float = 1.0,
        unit: str = 'seconds'
    ) -> Dict[str, float]:
        """
        Compute Relative Pose Error.
        
        Args:
            delta: Time delta for relative poses
            unit: 'seconds' or 'frames'
        """
```

### Consistency Metrics

```python
from src.evaluation.metrics import ConsistencyMetrics

class ConsistencyMetrics:
    @staticmethod
    def compute_nees(
        estimated_states: List[State],
        groundtruth_states: List[State]
    ) -> np.ndarray:
        """
        Compute Normalized Estimation Error Squared.
        
        Returns:
            Array of NEES values over time
        """
        
    @staticmethod
    def chi2_test(
        nees_values: np.ndarray,
        dof: int,
        confidence: float = 0.95
    ) -> Dict[str, Any]:
        """
        Perform chi-squared consistency test.
        
        Returns:
            {
                'consistent': bool,
                'percentage_in_bounds': float,
                'lower_bound': float,
                'upper_bound': float
            }
        """
```

### Comparison Framework

```python
from src.evaluation.comparison import ComparisonRunner

class ComparisonRunner:
    def __init__(self, simulation_data: SimulationData):
        """Initialize with ground truth data."""
        
    def run_estimator(
        self,
        estimator_class: Type[SLAMEstimator],
        config: Dict
    ) -> EstimationResult:
        """Run single estimator."""
        
    def compare_estimators(
        self,
        estimators: Dict[str, Dict]
    ) -> ComparisonResult:
        """
        Compare multiple estimators.
        
        Args:
            estimators: {
                'name': {
                    'class': EstimatorClass,
                    'config': {...}
                }
            }
            
        Returns:
            ComparisonResult with metrics for all estimators
        """
```

## Utility Functions

### Math Utilities

```python
from src.utils.math_utils import *

def se3_inverse(T: np.ndarray) -> np.ndarray:
    """Compute inverse of SE(3) transformation."""

def se3_log(T: np.ndarray) -> np.ndarray:
    """Compute logarithm map of SE(3)."""

def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Compute exponential map to SE(3)."""

def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton quaternion multiplication."""

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to 3x3 rotation matrix."""

def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion."""

def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Create skew-symmetric matrix from 3D vector."""
```

### Dataset Conversion

```python
from src.utils.tumvie_converter import convert_tumvie_dataset

def convert_tumvie_dataset(
    dataset_dir: Path,
    output_file: Path,
    num_landmarks: int = 200,
    keyframe_interval: float = 0.1,
    pixel_noise_std: float = 1.0
) -> None:
    """
    Convert TUM-VIE dataset to simulation format.
    
    Args:
        dataset_dir: Path to TUM-VIE dataset
        output_file: Output JSON file path
        num_landmarks: Number of synthetic landmarks to generate
        keyframe_interval: Time between keyframes in seconds
        pixel_noise_std: Standard deviation of pixel noise
    """
```

### Data Inspection

```python
from tools.inspect_data import DataInspector

class DataInspector:
    @staticmethod
    def summarize(data_file: Path) -> Dict:
        """Get summary statistics of dataset."""
        
    @staticmethod
    def validate(data_file: Path) -> List[str]:
        """Validate dataset integrity."""
        
    @staticmethod
    def extract_segment(
        data_file: Path,
        start_time: float,
        end_time: float
    ) -> SimulationData:
        """Extract time segment from dataset."""
```

## CLI Commands

### Simulation Commands

```bash
# Generate synthetic dataset
./run.sh simulate [trajectory_type] [options]

Options:
  --duration FLOAT       Simulation duration in seconds [default: 20.0]
  --rate FLOAT          IMU sampling rate in Hz [default: 200.0]
  --camera-rate FLOAT   Camera rate in Hz [default: 30.0]
  --num-landmarks INT   Number of landmarks [default: 100]
  --add-noise          Add realistic sensor noise
  --noise-level FLOAT  Noise scale factor [default: 1.0]
  --seed INT           Random seed for reproducibility
  --config FILE        Configuration YAML file
  --output FILE        Output JSON file path

Examples:
  # Circle trajectory with noise
  ./run.sh simulate circle --duration 60 --add-noise --output circle.json
  
  # Figure-8 with custom config
  ./run.sh simulate figure8 --config config/sim_figure8.yaml
```

### SLAM Commands

```bash
# Run SLAM estimation
./run.sh slam [estimator] --input FILE [options]

Estimators:
  ekf     Extended Kalman Filter
  swba    Sliding Window Bundle Adjustment
  srif    Square Root Information Filter

Options:
  --config FILE        Estimator configuration YAML
  --output FILE        Output trajectory JSON
  --visualize         Show live visualization
  --verbose           Verbose output

Examples:
  # Run EKF on dataset
  ./run.sh slam ekf --input data.json --output ekf_result.json
  
  # SWBA with custom config
  ./run.sh slam swba --input data.json --config config/swba_tuned.yaml
```

### Evaluation Commands

```bash
# Run evaluation pipeline
./run.sh evaluation [config_file] [options]

Options:
  --datasets LIST      Comma-separated dataset names
  --estimators LIST    Comma-separated estimator names
  --parallel INT       Number of parallel jobs
  --output DIR         Output directory
  --skip-generation    Skip dataset generation
  --skip-dashboard     Skip dashboard generation
  --dry-run           Show plan without executing

Examples:
  # Full evaluation
  ./run.sh evaluation config/evaluation_config.yaml
  
  # Quick test
  ./run.sh evaluation --datasets circle --estimators ekf --parallel 2
```

### Conversion Commands

```bash
# Convert external datasets
./run.sh convert [dataset_type] INPUT OUTPUT [options]

Dataset Types:
  tumvie   TUM Visual-Inertial-Event
  euroc    EuRoC MAV dataset
  kitti    KITTI dataset

Options:
  --num-landmarks INT      For synthetic observations [default: 200]
  --keyframe-interval FLOAT   Time between keyframes [default: 0.1]
  --pixel-noise FLOAT      Pixel noise std [default: 1.0]

Examples:
  # Convert TUM-VIE dataset
  ./run.sh convert tumvie data/TUM-VIE/room1 output/room1.json
  
  # Dense observations
  ./run.sh convert tumvie data/TUM-VIE/room1 output/dense.json \
    --num-landmarks 500 --keyframe-interval 0.05
```

### Utility Commands

```bash
# Download datasets
./run.sh download [dataset] --sequence NAME [options]

# Plot results
./run.sh plot INPUT [options]
  --compare FILE       Compare with another trajectory
  --output FILE        Save plot to HTML file
  --metrics           Show metrics subplot

# Clean generated files
./run.sh clean
  --all               Remove all generated files
  --cache             Remove only cache files

# Show system info
./run.sh info
```

## Configuration Examples

### Simulation Configuration

```yaml
# config/simulation.yaml
trajectory:
  type: circle
  params:
    radius: 5.0
    height: 1.5
    angular_velocity: 0.5

sensors:
  imu:
    rate: 200.0
    accelerometer:
      noise_density: 0.00018
      random_walk: 0.001
    gyroscope:
      noise_density: 0.00026
      random_walk: 0.0001
      
  camera:
    rate: 30.0
    model: pinhole-radtan
    resolution: [640, 480]
    intrinsics:
      fx: 458.654
      fy: 457.296
      cx: 367.215
      cy: 248.375
    distortion: [-0.28, 0.07, 0.0001, 0.0001, 0.0]
    
landmarks:
  count: 200
  distribution: uniform
  bounds: [-10, 10, -10, 10, 0, 5]
```

### Estimator Configuration

```yaml
# config/ekf.yaml
ekf:
  process_noise:
    position: 0.01
    orientation: 0.001
    velocity: 0.1
    bias_accel: 0.001
    bias_gyro: 0.0001
    
  measurement_noise:
    pixel: 1.0
    
  initialization:
    position_std: 0.1
    orientation_std: 0.01
    velocity_std: 0.1
    
  outlier_rejection:
    chi2_threshold: 5.991  # 95% confidence
    
  landmark_initialization:
    min_parallax: 1.0  # pixels
    min_observations: 3
```

## Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| E001 | Invalid trajectory type | Check supported types in documentation |
| E002 | Calibration file not found | Ensure calibration files exist |
| E003 | Time synchronization error | Check timestamp monotonicity |
| E004 | Insufficient observations | Increase sensor rate or landmarks |
| E005 | Optimization divergence | Tune optimization parameters |
| E006 | Memory allocation failure | Reduce batch size or landmarks |
| E007 | Invalid configuration | Validate YAML against schema |
| E008 | Dataset format error | Check JSON structure |

---

*Version: 1.0.0*
*Last Updated: January 2025*