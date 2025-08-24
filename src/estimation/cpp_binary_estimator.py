"""
C++ Binary Estimator wrapper for external SLAM algorithms.
"""

import os
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..common.data_structures import (
    Trajectory, 
    PreintegratedIMUData, CameraData,
    Landmark
)
from .result_io import EstimatorResultStorage


class CppBinaryEstimatorError(Exception):
    """Base exception for C++ binary estimator errors."""
    pass


class CppBinaryTimeoutError(CppBinaryEstimatorError):
    """Raised when the binary execution times out."""
    pass


class CppBinaryExecutionError(CppBinaryEstimatorError):
    """Raised when the binary execution fails."""
    pass


class CppBinaryEstimator:
    """
    Wrapper for executing external C++ SLAM estimators.
    
    This class handles:
    - Data serialization to JSON
    - Process execution with timeout
    - Output parsing
    - Error handling and retries
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the C++ binary estimator.
        
        Args:
            config: Configuration dictionary with parameters for the binary
        """
        params = config.get('parameters', config)
        
        self.executable = Path(params['executable'])
        self.timeout = params.get('timeout', 300)  # seconds
        self.working_dir = params.get('working_dir')
        self.env = params.get('env', {})
        self.args = params.get('args', [])
        self.input_file = params.get('input_file', 'simulation_data.json')
        self.output_file = params.get('output_file', 'estimation_result.json')
        self.retry_on_failure = params.get('retry_on_failure', False)
        self.max_retries = params.get('max_retries', 1)
        
        # Validate executable exists
        if not self.executable.exists():
            # Try relative to working directory
            if self.working_dir:
                exe_path = Path(self.working_dir) / self.executable
                if exe_path.exists():
                    self.executable = exe_path
                else:
                    raise FileNotFoundError(f"Executable not found: {self.executable}")
            else:
                raise FileNotFoundError(f"Executable not found: {self.executable}")
    
    def run(self, 
            trajectory_gt: Trajectory,
            landmarks: List[Landmark],
            camera_data: Optional[CameraData] = None,
            imu_data: Optional[PreintegratedIMUData] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Run the C++ binary estimator on the provided data.
        
        Args:
            trajectory_gt: Ground truth trajectory
            landmarks: List of landmarks
            camera_data: Camera measurements
            imu_data: IMU measurements (preintegrated or raw)
            **kwargs: Additional data to pass to the binary
        
        Returns:
            Dictionary containing the estimation results with 'trajectory' and 'landmarks'
        
        Raises:
            CppBinaryTimeoutError: If execution times out
            CppBinaryExecutionError: If execution fails
        """
        # Prepare simulation data for export
        simulation_data = self._prepare_simulation_data(
            trajectory_gt, landmarks, camera_data, imu_data, **kwargs
        )
        
        # Try execution with retries
        attempts = 0
        last_error = None
        
        while attempts <= self.max_retries:
            try:
                attempts += 1
                print(f"Executing C++ binary (attempt {attempts}/{self.max_retries + 1})...")
                
                # Write input data
                input_path = self._write_input_data(simulation_data)
                
                # Execute the binary (writes to output file)
                self._execute_binary(input_path)
                
                # Load results using EstimatorResultStorage
                output_path = Path(self.working_dir) / self.output_file if self.working_dir else Path(self.output_file)
                result = EstimatorResultStorage.load_result(output_path)
                
                return result
                
            except CppBinaryTimeoutError:
                last_error = f"Timeout after {self.timeout} seconds"
                if not self.retry_on_failure or attempts > self.max_retries:
                    raise
                    
            except CppBinaryExecutionError as e:
                last_error = str(e)
                if not self.retry_on_failure or attempts > self.max_retries:
                    raise
                    
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                if not self.retry_on_failure or attempts > self.max_retries:
                    raise CppBinaryExecutionError(last_error)
            
            # Wait before retry
            if attempts <= self.max_retries:
                print(f"Retrying in 1 second... (error: {last_error})")
                time.sleep(1)
        
        # Should not reach here
        raise CppBinaryExecutionError(f"Failed after {attempts} attempts: {last_error}")
    
    def _prepare_simulation_data(self,
                                 trajectory_gt: Trajectory,
                                 landmarks: List[Landmark],
                                 camera_data: Optional[CameraData],
                                 imu_data: Optional[PreintegratedIMUData],
                                 **kwargs) -> Dict[str, Any]:
        """
        Prepare simulation data for JSON export.
        """
        data = {
            "metadata": {
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "estimator": "cpp_binary"
            },
            "trajectory": self._trajectory_to_dict(trajectory_gt),
            "landmarks": [self._landmark_to_dict(lm) for lm in landmarks]
        }
        
        if camera_data:
            data["camera_data"] = self._camera_data_to_dict(camera_data)
        
        if imu_data:
            data["imu_data"] = self._imu_data_to_dict(imu_data)
        
        # Add any additional data
        for key, value in kwargs.items():
            if key not in data:
                data[key] = value
        
        return data
    
    def _trajectory_to_dict(self, trajectory: Trajectory) -> List[Dict]:
        """Convert Trajectory to dictionary format."""
        from ..utils.math_utils import rotation_matrix_to_quaternion
        points = []
        for state in trajectory.states:
            points.append({
                "timestamp": float(state.pose.timestamp),
                "position": state.pose.position.tolist(),
                "quaternion": rotation_matrix_to_quaternion(state.pose.rotation_matrix).tolist(),
                "velocity": state.velocity.tolist() if state.velocity is not None else None
            })
        return points
    
    def _landmark_to_dict(self, landmark: Landmark) -> Dict:
        """Convert Landmark to dictionary format."""
        return {
            "id": int(landmark.id),
            "position": landmark.position.tolist()
        }
    
    def _camera_data_to_dict(self, camera_data: CameraData) -> Dict:
        """Convert CameraData to dictionary format."""
        frames = []
        for frame in camera_data.frames:
            frames.append({
                "timestamp": float(frame.timestamp),
                "measurements": [
                    {
                        "landmark_id": int(m.landmark_id),
                        "pixel": m.pixel.tolist()
                    }
                    for m in frame.measurements
                ]
            })
        
        return {
            "frames": frames,
            "calibration": {
                "fx": camera_data.calibration.intrinsics.fx,
                "fy": camera_data.calibration.intrinsics.fy,
                "cx": camera_data.calibration.intrinsics.cx,
                "cy": camera_data.calibration.intrinsics.cy,
            } if hasattr(camera_data, 'calibration') else {}
        }
    
    def _imu_data_to_dict(self, imu_data: PreintegratedIMUData) -> Dict:
        """Convert IMU data to dictionary format."""
        if hasattr(imu_data, 'measurements'):
            # Raw IMU measurements
            measurements = []
            for m in imu_data.measurements:
                measurements.append({
                    "timestamp": float(m.timestamp),
                    "acceleration": m.acceleration.tolist(),
                    "angular_velocity": m.angular_velocity.tolist()
                })
            return {"measurements": measurements}
        else:
            # Preintegrated IMU data
            preintegrated = []
            for p in imu_data:
                preintegrated.append({
                    "start_time": float(p.start_time),
                    "end_time": float(p.end_time),
                    "delta_position": p.delta_position.tolist(),
                    "delta_velocity": p.delta_velocity.tolist(),
                    "delta_rotation": p.delta_rotation.tolist()
                })
            return {"preintegrated": preintegrated}
    
    def _write_input_data(self, data: Dict[str, Any]) -> Path:
        """
        Write input data to JSON file.
        
        Returns:
            Path to the written file
        """
        # Use temp directory if no working directory specified
        if self.working_dir:
            work_dir = Path(self.working_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
            input_path = work_dir / self.input_file
        else:
            # Create temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                input_path = Path(f.name)
        
        # Write JSON data
        with open(input_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return input_path
    
    def _execute_binary(self, input_path: Path) -> None:
        """
        Execute the C++ binary with the given input.
        
        Args:
            input_path: Path to input JSON file
        
        Raises:
            CppBinaryTimeoutError: If execution times out
            CppBinaryExecutionError: If execution fails
        """
        # Prepare command
        cmd = [str(self.executable)]
        
        # Add additional arguments first (like script name for Python)
        cmd.extend(self.args)
        
        # Add input file argument
        cmd.extend(['--input', str(input_path)])
        
        # Add output file argument if needed
        output_path = input_path.parent / self.output_file
        if '--output' not in ' '.join(self.args):
            cmd.extend(['--output', str(output_path)])
        
        # Prepare environment
        env = os.environ.copy()
        env.update(self.env)
        
        # Execute the binary
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
                cwd=self.working_dir
            )
            
            # Check return code
            if result.returncode != 0:
                raise CppBinaryExecutionError(
                    f"Binary exited with code {result.returncode}\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )
            
            # Check output file exists
            if not output_path.exists():
                # Try in working directory
                if self.working_dir:
                    output_path = Path(self.working_dir) / self.output_file
                
                if not output_path.exists():
                    raise CppBinaryExecutionError(
                        f"Output file not found: {output_path}\n"
                        f"stdout: {result.stdout}"
                    )
            
        except subprocess.TimeoutExpired:
            raise CppBinaryTimeoutError(
                f"Binary execution timed out after {self.timeout} seconds"
            )
