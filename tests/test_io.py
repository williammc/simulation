"""
Unit tests for I/O operations.
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path

from src.common.data_structures import (
    IMUMeasurement, IMUData,
    CameraFrame, CameraData, CameraObservation, ImagePoint,
    Trajectory, TrajectoryState, Pose,
    Map, Landmark,
    CameraCalibration, IMUCalibration,
    CameraIntrinsics, CameraExtrinsics, CameraModel
)
from src.common.json_io import (
    SimulationData,
    save_simulation_data,
    load_simulation_data
)
from src.simulation.trajectory_generator import (
    CircleTrajectory,
    TrajectoryParams,
    generate_trajectory
)


class TestSimulationDataIO:
    """Test SimulationData I/O operations."""
    
    def test_empty_simulation_data(self):
        """Test creating and saving empty simulation data."""
        sim_data = SimulationData()
        sim_data.set_metadata(
            trajectory_type="test",
            duration=10.0,
            coordinate_system="ENU"
        )
        
        assert sim_data.metadata["trajectory_type"] == "test"
        assert sim_data.metadata["duration"] == 10.0
        assert sim_data.metadata["coordinate_system"] == "ENU"
    
    def test_add_camera_calibration(self):
        """Test adding camera calibration."""
        sim_data = SimulationData()
        
        intrinsics = CameraIntrinsics(
            model=CameraModel.PINHOLE,
            width=640,
            height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            distortion=np.zeros(5)
        )
        
        extrinsics = CameraExtrinsics(B_T_C=np.eye(4))
        
        calib = CameraCalibration(
            camera_id="cam0",
            intrinsics=intrinsics,
            extrinsics=extrinsics
        )
        
        sim_data.add_camera_calibration(calib)
        
        assert len(sim_data.calibration["cameras"]) == 1
        assert sim_data.calibration["cameras"][0]["id"] == "cam0"
    
    def test_add_imu_calibration(self):
        """Test adding IMU calibration."""
        sim_data = SimulationData()
        
        calib = IMUCalibration(
            imu_id="imu0",
            accelerometer_noise_density=0.01,
            accelerometer_random_walk=0.001,
            gyroscope_noise_density=0.0001,
            gyroscope_random_walk=0.00001,
            rate=200.0
        )
        
        sim_data.add_imu_calibration(calib)
        
        assert len(sim_data.calibration["imus"]) == 1
        assert sim_data.calibration["imus"][0]["id"] == "imu0"
    
    def test_set_trajectory(self):
        """Test setting ground truth trajectory."""
        sim_data = SimulationData()
        
        trajectory = Trajectory()
        for t in [0.0, 1.0, 2.0]:
            pose = Pose(
                timestamp=t,
                position=np.array([t, 0, 0]),
                quaternion=np.array([1, 0, 0, 0])
            )
            state = TrajectoryState(
                pose=pose,
                velocity=np.array([1, 0, 0]),
                angular_velocity=np.array([0, 0, 0.1])
            )
            trajectory.add_state(state)
        
        sim_data.set_groundtruth_trajectory(trajectory)
        
        assert len(sim_data.groundtruth["trajectory"]) == 3
        assert sim_data.groundtruth["trajectory"][0]["timestamp"] == 0.0
        assert sim_data.groundtruth["trajectory"][0]["velocity"] == [1, 0, 0]
    
    def test_set_landmarks(self):
        """Test setting ground truth landmarks."""
        sim_data = SimulationData()
        
        map_data = Map()
        for i in range(5):
            landmark = Landmark(
                id=i,
                position=np.array([i, i*2, i*3]),
                descriptor=np.random.randn(128)
            )
            map_data.add_landmark(landmark)
        
        sim_data.set_groundtruth_landmarks(map_data)
        
        assert len(sim_data.groundtruth["landmarks"]) == 5
        assert sim_data.groundtruth["landmarks"][2]["id"] == 2
        assert sim_data.groundtruth["landmarks"][2]["position"] == [2, 4, 6]
    
    def test_set_imu_measurements(self):
        """Test setting IMU measurements."""
        sim_data = SimulationData()
        
        imu_data = IMUData()
        for t in [0.0, 0.005, 0.01]:
            meas = IMUMeasurement(
                timestamp=t,
                accelerometer=np.array([0, 0, 9.81]),
                gyroscope=np.array([0, 0, 0.1])
            )
            imu_data.add_measurement(meas)
        
        sim_data.set_imu_measurements(imu_data)
        
        assert len(sim_data.measurements["imu"]) == 3
        assert sim_data.measurements["imu"][0]["timestamp"] == 0.0
    
    def test_add_camera_measurements(self):
        """Test adding camera measurements."""
        sim_data = SimulationData()
        
        camera_data = CameraData(camera_id="cam0")
        
        for t in [0.0, 0.033, 0.066]:
            observations = [
                CameraObservation(
                    landmark_id=i,
                    pixel=ImagePoint(u=100+i*10, v=200+i*10)
                )
                for i in range(3)
            ]
            
            frame = CameraFrame(
                timestamp=t,
                camera_id="cam0",
                observations=observations
            )
            camera_data.add_frame(frame)
        
        sim_data.add_camera_measurements(camera_data)
        
        assert len(sim_data.measurements["camera_frames"]) == 3
        assert len(sim_data.measurements["camera_frames"][0]["observations"]) == 3
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading complete simulation data."""
        # Create simulation data
        sim_data = SimulationData()
        sim_data.set_metadata(
            trajectory_type="test",
            duration=2.0,
            coordinate_system="ENU",
            seed=42
        )
        
        # Add trajectory
        trajectory = Trajectory()
        for t in [0.0, 1.0, 2.0]:
            pose = Pose(
                timestamp=t,
                position=np.array([t, t*2, 0]),
                quaternion=np.array([1, 0, 0, 0])
            )
            state = TrajectoryState(pose=pose)
            trajectory.add_state(state)
        
        sim_data.set_groundtruth_trajectory(trajectory)
        
        # Save to file
        output_file = tmp_path / "test_simulation.json"
        sim_data.save(output_file)
        
        assert output_file.exists()
        
        # Load from file
        loaded_data = SimulationData.load(output_file)
        
        assert loaded_data.metadata["trajectory_type"] == "test"
        assert loaded_data.metadata["duration"] == 2.0
        assert len(loaded_data.groundtruth["trajectory"]) == 3
    
    def test_extract_trajectory(self):
        """Test extracting trajectory from loaded data."""
        sim_data = SimulationData()
        
        # Create and set trajectory
        trajectory = Trajectory()
        for t in [0.0, 1.0]:
            pose = Pose(
                timestamp=t,
                position=np.array([t, 0, 0]),
                quaternion=np.array([1, 0, 0, 0])
            )
            state = TrajectoryState(
                pose=pose,
                velocity=np.array([1, 0, 0])
            )
            trajectory.add_state(state)
        
        sim_data.set_groundtruth_trajectory(trajectory)
        
        # Extract trajectory
        extracted = sim_data.get_trajectory()
        
        assert extracted is not None
        assert len(extracted.states) == 2
        assert np.allclose(extracted.states[0].pose.position, [0, 0, 0])
        assert np.allclose(extracted.states[0].velocity, [1, 0, 0])


class TestConvenienceFunctions:
    """Test convenience save/load functions."""
    
    def test_save_simulation_data(self, tmp_path):
        """Test save_simulation_data convenience function."""
        # Create test data
        trajectory = Trajectory()
        pose = Pose(
            timestamp=0.0,
            position=np.array([0, 0, 0]),
            quaternion=np.array([1, 0, 0, 0])
        )
        state = TrajectoryState(pose=pose)
        trajectory.add_state(state)
        
        landmarks = Map()
        landmark = Landmark(id=0, position=np.array([1, 2, 3]))
        landmarks.add_landmark(landmark)
        
        imu_data = IMUData()
        meas = IMUMeasurement(
            timestamp=0.0,
            accelerometer=np.array([0, 0, 9.81]),
            gyroscope=np.zeros(3)
        )
        imu_data.add_measurement(meas)
        
        # Save using convenience function
        output_file = tmp_path / "test_save.json"
        save_simulation_data(
            filepath=output_file,
            trajectory=trajectory,
            landmarks=landmarks,
            imu_data=imu_data,
            metadata={"trajectory_type": "test", "duration": 1.0}
        )
        
        assert output_file.exists()
        
        # Verify saved data
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        assert data["metadata"]["trajectory_type"] == "test"
        assert len(data["groundtruth"]["trajectory"]) == 1
        assert len(data["groundtruth"]["landmarks"]) == 1
        assert len(data["measurements"]["imu"]) == 1
    
    def test_load_simulation_data(self, tmp_path):
        """Test load_simulation_data convenience function."""
        # Create and save test data
        sim_data = SimulationData()
        sim_data.set_metadata(trajectory_type="test", duration=1.0)
        
        trajectory = Trajectory()
        pose = Pose(
            timestamp=0.0,
            position=np.array([1, 2, 3]),
            quaternion=np.array([1, 0, 0, 0])
        )
        state = TrajectoryState(pose=pose)
        trajectory.add_state(state)
        sim_data.set_groundtruth_trajectory(trajectory)
        
        output_file = tmp_path / "test_load.json"
        sim_data.save(output_file)
        
        # Load using convenience function
        loaded = load_simulation_data(output_file)
        
        assert loaded["metadata"]["trajectory_type"] == "test"
        assert loaded["trajectory"] is not None
        assert len(loaded["trajectory"].states) == 1
        assert np.allclose(loaded["trajectory"].states[0].pose.position, [1, 2, 3])


class TestTrajectoryGenerator:
    """Test trajectory generation and export."""
    
    def test_circle_trajectory_generation(self):
        """Test circle trajectory generation."""
        params = TrajectoryParams(
            duration=2.0,
            rate=10.0,  # Low rate for testing
            start_time=0.0
        )
        
        generator = CircleTrajectory(
            radius=1.0,
            height=0.5,
            params=params
        )
        
        trajectory = generator.generate()
        
        assert len(trajectory.states) == 20  # 2 seconds * 10 Hz
        
        # Check first state
        first_state = trajectory.states[0]
        assert first_state.pose.timestamp == 0.0
        assert np.allclose(first_state.pose.position[2], 0.5)  # Height
        
        # Check velocities are present
        assert first_state.velocity is not None
        assert first_state.angular_velocity is not None
    
    def test_generate_trajectory_factory(self):
        """Test trajectory factory function."""
        params = {
            "duration": 1.0,
            "rate": 10.0,
            "radius": 2.0,
            "height": 1.0
        }
        
        trajectory = generate_trajectory("circle", params)
        
        assert len(trajectory.states) == 10
        assert trajectory.frame_id == "world"
    
    def test_trajectory_export_json(self, tmp_path):
        """Test exporting generated trajectory to JSON."""
        # Generate trajectory
        params = {
            "duration": 1.0,
            "rate": 5.0,
            "radius": 1.0,
            "height": 0.5
        }
        
        trajectory = generate_trajectory("circle", params)
        
        # Save to JSON
        output_file = tmp_path / "trajectory.json"
        save_simulation_data(
            filepath=output_file,
            trajectory=trajectory,
            metadata={"trajectory_type": "circle", "duration": 1.0}
        )
        
        # Load and verify
        loaded = load_simulation_data(output_file)
        
        assert loaded["trajectory"] is not None
        assert len(loaded["trajectory"].states) == 5
        assert loaded["metadata"]["trajectory_type"] == "circle"