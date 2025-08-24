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
    CameraIntrinsics, CameraExtrinsics, CameraModel,
    PreintegratedIMUData
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
                rotation_matrix=np.eye(3)
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
        """Test adding camera measurements with keyframe information."""
        sim_data = SimulationData()
        
        camera_data = CameraData(camera_id="cam0")
        
        for idx, t in enumerate([0.0, 0.033, 0.066]):
            observations = [
                CameraObservation(
                    landmark_id=i,
                    pixel=ImagePoint(u=100+i*10, v=200+i*10)
                )
                for i in range(3)
            ]
            
            # Mark first and last as keyframes
            is_keyframe = (idx == 0 or idx == 2)
            keyframe_id = idx // 2 if is_keyframe else None
            
            frame = CameraFrame(
                timestamp=t,
                camera_id="cam0",
                observations=observations,
                is_keyframe=is_keyframe,
                keyframe_id=keyframe_id
            )
            camera_data.add_frame(frame)
        
        sim_data.add_camera_measurements(camera_data)
        
        assert len(sim_data.measurements["camera_frames"]) == 3
        assert len(sim_data.measurements["camera_frames"][0]["observations"]) == 3
        assert sim_data.measurements["camera_frames"][0]["is_keyframe"] == True
        assert sim_data.measurements["camera_frames"][0]["keyframe_id"] == 0
        assert sim_data.measurements["camera_frames"][1]["is_keyframe"] == False
        assert sim_data.measurements["camera_frames"][2]["is_keyframe"] == True
        assert sim_data.measurements["camera_frames"][2]["keyframe_id"] == 1
    
    def test_set_preintegrated_imu(self):
        """Test setting preintegrated IMU data."""
        sim_data = SimulationData()
        
        # Create preintegrated IMU data dictionary
        preintegrated = {}
        for i in range(1, 4):  # Keyframes 1, 2, 3
            preint_data = PreintegratedIMUData(
                from_keyframe_id=i-1,
                to_keyframe_id=i,
                delta_position=np.array([0.1*i, 0.2*i, 0.0]),
                delta_velocity=np.array([0.01*i, 0.02*i, 0.0]),
                delta_rotation=np.eye(3),
                covariance=np.eye(15) * 0.01,
                dt=0.5,
                num_measurements=100
            )
            preintegrated[i] = preint_data
        
        sim_data.set_preintegrated_imu(preintegrated)
        
        assert len(sim_data.measurements["preintegrated_imu"]) == 3
        assert sim_data.measurements["preintegrated_imu"][0]["from_keyframe_id"] == 0
        assert sim_data.measurements["preintegrated_imu"][0]["to_keyframe_id"] == 1
        assert sim_data.measurements["preintegrated_imu"][0]["dt"] == 0.5
        assert sim_data.measurements["preintegrated_imu"][0]["num_measurements"] == 100
    
    def test_get_preintegrated_imu(self):
        """Test extracting preintegrated IMU data."""
        sim_data = SimulationData()
        
        # Create and set preintegrated IMU data
        preintegrated = {}
        for i in range(1, 3):
            preint_data = PreintegratedIMUData(
                from_keyframe_id=i-1,
                to_keyframe_id=i,
                delta_position=np.array([0.1, 0.2, 0.3]),
                delta_velocity=np.array([0.01, 0.02, 0.03]),
                delta_rotation=np.eye(3),
                covariance=np.eye(15),
                dt=0.1,
                num_measurements=20,
                jacobian=np.random.randn(15, 6)  # Include jacobian
            )
            preintegrated[i] = preint_data
        
        sim_data.set_preintegrated_imu(preintegrated)
        
        # Extract preintegrated IMU data
        extracted = sim_data.get_preintegrated_imu()
        
        assert len(extracted) == 2
        assert extracted[0].from_keyframe_id == 0
        assert extracted[0].to_keyframe_id == 1
        assert np.allclose(extracted[0].delta_position, [0.1, 0.2, 0.3])
        assert np.allclose(extracted[0].delta_velocity, [0.01, 0.02, 0.03])
        assert extracted[0].dt == 0.1
        assert extracted[0].num_measurements == 20
        # Jacobian should be loaded if it was saved
        assert extracted[0].jacobian is not None
        assert extracted[0].jacobian.shape == (15, 6)
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading complete simulation data with all components."""
        # Create simulation data
        sim_data = SimulationData()
        sim_data.set_metadata(
            trajectory_type="test",
            duration=2.0,
            coordinate_system="ENU",
            seed=42,
            preintegration_enabled=True
        )
        
        # Add calibrations
        cam_intrinsics = CameraIntrinsics(
            model=CameraModel.PINHOLE,
            width=640, height=480,
            fx=500.0, fy=500.0,
            cx=320.0, cy=240.0,
            distortion=np.zeros(5)
        )
        cam_extrinsics = CameraExtrinsics(B_T_C=np.eye(4))
        cam_calib = CameraCalibration(
            camera_id="cam0",
            intrinsics=cam_intrinsics,
            extrinsics=cam_extrinsics
        )
        sim_data.add_camera_calibration(cam_calib)
        
        imu_calib = IMUCalibration(
            imu_id="imu0",
            accelerometer_noise_density=0.01,
            accelerometer_random_walk=0.001,
            gyroscope_noise_density=0.001,
            gyroscope_random_walk=0.0001,
            rate=200.0
        )
        sim_data.add_imu_calibration(imu_calib)
        
        # Add trajectory
        trajectory = Trajectory()
        for t in [0.0, 1.0, 2.0]:
            pose = Pose(
                timestamp=t,
                position=np.array([t, t*2, 0]),
                rotation_matrix=np.eye(3)
            )
            state = TrajectoryState(
                pose=pose,
                velocity=np.array([1, 2, 0]),
                angular_velocity=np.array([0, 0, 0.1])
            )
            trajectory.add_state(state)
        
        sim_data.set_groundtruth_trajectory(trajectory)
        
        # Add landmarks
        landmarks = Map()
        for i in range(3):
            landmark = Landmark(
                id=i,
                position=np.array([i, i*2, i*3])
            )
            landmarks.add_landmark(landmark)
        sim_data.set_groundtruth_landmarks(landmarks)
        
        # Add IMU measurements
        imu_data = IMUData()
        for t in np.arange(0, 2, 0.005):  # 200Hz
            meas = IMUMeasurement(
                timestamp=t,
                accelerometer=np.array([0.1, 0, 9.81]),
                gyroscope=np.array([0, 0, 0.1])
            )
            imu_data.add_measurement(meas)
        sim_data.set_imu_measurements(imu_data)
        
        # Add camera measurements with keyframes
        camera_data = CameraData(camera_id="cam0")
        for idx, t in enumerate(np.arange(0, 2, 0.5)):  # 2Hz, 4 frames
            observations = [
                CameraObservation(
                    landmark_id=i,
                    pixel=ImagePoint(u=320+i*10, v=240+i*5)
                )
                for i in range(min(3, idx+1))  # Varying number of observations
            ]
            
            frame = CameraFrame(
                timestamp=t,
                camera_id="cam0",
                observations=observations,
                is_keyframe=True,  # All frames are keyframes
                keyframe_id=idx
            )
            camera_data.add_frame(frame)
        sim_data.add_camera_measurements(camera_data)
        
        # Add preintegrated IMU data
        preintegrated = {}
        for i in range(1, 4):  # Between keyframes 0-1, 1-2, 2-3
            preint_data = PreintegratedIMUData(
                from_keyframe_id=i-1,
                to_keyframe_id=i,
                delta_position=np.array([0.5, 1.0, 0.0]),
                delta_velocity=np.array([1.0, 2.0, 0.0]),
                delta_rotation=np.eye(3),
                covariance=np.eye(15) * 0.01,
                dt=0.5,
                num_measurements=100
            )
            preintegrated[i] = preint_data
        sim_data.set_preintegrated_imu(preintegrated)
        
        # Save to file
        output_file = tmp_path / "test_complete_simulation.json"
        sim_data.save(output_file)
        
        assert output_file.exists()
        
        # Load from file
        loaded_data = SimulationData.load(output_file)
        
        # Verify metadata
        assert loaded_data.metadata["trajectory_type"] == "test"
        assert loaded_data.metadata["duration"] == 2.0
        assert loaded_data.metadata["preintegration_enabled"] == True
        
        # Verify calibrations
        assert len(loaded_data.calibration["cameras"]) == 1
        assert loaded_data.calibration["cameras"][0]["id"] == "cam0"
        assert len(loaded_data.calibration["imus"]) == 1
        assert loaded_data.calibration["imus"][0]["id"] == "imu0"
        
        # Verify trajectory
        assert len(loaded_data.groundtruth["trajectory"]) == 3
        assert loaded_data.groundtruth["trajectory"][0]["position"] == [0, 0, 0]
        assert loaded_data.groundtruth["trajectory"][0]["velocity"] == [1, 2, 0]
        
        # Verify landmarks
        assert len(loaded_data.groundtruth["landmarks"]) == 3
        assert loaded_data.groundtruth["landmarks"][1]["id"] == 1
        assert loaded_data.groundtruth["landmarks"][1]["position"] == [1, 2, 3]
        
        # Verify IMU measurements
        assert len(loaded_data.measurements["imu"]) == 400  # 200Hz for 2 seconds
        
        # Verify camera frames with keyframe info
        assert len(loaded_data.measurements["camera_frames"]) == 4
        assert loaded_data.measurements["camera_frames"][0]["is_keyframe"] == True
        assert loaded_data.measurements["camera_frames"][0]["keyframe_id"] == 0
        assert len(loaded_data.measurements["camera_frames"][0]["observations"]) == 1
        assert len(loaded_data.measurements["camera_frames"][3]["observations"]) == 3
        
        # Verify preintegrated IMU data
        assert len(loaded_data.measurements["preintegrated_imu"]) == 3
        assert loaded_data.measurements["preintegrated_imu"][0]["from_keyframe_id"] == 0
        assert loaded_data.measurements["preintegrated_imu"][0]["to_keyframe_id"] == 1
        assert loaded_data.measurements["preintegrated_imu"][0]["dt"] == 0.5
    
    def test_extract_trajectory(self):
        """Test extracting trajectory from loaded data."""
        sim_data = SimulationData()
        
        # Create and set trajectory
        trajectory = Trajectory()
        for t in [0.0, 1.0]:
            pose = Pose(
                timestamp=t,
                position=np.array([t, 0, 0]),
                rotation_matrix=np.eye(3)
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
            rotation_matrix=np.eye(3)
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
        """Test load_simulation_data convenience function with preintegrated IMU."""
        # Create and save test data with all components
        sim_data = SimulationData()
        sim_data.set_metadata(
            trajectory_type="test", 
            duration=1.0,
            preintegration_enabled=True
        )
        
        # Add trajectory
        trajectory = Trajectory()
        for t in [0.0, 0.5, 1.0]:
            pose = Pose(
                timestamp=t,
                position=np.array([t, 2*t, 3*t]),
                rotation_matrix=np.eye(3)
            )
            state = TrajectoryState(pose=pose)
            trajectory.add_state(state)
        sim_data.set_groundtruth_trajectory(trajectory)
        
        # Add camera frames with keyframes
        camera_data = CameraData(camera_id="cam0")
        for idx, t in enumerate([0.0, 0.5, 1.0]):
            frame = CameraFrame(
                timestamp=t,
                camera_id="cam0",
                observations=[],
                is_keyframe=True,
                keyframe_id=idx
            )
            camera_data.add_frame(frame)
        sim_data.add_camera_measurements(camera_data)
        
        # Add preintegrated IMU data
        preintegrated = {}
        for i in [1, 2]:  # Between keyframes 0-1 and 1-2
            preint_data = PreintegratedIMUData(
                from_keyframe_id=i-1,
                to_keyframe_id=i,
                delta_position=np.array([0.5, 1.0, 1.5]),
                delta_velocity=np.array([1.0, 2.0, 3.0]),
                delta_rotation=np.eye(3),
                covariance=np.eye(15),
                dt=0.5,
                num_measurements=100
            )
            preintegrated[i] = preint_data
        sim_data.set_preintegrated_imu(preintegrated)
        
        output_file = tmp_path / "test_load_complete.json"
        sim_data.save(output_file)
        
        # Load using convenience function
        loaded = load_simulation_data(output_file)
        
        # Check basic loading
        assert loaded["metadata"]["trajectory_type"] == "test"
        assert loaded["metadata"]["preintegration_enabled"] == True
        assert loaded["trajectory"] is not None
        assert len(loaded["trajectory"].states) == 3
        assert np.allclose(loaded["trajectory"].states[0].pose.position, [0, 0, 0])
        assert np.allclose(loaded["trajectory"].states[1].pose.position, [0.5, 1.0, 1.5])
        
        # Check preintegrated IMU is loaded
        assert loaded["preintegrated_imu"] is not None
        assert len(loaded["preintegrated_imu"]) == 2
        assert loaded["preintegrated_imu"][0].from_keyframe_id == 0
        assert loaded["preintegrated_imu"][0].to_keyframe_id == 1
        assert np.allclose(loaded["preintegrated_imu"][0].delta_position, [0.5, 1.0, 1.5])
        
        # Check camera data with keyframes
        assert loaded["camera_data"] is not None
        assert len(loaded["camera_data"].frames) == 3
        assert all(f.is_keyframe for f in loaded["camera_data"].frames)
        
        # IMPORTANT: Check that preintegrated IMU is attached to camera frames
        frames_with_preint = [
            f for f in loaded["camera_data"].frames 
            if hasattr(f, 'preintegrated_imu') and f.preintegrated_imu is not None
        ]
        assert len(frames_with_preint) == 2  # Keyframes 1 and 2 should have preintegrated data
        
        # Verify the attachment is correct
        kf1 = loaded["camera_data"].frames[1]  # Keyframe 1
        assert kf1.preintegrated_imu is not None
        assert kf1.preintegrated_imu.from_keyframe_id == 0
        assert kf1.preintegrated_imu.to_keyframe_id == 1
        assert np.allclose(kf1.preintegrated_imu.delta_position, [0.5, 1.0, 1.5])
        
        kf2 = loaded["camera_data"].frames[2]  # Keyframe 2
        assert kf2.preintegrated_imu is not None
        assert kf2.preintegrated_imu.from_keyframe_id == 1
        assert kf2.preintegrated_imu.to_keyframe_id == 2


    def test_preintegrated_imu_with_jacobian(self, tmp_path):
        """Test saving and loading preintegrated IMU with jacobian."""
        sim_data = SimulationData()
        
        # Create preintegrated IMU data with jacobian
        preintegrated = {}
        preint_data = PreintegratedIMUData(
            from_keyframe_id=0,
            to_keyframe_id=1,
            delta_position=np.array([1.0, 2.0, 3.0]),
            delta_velocity=np.array([0.1, 0.2, 0.3]),
            delta_rotation=np.eye(3),
            covariance=np.eye(15) * 0.001,
            dt=1.0,
            num_measurements=200,
            jacobian=np.random.randn(15, 6)  # Include jacobian
        )
        preintegrated[1] = preint_data
        
        sim_data.set_preintegrated_imu(preintegrated)
        
        # Save and load
        output_file = tmp_path / "test_jacobian.json"
        sim_data.save(output_file)
        
        loaded_data = SimulationData.load(output_file)
        extracted = loaded_data.get_preintegrated_imu()
        
        assert len(extracted) == 1
        assert extracted[0].jacobian is not None
        assert extracted[0].jacobian.shape == (15, 6)
        # Note: We can't test exact equality due to float precision in JSON
        assert np.allclose(extracted[0].delta_position, [1.0, 2.0, 3.0])
        assert np.allclose(extracted[0].delta_velocity, [0.1, 0.2, 0.3])
        assert extracted[0].dt == 1.0
        assert extracted[0].num_measurements == 200


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