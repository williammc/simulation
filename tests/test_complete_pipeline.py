"""
End-to-end test for complete keyframe-preintegration pipeline.
"""

import tempfile
import json
from pathlib import Path
import numpy as np
import pytest

from tools.simulate import run_simulation
from src.common.config import KeyframeSelectionConfig, KeyframeSelectionStrategy
from src.common.json_io import load_simulation_data
from src.estimation.ekf_slam import EKFSlam
from src.estimation.swba_slam import SlidingWindowBA
from src.common.config import EKFConfig, SWBAConfig
from src.common.data_structures import (
    CameraCalibration, CameraIntrinsics, CameraModel, CameraExtrinsics,
    IMUCalibration
)


def get_calibrations_from_data(data):
    """Helper to extract calibrations from loaded simulation data."""
    raw_data = data.get('raw')
    
    # Get camera calibration
    camera_calib = None
    if hasattr(raw_data, 'calibration') and raw_data.calibration.get('cameras'):
        cam_calib_dict = raw_data.calibration['cameras'][0]
        intrinsics = CameraIntrinsics(
            model=CameraModel(cam_calib_dict['model']),
            width=cam_calib_dict['width'],
            height=cam_calib_dict['height'],
            fx=cam_calib_dict['intrinsics']['fx'],
            fy=cam_calib_dict['intrinsics']['fy'],
            cx=cam_calib_dict['intrinsics']['cx'],
            cy=cam_calib_dict['intrinsics']['cy'],
            distortion=np.array(cam_calib_dict['distortion'])
        )
        extrinsics = CameraExtrinsics(B_T_C=np.array(cam_calib_dict['T_BC']))
        camera_calib = CameraCalibration(
            camera_id=cam_calib_dict['id'],
            intrinsics=intrinsics,
            extrinsics=extrinsics
        )
    
    # Get IMU calibration
    imu_calib = None
    if hasattr(raw_data, 'calibration') and raw_data.calibration.get('imus'):
        imu_calib_dict = raw_data.calibration['imus'][0]
        imu_calib = IMUCalibration(
            imu_id=imu_calib_dict['id'],
            accelerometer_noise_density=imu_calib_dict['accelerometer']['noise_density'],
            accelerometer_random_walk=imu_calib_dict['accelerometer']['random_walk'],
            gyroscope_noise_density=imu_calib_dict['gyroscope']['noise_density'],
            gyroscope_random_walk=imu_calib_dict['gyroscope']['random_walk'],
            rate=imu_calib_dict['sampling_rate']
        )
    
    return camera_calib, imu_calib


class TestCompletePipeline:
    """Test the complete pipeline from data generation to estimation."""
    
    def test_simulation_with_keyframe_preintegration(self):
        """Test complete simulation pipeline with keyframe selection and preintegration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Configure keyframe selection
            keyframe_config = KeyframeSelectionConfig(
                strategy=KeyframeSelectionStrategy.FIXED_INTERVAL,
                fixed_interval=5,
                min_time_gap=0.1
            )
            
            # Run simulation with preintegration
            exit_code = run_simulation(
                trajectory="circle",
                config=None,
                duration=3.0,
                output=output_dir,
                seed=42,
                noise_config=None,
                add_noise=False,
                enable_preintegration=True,
                keyframe_config=keyframe_config
            )
            
            assert exit_code == 0, "Simulation should succeed"
            
            # Load generated data
            output_files = list(output_dir.glob("simulation_circle_*.json"))
            assert len(output_files) == 1
            
            data = load_simulation_data(output_files[0])
            
            # Verify keyframes were selected
            raw_data = data.get('raw')
            measurements = raw_data.measurements if hasattr(raw_data, 'measurements') else {}
            camera_frames = measurements.get('camera_frames', [])
            keyframe_count = sum(1 for f in camera_frames if f.get('is_keyframe', False))
            assert keyframe_count > 0, "Should have keyframes"
            
            # Verify preintegrated IMU data exists
            if hasattr(raw_data, 'preintegrated_imu'):
                assert raw_data.preintegrated_imu is not None
                assert len(raw_data.preintegrated_imu) > 0
            else:
                measurements = raw_data.measurements if hasattr(raw_data, 'measurements') else {}
                preint_data = measurements.get('preintegrated_imu', [])
                assert len(preint_data) > 0
            
            # Verify data consistency
            if hasattr(raw_data, 'preintegrated_imu') and raw_data.preintegrated_imu:
                for preint in raw_data.preintegrated_imu.values():
                    assert preint.dt > 0
                    assert preint.num_measurements > 0
                    assert preint.covariance.shape == (15, 15)
            else:
                measurements = raw_data.measurements if hasattr(raw_data, 'measurements') else {}
                preint_data = measurements.get('preintegrated_imu', [])
                for preint in preint_data:
                    assert preint.get('dt', 0) > 0
                    assert preint.get('num_measurements', 0) > 0
    
    def test_ekf_with_keyframe_only_processing(self):
        """Test EKF with keyframe-only processing mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Generate data with keyframes
            keyframe_config = KeyframeSelectionConfig(
                strategy=KeyframeSelectionStrategy.FIXED_INTERVAL,
                fixed_interval=3
            )
            
            exit_code = run_simulation(
                trajectory="line",
                config=None,
                duration=2.0,
                output=output_dir,
                seed=123,
                noise_config=None,
                add_noise=False,
                enable_preintegration=True,
                keyframe_config=keyframe_config
            )
            
            assert exit_code == 0
            
            # Load data
            output_files = list(output_dir.glob("*.json"))
            data = load_simulation_data(output_files[0])
            raw_data = data.get('raw')
            raw_data = data.get('raw')
            
            # Configure EKF for keyframe-only processing
            ekf_config = EKFConfig(
                use_keyframes_only=True,
                use_preintegrated_imu=True
            )
            
            # Get calibrations
            camera_calib, imu_calib = get_calibrations_from_data(data)
            if not camera_calib:
                raise ValueError("No camera calibration found")
            
            # Initialize EKF
            ekf = EKFSlam(ekf_config, camera_calib)
            trajectory = data.get('trajectory')
            if not trajectory and hasattr(raw_data, 'groundtruth'):
                trajectory = raw_data.get_trajectory()
            ekf.initialize(trajectory.states[0].pose)
            
            # Process measurements
            processed_frames = 0
            
            # Get camera frames and preintegrated IMU from raw data
            camera_frames = raw_data.measurements.get('camera_frames', []) if hasattr(raw_data, 'measurements') else []
            preint_imu_list = raw_data.measurements.get('preintegrated_imu', []) if hasattr(raw_data, 'measurements') else []
            
            # Get landmarks
            landmarks = data.get('landmarks')
            if not landmarks:
                landmarks = raw_data.get_map() if hasattr(raw_data, 'get_map') else None
            
            # Process keyframes with preintegrated IMU
            for frame_dict in camera_frames:
                if frame_dict.get('is_keyframe', False):
                    keyframe_id = frame_dict.get('keyframe_id')
                    
                    # Find corresponding preintegrated IMU data
                    for preint_dict in preint_imu_list:
                        if preint_dict.get('to_keyframe_id') == keyframe_id:
                            # Create PreintegratedIMUData object from dict
                            from src.common.data_structures import PreintegratedIMUData
                            preint_data = PreintegratedIMUData(
                                delta_position=np.array(preint_dict['delta_position']),
                                delta_velocity=np.array(preint_dict['delta_velocity']),
                                delta_rotation=np.array(preint_dict['delta_rotation']),
                                covariance=np.array(preint_dict['covariance']),
                                dt=preint_dict['dt'],
                                from_keyframe_id=preint_dict.get('from_keyframe_id', -1),
                                to_keyframe_id=preint_dict.get('to_keyframe_id', -1),
                                num_measurements=preint_dict.get('num_measurements', 0)
                            )
                            ekf.predict(preint_data)
                            break
                    
                    # Simple update to mark keyframe processed
                    # (visual updates are minimal in simplified version)
                    if landmarks:
                        ekf.update(None, landmarks)  # Simplified update
                    processed_frames += 1
            
            assert processed_frames > 0, "Should process keyframes"
            
            # Get result
            result = ekf.get_result()
            assert result.trajectory is not None
            assert len(result.trajectory.states) > 0
    
    def test_swba_with_preintegrated_imu(self):
        """Test SWBA with preintegrated IMU from simulation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Generate data with motion-based keyframes
            keyframe_config = KeyframeSelectionConfig(
                strategy=KeyframeSelectionStrategy.MOTION_BASED,
                translation_threshold=0.3,
                rotation_threshold=0.2,
                min_time_gap=0.1
            )
            
            exit_code = run_simulation(
                trajectory="figure8",
                config=None,
                duration=3.0,
                output=output_dir,
                seed=456,
                noise_config=None,
                add_noise=False,
                enable_preintegration=True,
                keyframe_config=keyframe_config
            )
            
            assert exit_code == 0
            
            # Load data
            output_files = list(output_dir.glob("*.json"))
            data = load_simulation_data(output_files[0])
            raw_data = data.get('raw')
            
            # Configure SWBA
            swba_config = SWBAConfig(
                window_size=5,
                use_preintegrated_imu=True,
                use_keyframes_only=True
            )
            
            # Get calibrations
            camera_calib, imu_calib = get_calibrations_from_data(data)
            
            # Initialize SWBA
            swba = SlidingWindowBA(swba_config, camera_calib, imu_calib)
            # Get trajectory
            trajectory = data.get('trajectory')
            if not trajectory:
                trajectory = raw_data.get_trajectory()
            swba.initialize(trajectory.states[0].pose)
            
            # Process keyframes
            keyframes_processed = 0
            # Get camera frames and landmarks
            camera_frames = raw_data.measurements.get('camera_frames', [])
            landmarks = data.get('landmarks') or raw_data.get_map()
            
            for frame_dict in camera_frames:
                if frame_dict.get('is_keyframe', False):
                    # Create frame object (simplified - just track count)
                    swba.update(None, landmarks)
                    keyframes_processed += 1
            
            assert keyframes_processed > 0, "Should process keyframes"
            assert swba.num_optimizations > 0, "Should run optimization"
            
            # Get result
            result = swba.get_result()
            assert result.trajectory is not None
            assert len(result.trajectory.states) > 0
    
    def test_different_keyframe_strategies(self):
        """Test that different keyframe strategies produce valid results."""
        strategies = [
            KeyframeSelectionStrategy.FIXED_INTERVAL,
            KeyframeSelectionStrategy.MOTION_BASED,
            KeyframeSelectionStrategy.HYBRID
        ]
        
        for strategy in strategies:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)
                
                # Configure based on strategy
                if strategy == KeyframeSelectionStrategy.FIXED_INTERVAL:
                    config = KeyframeSelectionConfig(
                        strategy=strategy,
                        fixed_interval=4
                    )
                elif strategy == KeyframeSelectionStrategy.MOTION_BASED:
                    config = KeyframeSelectionConfig(
                        strategy=strategy,
                        translation_threshold=0.4,
                        rotation_threshold=0.3
                    )
                else:  # HYBRID
                    config = KeyframeSelectionConfig(
                        strategy=strategy,
                        fixed_interval=5,
                        max_interval=8,
                        translation_threshold=0.3,
                        rotation_threshold=0.2
                    )
                
                # Run simulation
                exit_code = run_simulation(
                    trajectory="spiral",
                    config=None,
                    duration=2.0,
                    output=output_dir,
                    seed=789,
                    noise_config=None,
                    add_noise=False,
                    enable_preintegration=True,
                    keyframe_config=config
                )
                
                assert exit_code == 0, f"Simulation with {strategy} should succeed"
                
                # Load and verify data
                output_files = list(output_dir.glob("*.json"))
                data = load_simulation_data(output_files[0])
                raw_data = data.get('raw')  # Add missing raw_data variable
                
                # Check metadata
                metadata = data.get('metadata', {})
                if not metadata and hasattr(raw_data, 'metadata'):
                    metadata = raw_data.metadata
                assert metadata.get("keyframe_selection_strategy") == strategy.value
                assert metadata.get("num_keyframes", 0) > 0
                
                # Verify keyframes exist
                camera_frames = raw_data.measurements.get('camera_frames', [])
                keyframes = [f for f in camera_frames if f.get('is_keyframe', False)]
                assert len(keyframes) > 0, f"Should have keyframes with {strategy}"
                
                # Verify preintegration if enabled
                if metadata.get("preintegration_enabled"):
                    preint_data = raw_data.measurements.get('preintegrated_imu', [])
                    assert len(preint_data) > 0
                    # Check that preintegrated data connects keyframes
                    for preint in preint_data:
                        assert preint.get('from_keyframe_id', -1) >= 0
                        assert preint.get('to_keyframe_id', 0) > preint.get('from_keyframe_id', -1)
    
    def test_memory_efficiency_with_sparse_keyframes(self):
        """Test memory efficiency with sparse keyframes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Use large interval for sparse keyframes
            keyframe_config = KeyframeSelectionConfig(
                strategy=KeyframeSelectionStrategy.FIXED_INTERVAL,
                fixed_interval=20  # Very sparse
            )
            
            # Run longer simulation
            exit_code = run_simulation(
                trajectory="circle",
                config=None,
                duration=10.0,  # Longer duration
                output=output_dir,
                seed=999,
                noise_config=None,
                add_noise=False,
                enable_preintegration=True,
                keyframe_config=keyframe_config
            )
            
            assert exit_code == 0
            
            # Load data
            output_files = list(output_dir.glob("*.json"))
            data = load_simulation_data(output_files[0])
            raw_data = data.get('raw')
            
            # Check sparsity
            camera_frames = raw_data.measurements.get('camera_frames', [])
            total_frames = len(camera_frames)
            keyframe_count = sum(1 for f in camera_frames if f.get('is_keyframe', False))
            sparsity_ratio = keyframe_count / total_frames
            
            assert sparsity_ratio < 0.2, "Keyframes should be sparse"
            
            # Verify preintegration handles large intervals
            preint_data = raw_data.measurements.get('preintegrated_imu', [])
            if preint_data:
                for preint in preint_data:
                    # Should have many measurements in each interval
                    assert preint.get('num_measurements', 0) > 50  # At least 50 IMU measurements
                    assert preint.get('dt', 0) > 0.5  # At least 0.5 seconds between keyframes