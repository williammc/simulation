"""
Test that frame-to-frame preintegration has correct dt values.

This test specifically verifies that preintegrated IMU between consecutive
camera frames has dt values matching the frame interval (not keyframe interval).
"""

import numpy as np
import pytest
from src.utils.preintegration_utils import preintegrate_between_frames
from src.common.data_structures import IMUMeasurement


def test_frame_to_frame_dt_values():
    """Test that preintegration between frames has correct dt values."""
    
    # Create IMU measurements at 200Hz
    imu_measurements = []
    imu_rate = 200.0
    imu_dt = 1.0 / imu_rate
    
    for i in range(200):  # 1 second of IMU data
        t = i * imu_dt
        imu_measurements.append(IMUMeasurement(
            timestamp=t,
            accelerometer=np.array([0, 0, 9.81]),
            gyroscope=np.array([0, 0, 0.1])
        ))
    
    # Create frame times at 30Hz (camera frame rate)
    frame_rate = 30.0
    frame_dt = 1.0 / frame_rate
    frame_times = [i * frame_dt for i in range(31)]  # 1 second of frames
    
    # Preintegrate between consecutive frames
    result = preintegrate_between_frames(imu_measurements, frame_times)
    
    # Verify we got the right number of segments
    assert len(result) == len(frame_times) - 1, \
        f"Should have {len(frame_times)-1} segments, got {len(result)}"
    
    # Check each segment's dt
    for i, segment in enumerate(result):
        expected_dt = frame_times[i+1] - frame_times[i]
        
        # The dt should be close to the frame interval (0.033s for 30Hz)
        assert abs(segment.dt - expected_dt) < 1e-6, \
            f"Segment {i}: dt={segment.dt:.6f}, expected={expected_dt:.6f}"
        
        # dt should NOT be keyframe interval (0.5s)
        assert segment.dt < 0.1, \
            f"Segment {i}: dt={segment.dt} looks like keyframe interval!"
        
        # Should have approximately imu_rate * frame_dt measurements
        expected_measurements = int(imu_rate * expected_dt)
        # Allow for rounding
        assert abs(segment.num_measurements - expected_measurements) <= 1, \
            f"Segment {i}: has {segment.num_measurements} measurements, expected ~{expected_measurements}"
    
    # Specifically check the first few segments
    assert 0.030 <= result[0].dt <= 0.035, f"First segment dt={result[0].dt} out of range"
    assert 0.030 <= result[1].dt <= 0.035, f"Second segment dt={result[1].dt} out of range"
    
    # Check that num_measurements is reasonable (6-7 for 30Hz frames at 200Hz IMU)
    assert 6 <= result[0].num_measurements <= 8, \
        f"First segment has {result[0].num_measurements} measurements, expected 6-7"


def test_dt_field_exists_in_preintegrated_data():
    """Test that PreintegratedIMUData has dt field (not delta_t)."""
    from src.common.data_structures import PreintegratedIMUData
    
    # Create a preintegrated IMU data object
    data = PreintegratedIMUData(
        from_frame_id=0,
        to_frame_id=1,
        delta_position=np.zeros(3),
        delta_velocity=np.zeros(3),
        delta_rotation=np.eye(3),
        covariance=np.eye(15),
        dt=0.033,  # This should be the field name
        num_measurements=7
    )
    
    # Verify dt field exists and has correct value
    assert hasattr(data, 'dt'), "PreintegratedIMUData should have 'dt' field"
    assert data.dt == 0.033, f"dt should be 0.033, got {data.dt}"
    
    # Verify delta_t does NOT exist (common naming mistake)
    assert not hasattr(data, 'delta_t'), "PreintegratedIMUData should not have 'delta_t' field"


def test_serialization_preserves_dt():
    """Test that to_dict/from_dict preserves dt values correctly."""
    from src.common.data_structures import PreintegratedIMUData
    
    # Create test data with specific dt
    original = PreintegratedIMUData(
        from_frame_id=0,
        to_frame_id=1,
        delta_position=np.array([1, 2, 3]),
        delta_velocity=np.array([4, 5, 6]),
        delta_rotation=np.eye(3),
        covariance=np.eye(15),
        dt=0.0333,  # Frame interval
        num_measurements=7
    )
    
    # Serialize and deserialize
    dict_form = original.to_dict()
    restored = PreintegratedIMUData.from_dict(dict_form)
    
    # Check dt is preserved
    assert restored.dt == original.dt, \
        f"dt not preserved: original={original.dt}, restored={restored.dt}"
    
    # Verify it's not accidentally using keyframe interval
    assert restored.dt < 0.1, \
        f"Restored dt={restored.dt} looks like keyframe interval!"


if __name__ == "__main__":
    test_frame_to_frame_dt_values()
    test_dt_field_exists_in_preintegrated_data()
    test_serialization_preserves_dt()
    print("All tests passed!")