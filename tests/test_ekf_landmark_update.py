"""
Test landmark update functionality in EKF SLAM.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.estimation.legacy.ekf_slam import EKFSlam
from src.common.config import EKFConfig
from src.common.data_structures import (
    CameraFrame, CameraObservation, ImagePoint,
    Pose, Map, Landmark,
    CameraCalibration, CameraIntrinsics, CameraExtrinsics,
    CameraModel, IMUCalibration
)


def test_landmark_update():
    """Test that EKF correctly updates state using landmark observations."""
    
    # Create camera calibration
    intrinsics = CameraIntrinsics(
        model=CameraModel.PINHOLE,
        width=640,
        height=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        distortion=np.zeros(4)
    )
    extrinsics = CameraExtrinsics(
        B_T_C=np.eye(4)  # Camera at body frame origin
    )
    camera_calib = CameraCalibration(
        camera_id="cam0",
        intrinsics=intrinsics,
        extrinsics=extrinsics
    )
    
    # Create IMU calibration
    imu_calib = IMUCalibration(
        imu_id="imu0",
        accelerometer_noise_density=0.01,
        gyroscope_noise_density=0.001,
        accelerometer_random_walk=0.0001,
        gyroscope_random_walk=0.00001
    )
    
    # Create EKF config
    config = EKFConfig(
        gravity_magnitude=9.81,
        pixel_noise_std=2.0,  # Pixel noise
        chi2_threshold=9.21,  # Chi2 95% threshold for 2 DOF
    )
    
    # Create EKF
    ekf = EKFSlam(config, camera_calib, imu_calib)
    
    # Initialize at origin
    initial_pose = Pose(
        timestamp=0.0,
        position=np.array([0, 0, 0]),
        rotation_matrix=np.eye(3)
    )
    ekf.initialize(initial_pose)
    
    # Create a map with known landmarks
    landmarks_map = Map()
    
    # Add some landmarks in front of the camera
    landmark1 = Landmark(
        id=1,
        position=np.array([5.0, 0.0, 0.0])  # 5m in front
    )
    landmark2 = Landmark(
        id=2,
        position=np.array([5.0, 1.0, 0.0])  # 5m front, 1m right
    )
    landmark3 = Landmark(
        id=3,
        position=np.array([5.0, -1.0, 1.0])  # 5m front, 1m left, 1m up
    )
    
    landmarks_map.add_landmark(landmark1)
    landmarks_map.add_landmark(landmark2)
    landmarks_map.add_landmark(landmark3)
    
    # Create camera observations of these landmarks
    # For a landmark at [5, 0, 0], with camera at origin looking along +X:
    # In camera frame (assuming Z forward, X right, Y down convention):
    # We need to transform: camera looks along +X in body frame
    # So body X -> camera Z, body Y -> camera X, body Z -> camera Y
    
    observations = []
    
    # Project landmark1 [5, 0, 0] 
    # In simple pinhole: u = fx * y/x + cx, v = fy * z/x + cy
    # With our setup (camera looking along +X): u = 320, v = 240 (center)
    obs1 = CameraObservation(
        landmark_id=1,
        pixel=ImagePoint(u=320.0, v=240.0)  # Center of image
    )
    observations.append(obs1)
    
    # Project landmark2 [5, 1, 0]
    # u = fx * y/x + cx = 500 * 1/5 + 320 = 420
    obs2 = CameraObservation(
        landmark_id=2,
        pixel=ImagePoint(u=420.0, v=240.0)  # Right of center
    )
    observations.append(obs2)
    
    # Project landmark3 [5, -1, 1]
    # u = fx * y/x + cx = 500 * (-1)/5 + 320 = 220
    # v = fy * z/x + cy = 500 * 1/5 + 240 = 340
    obs3 = CameraObservation(
        landmark_id=3,
        pixel=ImagePoint(u=220.0, v=340.0)  # Left and down
    )
    observations.append(obs3)
    
    # Create camera frame
    camera_frame = CameraFrame(
        timestamp=0.1,
        camera_id="cam0",
        observations=observations,
        is_keyframe=True,
        keyframe_id=0
    )
    
    print("Initial state:")
    print(f"  Position: {ekf.state.position}")
    print(f"  Velocity: {ekf.state.velocity}")
    print(f"  Rotation:\n{ekf.state.rotation_matrix}")
    
    # Artificially perturb the state to test correction
    ekf.state.position = np.array([0.1, -0.05, 0.02])  # Small error
    
    print("\nPerturbed state:")
    print(f"  Position: {ekf.state.position}")
    
    # Perform update with landmarks
    print("\nPerforming update...")
    
    # Debug: manually test projection for landmark 1
    predicted, jacobian = ekf._predict_measurement(
        landmark1.position,
        ekf.state.position,
        ekf.state.rotation_matrix
    )
    print(f"  Landmark 1 projection: {predicted}")
    print(f"  Jacobian shape: {jacobian.shape if jacobian is not None else None}")
    
    ekf.update(camera_frame, landmarks_map)
    
    print("\nAfter update:")
    print(f"  Position: {ekf.state.position}")
    print(f"  Velocity: {ekf.state.velocity}")
    print(f"  Num landmarks tracked: {len(ekf.landmarks)}")
    print(f"  Num updates: {ekf.num_updates}")
    print(f"  Num outliers: {ekf.num_outliers}")
    
    # Check that position was corrected towards truth
    position_error = np.linalg.norm(ekf.state.position)
    print(f"\nFinal position error: {position_error:.4f} m")
    
    # The update should have reduced the error
    assert position_error < 0.05, f"Position error too large after update: {position_error}"
    
    # Check that landmarks were stored (at least 2 out of 3)
    assert len(ekf.landmarks) >= 2, f"Expected at least 2 landmarks, got {len(ekf.landmarks)}"
    
    # Check that update was counted
    assert ekf.num_updates == 1, f"Expected 1 update, got {ekf.num_updates}"
    
    print("\n✓ Test passed: EKF landmark update working correctly")
    return True


if __name__ == "__main__":
    test_landmark_update()