"""
Unit tests for camera measurement model.
"""

import pytest
import numpy as np

from src.estimation.camera_model import (
    CameraMeasurementModel, ReprojectionError,
    batch_compute_reprojection_errors
)
from src.common.data_structures import (
    CameraCalibration, CameraIntrinsics, CameraExtrinsics,
    CameraModel, CameraObservation, ImagePoint,
    Pose, Landmark
)


class TestCameraMeasurementModel:
    """Test camera measurement model."""
    
    @pytest.fixture
    def simple_calibration(self):
        """Create simple pinhole camera calibration."""
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
            B_T_C=np.eye(4)  # Camera aligned with body
        )
        
        return CameraCalibration(
            camera_id="cam0",
            intrinsics=intrinsics,
            extrinsics=extrinsics
        )
    
    @pytest.fixture
    def camera_pose(self):
        """Create camera pose at origin."""
        return Pose(
            timestamp=0.0,
            position=np.zeros(3),
            quaternion=np.array([1, 0, 0, 0])
        )
    
    def test_projection_simple(self, simple_calibration, camera_pose):
        """Test simple projection without distortion."""
        model = CameraMeasurementModel(simple_calibration)
        
        # Point directly in front of camera
        landmark = np.array([0, 0, 1])
        
        pixel, J_pose, J_landmark = model.project(
            landmark, camera_pose, compute_jacobian=False
        )
        
        assert pixel is not None
        assert np.isclose(pixel.u, 320.0)  # Center of image
        assert np.isclose(pixel.v, 240.0)
    
    def test_projection_with_translation(self, simple_calibration):
        """Test projection with camera translation."""
        model = CameraMeasurementModel(simple_calibration)
        
        # Camera at origin looking forward
        camera_pose = Pose(
            timestamp=0.0,
            position=np.array([0, 0, 0]),
            quaternion=np.array([1, 0, 0, 0])
        )
        
        # Landmark to the right and in front
        landmark = np.array([1, 0, 5])
        
        pixel, _, _ = model.project(landmark, camera_pose)
        
        # Should project to right side of image
        assert pixel is not None
        assert pixel.u > 320.0  # Right of center
    
    def test_projection_behind_camera(self, simple_calibration, camera_pose):
        """Test that points behind camera return None."""
        model = CameraMeasurementModel(simple_calibration)
        
        # Point behind camera
        landmark = np.array([0, 0, -1])
        
        pixel, _, _ = model.project(landmark, camera_pose)
        
        assert pixel is None
    
    def test_projection_outside_image(self, simple_calibration, camera_pose):
        """Test that points projecting outside image return None."""
        model = CameraMeasurementModel(simple_calibration)
        
        # Point far to the right
        landmark = np.array([100, 0, 1])
        
        pixel, _, _ = model.project(landmark, camera_pose)
        
        assert pixel is None
    
    def test_unprojection(self, simple_calibration, camera_pose):
        """Test unprojection with known depth."""
        model = CameraMeasurementModel(simple_calibration)
        
        # Center pixel with depth 5
        pixel = ImagePoint(u=320.0, v=240.0)
        depth = 5.0
        
        point_3d = model.unproject(pixel, depth, camera_pose)
        
        # Should be directly in front at depth 5
        assert np.allclose(point_3d, [0, 0, 5])
    
    def test_projection_unprojection_consistency(self, simple_calibration, camera_pose):
        """Test that projection followed by unprojection recovers the point."""
        model = CameraMeasurementModel(simple_calibration)
        
        # Original 3D point
        landmark = np.array([1, 2, 5])
        
        # Project to image
        pixel, _, _ = model.project(landmark, camera_pose)
        assert pixel is not None
        
        # Unproject with correct depth
        depth = 5.0  # z-coordinate
        recovered = model.unproject(pixel, depth, camera_pose)
        
        # Should recover original point
        assert np.allclose(recovered, landmark, atol=1e-10)
    
    def test_jacobian_dimensions(self, simple_calibration, camera_pose):
        """Test that Jacobians have correct dimensions."""
        model = CameraMeasurementModel(simple_calibration)
        
        landmark = np.array([1, 1, 5])
        
        pixel, J_pose, J_landmark = model.project(
            landmark, camera_pose, compute_jacobian=True
        )
        
        assert pixel is not None
        assert J_pose.shape == (2, 6)  # 2D pixel, 6 DOF pose
        assert J_landmark.shape == (2, 3)  # 2D pixel, 3D landmark
    
    def test_jacobian_finite_difference(self, simple_calibration):
        """Test Jacobians using finite differences."""
        model = CameraMeasurementModel(simple_calibration)
        
        # Test point and pose - make sure landmark is in front of camera
        landmark = np.array([0.5, 0.5, 10])  # Well in front of camera
        pose = Pose(
            timestamp=0.0,
            position=np.array([0, 0, 0]),  # Camera at origin
            quaternion=np.array([1, 0, 0, 0])  # No rotation
        )
        
        # Analytical Jacobians
        pixel, J_pose, J_landmark = model.project(
            landmark, pose, compute_jacobian=True
        )
        
        # Make sure projection succeeded
        assert pixel is not None, "Landmark should be visible"
        assert J_landmark is not None, "Jacobian should be computed"
        
        # Finite difference for landmark Jacobian
        eps = 1e-6
        J_landmark_fd = np.zeros((2, 3))
        
        for i in range(3):
            landmark_plus = landmark.copy()
            landmark_plus[i] += eps
            
            pixel_plus, _, _ = model.project(landmark_plus, pose)
            
            if pixel_plus is not None:
                J_landmark_fd[0, i] = (pixel_plus.u - pixel.u) / eps
                J_landmark_fd[1, i] = (pixel_plus.v - pixel.v) / eps
        
        # Check agreement (allow some numerical error)
        assert np.allclose(J_landmark, J_landmark_fd, rtol=1e-4, atol=1e-6)
    
    def test_reprojection_error(self, simple_calibration, camera_pose):
        """Test reprojection error computation."""
        model = CameraMeasurementModel(simple_calibration)
        
        # Create landmark
        landmark = Landmark(id=0, position=np.array([1, 1, 5]))
        
        # Create observation (project first to get ground truth)
        true_pixel, _, _ = model.project(landmark.position, camera_pose)
        
        # Add small error
        observed_pixel = ImagePoint(
            u=true_pixel.u + 2.0,
            v=true_pixel.v - 1.0
        )
        observation = CameraObservation(
            landmark_id=0,
            pixel=observed_pixel
        )
        
        # Compute reprojection error
        error = model.compute_reprojection_error(
            observation, landmark, camera_pose
        )
        
        # Check residual
        assert np.allclose(error.residual, [2.0, -1.0])
        assert np.isclose(error.squared_error, 5.0)
    
    def test_robust_kernel_huber(self, simple_calibration, camera_pose):
        """Test Huber robust kernel."""
        model = CameraMeasurementModel(simple_calibration)
        
        landmark = Landmark(id=0, position=np.array([0, 0, 5]))
        
        # Large error observation
        observation = CameraObservation(
            landmark_id=0,
            pixel=ImagePoint(u=400.0, v=240.0)  # 80 pixels off
        )
        
        # Without robust kernel
        error_normal = model.compute_reprojection_error(
            observation, landmark, camera_pose
        )
        
        # With Huber kernel
        error_robust = model.compute_reprojection_error(
            observation, landmark, camera_pose,
            robust_kernel='huber',
            kernel_threshold=5.0
        )
        
        # Robust error should be smaller
        assert error_robust.squared_error < error_normal.squared_error
    
    def test_distortion_model(self):
        """Test radial-tangential distortion model."""
        # Create calibration with distortion
        intrinsics = CameraIntrinsics(
            model=CameraModel.PINHOLE,
            width=640,
            height=480,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            distortion=np.array([0.1, -0.05, 0.01, -0.01])  # k1, k2, p1, p2
        )
        
        extrinsics = CameraExtrinsics(B_T_C=np.eye(4))
        calibration = CameraCalibration(
            camera_id="cam0",
            intrinsics=intrinsics,
            extrinsics=extrinsics
        )
        
        model = CameraMeasurementModel(calibration)
        
        # Test distortion application
        x_n, y_n = 0.5, 0.3
        x_d, y_d = model._apply_distortion(x_n, y_n)
        
        # Distorted coordinates should be different
        assert not np.isclose(x_d, x_n)
        assert not np.isclose(y_d, y_n)
        
        # Test distortion removal (should recover original)
        x_recovered, y_recovered = model._remove_distortion(x_d, y_d)
        
        assert np.isclose(x_recovered, x_n, atol=1e-6)
        assert np.isclose(y_recovered, y_n, atol=1e-6)


class TestBatchOperations:
    """Test batch reprojection operations."""
    
    def test_batch_reprojection(self):
        """Test batch reprojection error computation."""
        # Setup
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
        
        extrinsics = CameraExtrinsics(B_T_C=np.eye(4))
        calibration = CameraCalibration(
            camera_id="cam0",
            intrinsics=intrinsics,
            extrinsics=extrinsics
        )
        
        camera_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            quaternion=np.array([1, 0, 0, 0])
        )
        
        # Create multiple observations and landmarks
        observations = []
        landmarks = []
        
        for i in range(5):
            landmark = Landmark(
                id=i,
                position=np.array([i-2, 0, 5])  # Spread horizontally
            )
            landmarks.append(landmark)
            
            # Create perfect observation
            model = CameraMeasurementModel(calibration)
            pixel, _, _ = model.project(landmark.position, camera_pose)
            
            observation = CameraObservation(
                landmark_id=i,
                pixel=pixel
            )
            observations.append(observation)
        
        # Compute batch errors
        residuals, J_pose, J_landmarks, total_error = batch_compute_reprojection_errors(
            observations, landmarks, camera_pose, calibration
        )
        
        # Check dimensions
        assert residuals.shape == (10,)  # 5 observations * 2 components
        assert J_pose.shape == (10, 6)
        assert J_landmarks.shape == (10, 15)  # 5 landmarks * 3 components
        
        # Perfect observations should have zero error
        assert np.allclose(residuals, 0)
        assert np.isclose(total_error, 0)
    
    def test_batch_with_outliers(self):
        """Test batch computation with outlier observations."""
        # Setup calibration
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
        
        extrinsics = CameraExtrinsics(B_T_C=np.eye(4))
        calibration = CameraCalibration(
            camera_id="cam0",
            intrinsics=intrinsics,
            extrinsics=extrinsics
        )
        
        camera_pose = Pose(
            timestamp=0.0,
            position=np.zeros(3),
            quaternion=np.array([1, 0, 0, 0])
        )
        
        # Create observations with one outlier
        observations = []
        landmarks = []
        
        landmark = Landmark(id=0, position=np.array([0, 0, 5]))
        landmarks.append(landmark)
        
        # Good observation
        good_obs = CameraObservation(
            landmark_id=0,
            pixel=ImagePoint(u=320.0, v=240.0)
        )
        observations.append(good_obs)
        
        # Outlier observation
        outlier_obs = CameraObservation(
            landmark_id=0,
            pixel=ImagePoint(u=400.0, v=300.0)  # Way off
        )
        observations.append(outlier_obs)
        
        # Without robust kernel
        residuals_normal, _, _, error_normal = batch_compute_reprojection_errors(
            observations, [landmark, landmark], camera_pose, calibration
        )
        
        # With robust kernel
        residuals_robust, _, _, error_robust = batch_compute_reprojection_errors(
            observations, [landmark, landmark], camera_pose, calibration,
            robust_kernel='huber', kernel_threshold=5.0
        )
        
        # Robust version should have lower total error
        assert error_robust < error_normal