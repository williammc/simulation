"""
Tests for stereo camera functionality.
"""

import pytest
import numpy as np

from src.estimation.stereo_camera import (
    StereoCameraModel, StereoCalibration, StereoObservation,
    create_stereo_calibration
)
from src.common.data_structures import (
    Pose, ImagePoint, Landmark,
    CameraIntrinsics, CameraExtrinsics, CameraModel
)


class TestStereoCalibration:
    """Test stereo camera calibration."""
    
    def test_create_stereo_calibration(self):
        """Test stereo calibration creation with default parameters."""
        calib = create_stereo_calibration(
            baseline=0.12,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0
        )
        
        assert isinstance(calib, StereoCalibration)
        assert calib.baseline == 0.12
        assert calib.left_calib.intrinsics.fx == 500.0
        assert calib.right_calib.intrinsics.fx == 500.0
        assert calib.left_calib.intrinsics.cx == 320.0
        assert calib.right_calib.intrinsics.cx == 320.0
    
    def test_stereo_transformation_matrix(self):
        """Test that stereo transformation matrix is correct."""
        baseline = 0.15
        calib = create_stereo_calibration(baseline=baseline)
        
        # Right camera should be translated by -baseline in x
        assert np.allclose(calib.T_RL[0, 3], -baseline)
        # No rotation between cameras
        assert np.allclose(calib.T_RL[:3, :3], np.eye(3))
        # Homogeneous coordinates
        assert np.allclose(calib.T_RL[3, :], [0, 0, 0, 1])
    
    def test_custom_image_size(self):
        """Test stereo calibration with custom image size."""
        calib = create_stereo_calibration(
            baseline=0.1,
            fx=600.0,
            fy=600.0,
            cx=640.0,
            cy=360.0,
            width=1280,
            height=720
        )
        
        assert calib.left_calib.intrinsics.width == 1280
        assert calib.left_calib.intrinsics.height == 720
        assert calib.right_calib.intrinsics.width == 1280
        assert calib.right_calib.intrinsics.height == 720
    
    def test_calibration_with_distortion(self):
        """Test stereo calibration with distortion parameters."""
        distortion = np.array([0.1, -0.05, 0.001, 0.001, 0.02])
        calib = create_stereo_calibration(
            baseline=0.12,
            distortion=distortion
        )
        
        np.testing.assert_array_equal(
            calib.left_calib.intrinsics.distortion,
            distortion
        )
        np.testing.assert_array_equal(
            calib.right_calib.intrinsics.distortion,
            distortion
        )


class TestStereoProjection:
    """Test stereo camera projection."""
    
    @pytest.fixture
    def stereo_model(self):
        """Create a stereo camera model for testing."""
        calib = create_stereo_calibration(
            baseline=0.1,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0
        )
        return StereoCameraModel(calib)
    
    def test_project_point_in_front(self, stereo_model):
        """Test projecting a point directly in front of the camera."""
        landmark = np.array([0, 0, 5])  # 5 meters in front
        pose = Pose(
            timestamp=0.0,
            position=np.array([0, 0, 0]),
            rotation_matrix=np.eye(3)  # Identity
        )
        
        stereo_obs, left_depth, right_depth = stereo_model.project_stereo(
            landmark, pose, add_noise=False
        )
        
        assert stereo_obs is not None
        assert isinstance(stereo_obs, StereoObservation)
        
        # Point should project to image center
        assert abs(stereo_obs.left_pixel.u - 320) < 1
        assert abs(stereo_obs.left_pixel.v - 240) < 1
        
        # Right pixel should have disparity
        expected_disparity = (0.1 * 500.0) / 5.0  # baseline * fx / depth
        actual_disparity = stereo_obs.left_pixel.u - stereo_obs.right_pixel.u
        assert abs(actual_disparity - expected_disparity) < 1
        
        # Depths should be positive and equal
        assert left_depth > 0
        assert right_depth > 0
        assert abs(left_depth - 5.0) < 0.1
    
    def test_project_off_center_point(self, stereo_model):
        """Test projecting a point off-center."""
        landmark = np.array([1, 0.5, 4])  # Off-center point
        pose = Pose(
            timestamp=0.0,
            position=np.array([0, 0, 0]),
            rotation_matrix=np.eye(3)
        )
        
        stereo_obs, _, _ = stereo_model.project_stereo(
            landmark, pose, add_noise=False
        )
        
        assert stereo_obs is not None
        
        # Point should be to the right of center
        assert stereo_obs.left_pixel.u > 320
        # Point should be below center (positive Y is down in image)
        assert stereo_obs.left_pixel.v > 240
        
        # Disparity should still be positive
        assert stereo_obs.disparity > 0
    
    def test_project_behind_camera(self, stereo_model):
        """Test that points behind camera are not projected."""
        landmark = np.array([0, 0, -1])  # Behind camera
        pose = Pose(
            timestamp=0.0,
            position=np.array([0, 0, 0]),
            rotation_matrix=np.eye(3)
        )
        
        stereo_obs, _, _ = stereo_model.project_stereo(
            landmark, pose, add_noise=False
        )
        
        assert stereo_obs is None  # Should not project
    
    def test_project_with_camera_pose(self, stereo_model):
        """Test projection with non-identity camera pose."""
        landmark = np.array([1, 0, 5])  # Point in front of the translated camera
        
        # Camera at (1, 0, 0) looking along positive X (same as initial orientation)
        pose = Pose(
            timestamp=0.0,
            position=np.array([1, 0, 0]),
            rotation_matrix=np.eye(3)  # Identity rotation
        )
        
        stereo_obs, _, _ = stereo_model.project_stereo(
            landmark, pose, add_noise=False
        )
        
        assert stereo_obs is not None
        # The point should now appear in front of the rotated camera
    
    def test_project_with_noise(self, stereo_model):
        """Test projection with noise."""
        landmark = np.array([0, 0, 5])
        pose = Pose(
            timestamp=0.0,
            position=np.array([0, 0, 0]),
            rotation_matrix=np.eye(3)
        )
        
        # Project multiple times with noise
        observations = []
        for _ in range(10):
            stereo_obs, _, _ = stereo_model.project_stereo(
                landmark, pose, add_noise=True
            )
            if stereo_obs:
                observations.append(stereo_obs)
        
        assert len(observations) > 0
        
        # Check that noise causes variation
        left_u_values = [obs.left_pixel.u for obs in observations]
        if len(left_u_values) > 1:
            assert np.std(left_u_values) > 0  # Should have some variation


class TestStereoTriangulation:
    """Test stereo triangulation."""
    
    @pytest.fixture
    def stereo_model(self):
        """Create a stereo camera model for testing."""
        calib = create_stereo_calibration(
            baseline=0.1,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0
        )
        return StereoCameraModel(calib)
    
    def test_triangulate_perfect_observation(self, stereo_model):
        """Test triangulation with perfect stereo observation."""
        # Create observation with known disparity
        disparity = 10.0  # pixels
        stereo_obs = StereoObservation(
            left_pixel=ImagePoint(u=320, v=240),
            right_pixel=ImagePoint(u=310, v=240),  # 10 pixel disparity
            landmark_id=0,
            timestamp=0.0
        )
        
        pose = Pose(
            timestamp=0.0,
            position=np.array([0, 0, 0]),
            rotation_matrix=np.eye(3)
        )
        
        # Triangulate
        point_3d, uncertainty = stereo_model.triangulate(stereo_obs, pose)
        
        assert point_3d is not None
        assert uncertainty is not None
        
        # Check depth from disparity formula: Z = baseline * fx / disparity
        expected_depth = (0.1 * 500.0) / 10.0  # = 5.0 meters
        assert abs(point_3d[2] - expected_depth) < 0.1
        
        # Point should be at optical center in X and Y
        assert abs(point_3d[0]) < 0.1
        assert abs(point_3d[1]) < 0.1
    
    def test_triangulate_off_center(self, stereo_model):
        """Test triangulation of off-center point."""
        stereo_obs = StereoObservation(
            left_pixel=ImagePoint(u=400, v=300),  # Right and down from center
            right_pixel=ImagePoint(u=385, v=300),  # 15 pixel disparity
            landmark_id=0,
            timestamp=0.0
        )
        
        pose = Pose(
            timestamp=0.0,
            position=np.array([0, 0, 0]),
            rotation_matrix=np.eye(3)
        )
        
        point_3d, _ = stereo_model.triangulate(stereo_obs, pose)
        
        assert point_3d is not None
        # Point should be to the right (positive X)
        assert point_3d[0] > 0
        # Point should be below (positive Y in 3D for down in image)
        assert point_3d[1] > 0
        # Depth should be positive
        assert point_3d[2] > 0
    
    def test_triangulate_with_pose(self, stereo_model):
        """Test triangulation with non-identity camera pose."""
        stereo_obs = StereoObservation(
            left_pixel=ImagePoint(u=320, v=240),
            right_pixel=ImagePoint(u=310, v=240),
            landmark_id=0,
            timestamp=0.0
        )
        
        # Camera translated and rotated
        pose = Pose(
            timestamp=0.0,
            position=np.array([1, 2, 3]),
            rotation_matrix=np.eye(3)  # Identity rotation for simplicity
        )
        
        point_3d, _ = stereo_model.triangulate(stereo_obs, pose)
        
        assert point_3d is not None
        # Point should be offset by camera position
        assert point_3d[0] > 0.9  # Near 1
        assert point_3d[1] > 1.9  # Near 2
        assert point_3d[2] > 3.0  # Greater than camera Z
    
    def test_triangulation_uncertainty(self, stereo_model):
        """Test that triangulation uncertainty increases with distance."""
        pose = Pose(
            timestamp=0.0,
            position=np.array([0, 0, 0]),
            rotation_matrix=np.eye(3)
        )
        
        # Near point (large disparity)
        near_obs = StereoObservation(
            left_pixel=ImagePoint(u=320, v=240),
            right_pixel=ImagePoint(u=270, v=240),  # 50 pixel disparity
            landmark_id=0,
            timestamp=0.0
        )
        
        # Far point (small disparity)
        far_obs = StereoObservation(
            left_pixel=ImagePoint(u=320, v=240),
            right_pixel=ImagePoint(u=318, v=240),  # 2 pixel disparity
            landmark_id=0,
            timestamp=0.0
        )
        
        _, near_uncertainty = stereo_model.triangulate(near_obs, pose)
        _, far_uncertainty = stereo_model.triangulate(far_obs, pose)
        
        assert near_uncertainty is not None
        assert far_uncertainty is not None
        
        # Far point should have higher uncertainty
        assert np.trace(far_uncertainty) > np.trace(near_uncertainty)


class TestStereoReprojectionError:
    """Test stereo reprojection error computation."""
    
    @pytest.fixture
    def stereo_model(self):
        """Create a stereo camera model for testing."""
        calib = create_stereo_calibration(
            baseline=0.1,
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0
        )
        return StereoCameraModel(calib)
    
    def test_zero_reprojection_error(self, stereo_model):
        """Test reprojection error for perfect observation."""
        # Create a landmark
        landmark = Landmark(id=0, position=np.array([0, 0, 5]))
        
        pose = Pose(
            timestamp=0.0,
            position=np.array([0, 0, 0]),
            rotation_matrix=np.eye(3)
        )
        
        # Project the landmark to get perfect observation
        stereo_obs, _, _ = stereo_model.project_stereo(
            landmark.position, pose, add_noise=False
        )
        stereo_obs.landmark_id = 0
        
        # Compute reprojection error
        left_error, right_error = stereo_model.compute_stereo_reprojection_error(
            stereo_obs, landmark, pose
        )
        
        assert left_error is not None
        assert right_error is not None
        assert hasattr(left_error, 'residual')
        assert hasattr(right_error, 'residual')
        
        # Error should be near zero
        assert np.linalg.norm(left_error.residual) < 1e-10
        assert np.linalg.norm(right_error.residual) < 1e-10
    
    def test_nonzero_reprojection_error(self, stereo_model):
        """Test reprojection error with incorrect observation."""
        landmark = Landmark(id=0, position=np.array([1, 1, 5]))
        
        # Create observation that doesn't match the landmark
        stereo_obs = StereoObservation(
            left_pixel=ImagePoint(u=400, v=300),  # Wrong pixels
            right_pixel=ImagePoint(u=390, v=300),
            landmark_id=0,
            timestamp=0.0
        )
        
        pose = Pose(
            timestamp=0.0,
            position=np.array([0, 0, 0]),
            rotation_matrix=np.eye(3)
        )
        
        left_error, right_error = stereo_model.compute_stereo_reprojection_error(
            stereo_obs, landmark, pose
        )
        
        # Errors should be non-zero
        assert np.linalg.norm(left_error.residual) > 1.0
        assert np.linalg.norm(right_error.residual) > 1.0
        
        # Jacobians should be computed
        assert left_error.jacobian_pose is not None
        assert left_error.jacobian_landmark is not None
        assert right_error.jacobian_pose is not None
        assert right_error.jacobian_landmark is not None
    
    def test_reprojection_error_jacobians(self, stereo_model):
        """Test that reprojection error Jacobians have correct shape."""
        landmark = Landmark(id=0, position=np.array([1, 0, 3]))
        
        stereo_obs = StereoObservation(
            left_pixel=ImagePoint(u=320, v=240),
            right_pixel=ImagePoint(u=310, v=240),
            landmark_id=0,
            timestamp=0.0
        )
        
        pose = Pose(
            timestamp=0.0,
            position=np.array([0, 0, 0]),
            rotation_matrix=np.eye(3)
        )
        
        left_error, right_error = stereo_model.compute_stereo_reprojection_error(
            stereo_obs, landmark, pose
        )
        
        # Check Jacobian shapes
        # Residual is 2D (u, v), pose is 6D (position + orientation)
        assert left_error.jacobian_pose.shape == (2, 6)
        assert right_error.jacobian_pose.shape == (2, 6)
        
        # Landmark is 3D
        assert left_error.jacobian_landmark.shape == (2, 3)
        assert right_error.jacobian_landmark.shape == (2, 3)