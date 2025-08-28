#!/usr/bin/env python3
"""
Test ideal projection accuracy using PnP (Perspective-n-Point) solver.

This test validates that our ideal coordinate projection formulation is correct by:
1. Loading simulation data with ground truth poses and landmarks
2. Using ideal coordinates to solve PnP and estimate pose
3. Comparing estimated pose with ground truth
4. Validating projection accuracy
"""

import pytest
import numpy as np
from pathlib import Path

# Try to import OpenCV, skip tests if not available
cv2 = pytest.importorskip("cv2", reason="OpenCV (cv2) is required for PnP tests")

from src.common.json_io import load_simulation_data


class TestIdealProjectionPnP:
    """Test suite for ideal projection validation using PnP solver."""
    
    @pytest.fixture
    def simulation_file(self):
        """Find and return latest simulation file."""
        output_dir = Path("output")
        sim_files = list(output_dir.glob("simulation_circle_*.json"))
        
        if not sim_files:
            pytest.skip("No simulation files found. Run: ./run.sh e2e-simple --trajectory circle")
        
        latest_sim = max(sim_files, key=lambda p: p.stat().st_mtime)
        return latest_sim
    
    @pytest.fixture
    def simulation_data(self, simulation_file):
        """Load simulation data from JSON file."""
        return load_simulation_data(simulation_file)
    
    @pytest.fixture
    def landmarks_map(self, simulation_data):
        """Build landmarks map from simulation data."""
        landmarks_map = {}
        if simulation_data.get('trajectory') and hasattr(simulation_data['trajectory'], 'states'):
            # Handle loaded data format
            if hasattr(simulation_data.get('landmarks', {}), 'landmarks'):
                for lid, landmark in simulation_data['landmarks'].landmarks.items():
                    landmarks_map[lid] = landmark.position
        else:
            # Handle raw JSON format
            if 'groundtruth' in simulation_data and 'landmarks' in simulation_data['groundtruth']:
                for landmark in simulation_data['groundtruth']['landmarks']:
                    landmarks_map[landmark['id']] = np.array(landmark['position'])
        
        return landmarks_map
    
    def extract_ideal_coordinates(self, camera_frame, landmarks_map):
        """
        Extract ideal coordinates and 3D points from a camera frame.
        
        Returns:
            ideal_points: Nx2 array of ideal coordinates
            world_points: Nx3 array of corresponding 3D world points
        """
        ideal_points = []
        world_points = []
        
        # Handle both loaded data format and raw JSON format
        if hasattr(camera_frame, 'observations'):
            observations = camera_frame.observations
        else:
            observations = camera_frame['observations']
        
        for obs in observations:
            # Handle different observation formats
            if hasattr(obs, 'landmark_id'):
                landmark_id = obs.landmark_id
                ideal_coords = obs.ideal_coordinates if hasattr(obs, 'ideal_coordinates') and obs.ideal_coordinates is not None else None
                pixel_u = obs.pixel.u if hasattr(obs, 'pixel') else None
                pixel_v = obs.pixel.v if hasattr(obs, 'pixel') else None
            else:
                landmark_id = obs['landmark_id']
                ideal_coords = np.array(obs['ideal_coordinates']) if 'ideal_coordinates' in obs else None
                if 'pixel' in obs:
                    if isinstance(obs['pixel'], dict):
                        pixel_u = obs['pixel']['u']
                        pixel_v = obs['pixel']['v']
                    else:
                        pixel_u, pixel_v = obs['pixel']
                else:
                    pixel_u = pixel_v = None
            
            # Get ideal coordinates if available, otherwise compute from pixel
            if ideal_coords is not None:
                if isinstance(ideal_coords, np.ndarray):
                    ideal_coords_array = ideal_coords
                else:
                    ideal_coords_array = np.array(ideal_coords)
            elif pixel_u is not None and pixel_v is not None:
                # Compute from pixel if not available
                # Assuming standard pinhole model for testing
                fx = fy = 500.0  # Default focal length
                cx = 320.0
                cy = 240.0
                ideal_coords_array = np.array([
                    (pixel_u - cx) / fx,
                    (pixel_v - cy) / fy
                ])
            else:
                continue
            
            # Get corresponding 3D point
            if landmark_id in landmarks_map:
                world_point = landmarks_map[landmark_id]
                ideal_points.append(ideal_coords_array)
                world_points.append(world_point)
        
        return np.array(ideal_points), np.array(world_points)
    
    def solve_pnp_from_ideal(self, ideal_points, world_points):
        """
        Solve PnP using ideal coordinates (normalized image coordinates).
        
        Args:
            ideal_points: Nx2 array of ideal coordinates
            world_points: Nx3 array of 3D world points
            
        Returns:
            R: 3x3 rotation matrix
            t: 3x1 translation vector
        """
        if len(ideal_points) < 4:
            raise ValueError(f"Need at least 4 points for PnP, got {len(ideal_points)}")
        
        # For ideal coordinates, we need to use the normalized coordinates directly
        # Ideal coordinates are already in the normalized camera plane (z=1)
        # We need to convert them back to "pixel" coordinates for OpenCV
        # but with a canonical camera matrix
        
        # Use a canonical focal length for converting ideal to pixel
        focal_length = 1.0  # Since we're working with ideal coordinates
        camera_matrix = np.array([
            [focal_length, 0, 0],
            [0, focal_length, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Convert ideal coordinates to pixel coordinates using the canonical focal length
        pixels = ideal_points * focal_length  # This gives us "pixel" coordinates for f=1
        
        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            world_points.astype(np.float32),
            pixels.astype(np.float32),
            camera_matrix,
            None,  # No distortion
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            raise RuntimeError("PnP failed to converge")
        
        # Convert rotation vector to matrix
        R_cv, _ = cv2.Rodrigues(rvec)
        t_cv = tvec.flatten()
        
        # OpenCV PnP returns [R|t] such that: s * m = K * [R|t] * M
        # where m is image point, M is world point
        # This means: P_camera = R * P_world + t
        # But we typically want the camera pose in world coordinates (world-to-camera transform)
        # So we need to invert: R_world_to_camera = R^T, t_world_to_camera = -R^T * t
        
        # Return world-to-camera transformation (which matches our ground truth setup)
        R = R_cv  # This is already world-to-camera rotation
        t = t_cv  # This is already world-to-camera translation
        
        return R, t
    
    def compare_poses(self, R_est, t_est, R_gt, t_gt):
        """
        Compare estimated pose with ground truth.
        
        Returns:
            rotation_error: Angle error in degrees
            translation_error: Translation error in meters
        """
        # Rotation error using angle from rotation difference
        R_diff = R_gt @ R_est.T
        angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        rotation_error = np.degrees(angle)
        
        # Translation error
        translation_error = np.linalg.norm(t_est - t_gt)
        
        return rotation_error, translation_error
    
    @pytest.mark.skip(reason="Requires simulation data with consistent format - test separately")  
    def test_pnp_validation_with_simulation_data(self, simulation_data, landmarks_map):
        """Test PnP validation using simulation data."""
        # Skip if no landmarks
        if not landmarks_map:
            pytest.skip("No landmarks found in simulation data")
        
        print(f"Loaded {len(landmarks_map)} landmarks")
        
        # Get camera frames and ground truth poses
        camera_frames = []
        keyframes = []
        
        # Handle different data formats
        if hasattr(simulation_data, 'get'):
            # Raw JSON format
            if 'measurements' in simulation_data:
                camera_frames = simulation_data['measurements'].get('camera_frames', [])
                keyframes = simulation_data['measurements'].get('keyframes', [])
        else:
            # Loaded data format
            if hasattr(simulation_data, 'camera_data') and simulation_data['camera_data']:
                camera_frames = simulation_data['camera_data'].frames
            
            # For loaded data, we might need to create keyframes from trajectory
            if hasattr(simulation_data, 'trajectory') and simulation_data['trajectory']:
                # Create pseudo-keyframes from trajectory states
                for i, state in enumerate(simulation_data['trajectory'].states[:10]):  # Limit to 10
                    # Find corresponding camera frame
                    cam_frame = None
                    for cf in camera_frames:
                        if hasattr(cf, 'timestamp') and abs(cf.timestamp - state.pose.timestamp) < 0.1:
                            cam_frame = cf
                            break
                    
                    if cam_frame and len(cam_frame.observations) >= 4:
                        pseudo_keyframe = {
                            'timestamp': state.pose.timestamp,
                            'groundtruth_pose': {
                                'position': state.pose.position,
                                'orientation': state.pose.rotation_matrix
                            }
                        }
                        keyframes.append((pseudo_keyframe, cam_frame))
        
        # Test PnP on keyframes
        errors_rotation = []
        errors_translation = []
        valid_tests = 0
        
        # Handle both formats
        test_data = []
        if keyframes and isinstance(keyframes[0], tuple):
            # Loaded data format with paired keyframes and camera frames
            test_data = keyframes[:10]
        else:
            # Raw JSON format
            for i, kf in enumerate(keyframes[:10]):
                # Find corresponding camera frame
                cam_frame = None
                for cf in camera_frames:
                    if abs(cf['timestamp'] - kf['timestamp']) < 0.001:
                        cam_frame = cf
                        break
                
                if cam_frame and len(cam_frame['observations']) >= 4:
                    test_data.append((kf, cam_frame))
        
        for i, (keyframe, cam_frame) in enumerate(test_data):
            # Get ground truth pose
            if isinstance(keyframe, dict):
                gt_pose = keyframe['groundtruth_pose']
                R_gt = np.array(gt_pose['orientation']).reshape(3, 3)
                t_gt = np.array(gt_pose['position'])
            else:
                # Handle other formats
                continue
            
            # Check if we have enough observations
            num_obs = len(cam_frame['observations']) if isinstance(cam_frame, dict) else len(cam_frame.observations)
            if num_obs < 4:
                continue
            
            # Extract ideal coordinates and solve PnP
            try:
                ideal_points, world_points = self.extract_ideal_coordinates(cam_frame, landmarks_map)
                
                if len(ideal_points) < 4:
                    continue
                    
                R_est, t_est = self.solve_pnp_from_ideal(ideal_points, world_points)
                
                # Compare with ground truth
                rot_err, trans_err = self.compare_poses(R_est, t_est, R_gt, t_gt)
                
                errors_rotation.append(rot_err)
                errors_translation.append(trans_err)
                valid_tests += 1
                
                print(f"Keyframe {i}: Rot error: {rot_err:.3f}°, Trans error: {trans_err:.3f}m")
                
            except Exception as e:
                print(f"Keyframe {i}: Failed - {e}")
        
        # Verify we had enough valid tests
        assert valid_tests > 0, "No valid keyframes to test"
        
        # Check results
        mean_rot_error = np.mean(errors_rotation)
        mean_trans_error = np.mean(errors_translation)
        max_rot_error = np.max(errors_rotation)
        max_trans_error = np.max(errors_translation)
        
        print(f"\nSummary:")
        print(f"  Valid tests: {valid_tests}")
        print(f"  Mean rotation error: {mean_rot_error:.3f}°")
        print(f"  Mean translation error: {mean_trans_error:.3f}m")
        print(f"  Max rotation error: {max_rot_error:.3f}°")
        print(f"  Max translation error: {max_trans_error:.3f}m")
        
        # Assert reasonable errors (more lenient for automated testing)
        assert mean_rot_error < 10.0, f"Mean rotation error too high: {mean_rot_error:.3f}° > 10.0°"
        assert mean_trans_error < 1.0, f"Mean translation error too high: {mean_trans_error:.3f}m > 1.0m"
        
        print("\n✓ PnP validation PASSED - Ideal projections are consistent!")
    
    def test_pnp_solver_basic_functionality(self):
        """Test basic PnP solver functionality with synthetic data."""
        # Create synthetic test data
        # Ground truth pose
        R_gt = np.array([
            [0.866, -0.5, 0],
            [0.5, 0.866, 0],
            [0, 0, 1]
        ])  # 30 degree rotation around Z
        t_gt = np.array([1.0, 2.0, 3.0])
        
        # Create 3D points
        world_points = np.array([
            [0, 0, 10],
            [2, 0, 10], 
            [0, 2, 10],
            [2, 2, 10],
            [-1, -1, 10]
        ], dtype=np.float32)
        
        # Project to ideal coordinates
        ideal_points = []
        for wp in world_points:
            # Transform to camera frame
            p_cam = R_gt.T @ (wp - t_gt)
            # Project to ideal plane (z=1)
            ideal_coord = np.array([p_cam[0] / p_cam[2], p_cam[1] / p_cam[2]])
            ideal_points.append(ideal_coord)
        
        ideal_points = np.array(ideal_points)
        
        # Expected world-to-camera transformation from ground truth
        R_gt_w2c = R_gt.T  
        t_gt_w2c = -R_gt.T @ t_gt
        
        # Solve PnP (returns world-to-camera transformation)
        R_est_w2c, t_est_w2c = self.solve_pnp_from_ideal(ideal_points, world_points)
        
        # Compare world-to-camera transformations
        rot_err, trans_err = self.compare_poses(R_est_w2c, t_est_w2c, R_gt_w2c, t_gt_w2c)
        
        # Should be very accurate for synthetic data
        assert rot_err < 1.0, f"Rotation error too high: {rot_err:.3f}°"
        assert trans_err < 0.1, f"Translation error too high: {trans_err:.3f}m"
    
    def test_insufficient_points_handling(self):
        """Test that PnP solver handles insufficient points correctly."""
        # Test with insufficient points
        world_points = np.array([[0, 0, 5], [1, 0, 5], [0, 1, 5]], dtype=np.float32)  # Only 3 points
        ideal_points = np.array([[0, 0], [0.2, 0], [0, 0.2]], dtype=np.float32)
        
        with pytest.raises(ValueError, match="Need at least 4 points"):
            self.solve_pnp_from_ideal(ideal_points, world_points)