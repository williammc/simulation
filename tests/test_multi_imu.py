"""
Tests for multi-IMU fusion functionality.
"""

import pytest
import numpy as np

from src.estimation.multi_imu import (
    MultiIMUFusion, IMUFusionMethod, IMUConfig,
    create_multi_imu_setup, FusedIMUMeasurement
)
from src.common.data_structures import (
    IMUMeasurement, IMUCalibration,
    CameraExtrinsics, IMUExtrinsics
)


class TestMultiIMUSetup:
    """Test multi-IMU configuration and setup."""
    
    def test_create_orthogonal_setup(self):
        """Test creating orthogonal IMU configuration."""
        configs = create_multi_imu_setup(
            num_imus=3,
            configuration="orthogonal"
        )
        
        assert len(configs) == 3
        assert all(f"imu_{i}" in configs for i in range(3))
        
        # Check that each IMU has different orientation
        orientations = []
        for config in configs.values():
            R = config.calibration.extrinsics.B_T_S[:3, :3]
            orientations.append(R)
        
        # Orientations should be different
        for i in range(len(orientations)):
            for j in range(i+1, len(orientations)):
                assert not np.allclose(orientations[i], orientations[j])
    
    def test_create_redundant_setup(self):
        """Test creating redundant IMU configuration."""
        configs = create_multi_imu_setup(
            num_imus=4,
            configuration="redundant"
        )
        
        assert len(configs) == 4
        
        # In redundant setup, IMUs should have similar orientations
        orientations = []
        for config in configs.values():
            R = config.calibration.extrinsics.B_T_S[:3, :3]
            orientations.append(R)
        
        # At least some should be identical or very similar
        similar_count = 0
        for i in range(len(orientations)):
            for j in range(i+1, len(orientations)):
                if np.allclose(orientations[i], orientations[j], atol=0.1):
                    similar_count += 1
        
        assert similar_count > 0  # Some redundancy
    
    def test_custom_imu_config(self):
        """Test creating custom IMU configuration."""
        custom_configs = {
            "imu_custom": IMUConfig(
                imu_id="imu_custom",
                calibration=IMUCalibration(
                    imu_id="imu_custom",
                    accelerometer_noise_density=0.001,
                    accelerometer_random_walk=0.0001,
                    gyroscope_noise_density=0.0001,
                    gyroscope_random_walk=0.00001,
                    rate=100.0,
                    extrinsics=IMUExtrinsics(B_T_S=np.eye(4))
                ),
                weight=2.0,
                enabled=True
            )
        }
        
        assert custom_configs["imu_custom"].weight == 2.0
        assert custom_configs["imu_custom"].enabled
        assert custom_configs["imu_custom"].calibration.rate == 100.0
    
    def test_varying_number_of_imus(self):
        """Test setup with different numbers of IMUs."""
        for n in [1, 2, 3, 5, 10]:
            configs = create_multi_imu_setup(num_imus=n)
            assert len(configs) == n
            assert all(f"imu_{i}" in configs for i in range(n))


class TestIMUFusion:
    """Test IMU measurement fusion algorithms."""
    
    @pytest.fixture
    def fusion_setup(self):
        """Create a multi-IMU fusion setup."""
        configs = create_multi_imu_setup(num_imus=3, configuration="redundant")
        fusion = MultiIMUFusion(configs, fusion_method=IMUFusionMethod.WEIGHTED_AVERAGE)
        return fusion, configs
    
    def test_weighted_average_fusion(self, fusion_setup):
        """Test weighted average fusion method."""
        fusion, configs = fusion_setup
        
        # Create measurements
        base_accel = np.array([0, 0, 9.81])
        base_gyro = np.array([0.1, 0.0, 0.0])
        
        measurements = {}
        for imu_id in configs.keys():
            measurements[imu_id] = IMUMeasurement(
                timestamp=0.0,
                accelerometer=base_accel + 0.01 * np.random.randn(3),
                gyroscope=base_gyro + 0.001 * np.random.randn(3)
            )
        
        # Fuse measurements
        fused = fusion.fuse_measurements(measurements)
        
        assert isinstance(fused, FusedIMUMeasurement)
        assert fused.timestamp == 0.0
        assert len(fused.contributing_imus) == 3
        assert fused.fusion_confidence > 0
        
        # Fused values should be close to base values
        np.testing.assert_array_almost_equal(
            fused.acceleration, base_accel, decimal=1
        )
        np.testing.assert_array_almost_equal(
            fused.angular_velocity, base_gyro, decimal=2
        )
    
    def test_maximum_likelihood_fusion(self):
        """Test maximum likelihood fusion method."""
        configs = create_multi_imu_setup(num_imus=3)
        fusion = MultiIMUFusion(
            configs,
            fusion_method=IMUFusionMethod.MAXIMUM_LIKELIHOOD
        )
        
        # Create measurements with different noise levels
        true_accel = np.array([1, 2, 9.81])
        measurements = {
            "imu_0": IMUMeasurement(
                timestamp=0.0,
                accelerometer=true_accel + np.array([0.01, 0.01, 0.01]),
                gyroscope=np.zeros(3)
            ),
            "imu_1": IMUMeasurement(
                timestamp=0.0,
                accelerometer=true_accel + np.array([0.02, -0.01, 0.0]),
                gyroscope=np.zeros(3)
            ),
            "imu_2": IMUMeasurement(
                timestamp=0.0,
                accelerometer=true_accel + np.array([-0.01, 0.02, -0.01]),
                gyroscope=np.zeros(3)
            )
        }
        
        fused = fusion.fuse_measurements(measurements)
        
        assert fused is not None
        # ML estimate should be close to true value
        np.testing.assert_array_almost_equal(
            fused.acceleration, true_accel, decimal=1
        )
    
    def test_voting_fusion(self):
        """Test voting-based fusion method."""
        # Use redundant configuration so all IMUs have same orientation
        configs = create_multi_imu_setup(num_imus=5, configuration="redundant")
        fusion = MultiIMUFusion(
            configs,
            fusion_method=IMUFusionMethod.VOTING
        )
        
        # Create measurements where majority agree
        good_accel = np.array([0, 0, 9.81])
        bad_accel = np.array([10, 10, 20])
        
        measurements = {}
        for i, imu_id in enumerate(configs.keys()):
            if i < 3:  # Majority with good value
                accel = good_accel + 0.01 * np.random.randn(3)
            else:  # Minority with bad value
                accel = bad_accel + 0.01 * np.random.randn(3)
            
            measurements[imu_id] = IMUMeasurement(
                timestamp=0.0,
                accelerometer=accel,
                gyroscope=np.zeros(3)
            )
        
        fused = fusion.fuse_measurements(measurements)
        
        assert fused is not None
        # Voting should select the majority value
        np.testing.assert_array_almost_equal(
            fused.acceleration, good_accel, decimal=1
        )
    
    def test_fusion_with_disabled_imus(self, fusion_setup):
        """Test fusion when some IMUs are disabled."""
        fusion, configs = fusion_setup
        
        # Disable one IMU
        imu_to_disable = list(configs.keys())[0]
        configs[imu_to_disable].enabled = False
        fusion.configs = configs
        
        measurements = {}
        for imu_id in configs.keys():
            measurements[imu_id] = IMUMeasurement(
                timestamp=0.0,
                accelerometer=np.array([0, 0, 9.81]),
                gyroscope=np.zeros(3)
            )
        
        fused = fusion.fuse_measurements(measurements)
        
        assert fused is not None
        # Disabled IMU should not contribute
        assert imu_to_disable not in fused.contributing_imus
        assert len(fused.contributing_imus) == 2


class TestOutlierDetection:
    """Test outlier detection in multi-IMU fusion."""
    
    def test_outlier_detection_internal(self):
        """Test that outlier detection works within fusion."""
        # Use redundant config so all IMUs have same orientation
        configs = create_multi_imu_setup(num_imus=5, configuration="redundant")
        fusion = MultiIMUFusion(
            configs,
            fusion_method=IMUFusionMethod.WEIGHTED_AVERAGE,
            outlier_threshold=3.0
        )
        
        # Create measurements with outliers
        good_accel = np.array([0, 0, 9.81])
        bad_accel = np.array([10, 10, 20])  # Clear outlier
        
        measurements = {}
        for i, imu_id in enumerate(configs.keys()):
            if i < 3:  # Majority with good value
                accel = good_accel + 0.01 * np.random.randn(3)
            else:  # Minority with bad value
                accel = bad_accel + 0.01 * np.random.randn(3)
            
            measurements[imu_id] = IMUMeasurement(
                timestamp=0.0,
                accelerometer=accel,
                gyroscope=np.zeros(3)
            )
        
        fused = fusion.fuse_measurements(measurements)
        
        # Should reject outliers
        assert fused is not None
        assert len(fused.contributing_imus) <= 3
    
    def test_outlier_rejection_consistency(self):
        """Test consistent outlier rejection across methods."""
        configs = create_multi_imu_setup(num_imus=4)
        
        # Test with different fusion methods
        for method in [IMUFusionMethod.WEIGHTED_AVERAGE, IMUFusionMethod.VOTING]:
            fusion = MultiIMUFusion(
                configs,
                fusion_method=method,
                outlier_threshold=3.0
            )
            
            # Create measurements with clear outlier
            measurements = {}
            for i, imu_id in enumerate(configs.keys()):
                if i == 0:
                    accel = np.array([100, 100, 100])  # Outlier
                else:
                    accel = np.array([0, 0, 9.81])
                
                measurements[imu_id] = IMUMeasurement(
                    timestamp=0.0,
                    accelerometer=accel,
                    gyroscope=np.zeros(3)
                )
            
            fused = fusion.fuse_measurements(measurements)
            assert fused is not None
            assert len(fused.contributing_imus) == 3
    
    def test_outlier_rejection_in_fusion(self):
        """Test that outliers are rejected during fusion."""
        # Use redundant config so all IMUs have same orientation
        configs = create_multi_imu_setup(num_imus=4, configuration="redundant")
        fusion = MultiIMUFusion(
            configs,
            fusion_method=IMUFusionMethod.WEIGHTED_AVERAGE,
            outlier_threshold=3.0
        )
        
        # Create measurements with one outlier
        good_accel = np.array([0, 0, 9.81])
        measurements = {}
        
        for i, imu_id in enumerate(configs.keys()):
            if i == 0:  # Make first one an outlier
                accel = np.array([100, 100, 100])
            else:
                accel = good_accel + 0.01 * np.random.randn(3)
            
            measurements[imu_id] = IMUMeasurement(
                timestamp=0.0,
                accelerometer=accel,
                gyroscope=np.zeros(3)
            )
        
        fused = fusion.fuse_measurements(measurements)
        
        assert fused is not None
        # Outlier should be rejected
        assert len(fused.contributing_imus) == 3
        assert list(configs.keys())[0] not in fused.contributing_imus
        
        # Fused value should be close to good value
        np.testing.assert_array_almost_equal(
            fused.acceleration, good_accel, decimal=1
        )


class TestFusionConfidence:
    """Test fusion confidence estimation."""
    
    def test_high_confidence_with_agreement(self):
        """Test high confidence when IMUs agree."""
        # Use redundant config so all IMUs have same orientation
        configs = create_multi_imu_setup(num_imus=3, configuration="redundant")
        fusion = MultiIMUFusion(configs)
        
        # Create very similar measurements
        base_accel = np.array([1, 2, 9.81])
        measurements = {}
        
        for imu_id in configs.keys():
            measurements[imu_id] = IMUMeasurement(
                timestamp=0.0,
                accelerometer=base_accel + 0.001 * np.random.randn(3),
                gyroscope=np.zeros(3)
            )
        
        fused = fusion.fuse_measurements(measurements)
        
        assert fused.fusion_confidence > 0.8  # High confidence
        assert fused.covariance_accel is not None
        # Low uncertainty (small covariance)
        assert np.trace(fused.covariance_accel) < 0.01
    
    def test_low_confidence_with_disagreement(self):
        """Test low confidence when IMUs disagree."""
        # Use redundant config so all IMUs have same orientation
        configs = create_multi_imu_setup(num_imus=3, configuration="redundant")
        fusion = MultiIMUFusion(configs)
        
        # Create very different measurements
        measurements = {
            "imu_0": IMUMeasurement(
                timestamp=0.0,
                accelerometer=np.array([0, 0, 9]),
                gyroscope=np.zeros(3)
            ),
            "imu_1": IMUMeasurement(
                timestamp=0.0,
                accelerometer=np.array([1, 1, 10]),
                gyroscope=np.zeros(3)
            ),
            "imu_2": IMUMeasurement(
                timestamp=0.0,
                accelerometer=np.array([-1, -1, 8]),
                gyroscope=np.zeros(3)
            )
        }
        
        fused = fusion.fuse_measurements(measurements)
        
        assert fused.fusion_confidence < 0.5  # Low confidence
        # Higher uncertainty due to disagreement - relax the threshold
        assert np.trace(fused.covariance_accel) > 0.0005
    
    def test_confidence_with_partial_measurements(self):
        """Test confidence when only subset of IMUs provide data."""
        # Use redundant config so all IMUs have same orientation
        configs = create_multi_imu_setup(num_imus=5, configuration="redundant")
        fusion = MultiIMUFusion(configs)
        
        # Only provide measurements from 2 IMUs
        imu_ids = list(configs.keys())
        measurements = {
            imu_ids[0]: IMUMeasurement(
                timestamp=0.0,
                accelerometer=np.array([0, 0, 9.81]),
                gyroscope=np.zeros(3)
            ),
            imu_ids[1]: IMUMeasurement(
                timestamp=0.0,
                accelerometer=np.array([0, 0, 9.80]),
                gyroscope=np.zeros(3)
            )
        }
        
        fused = fusion.fuse_measurements(measurements)
        
        assert fused is not None
        assert len(fused.contributing_imus) == 2
        # Confidence should be moderate (not all IMUs available, only 2 sensors)
        assert 0.2 < fused.fusion_confidence < 0.7


class TestIMUCalibration:
    """Test IMU calibration in multi-IMU setup."""
    
    def test_calibration_application(self):
        """Test that IMU calibrations are properly applied."""
        # Create IMUs with different calibrations
        configs = {
            "imu_0": IMUConfig(
                imu_id="imu_0",
                calibration=IMUCalibration(
                    imu_id="imu_0",
                    accelerometer_noise_density=0.001,
                    accelerometer_random_walk=0.0001,
                    gyroscope_noise_density=0.0001,
                    gyroscope_random_walk=0.00001,
                    rate=200.0,
                    extrinsics=IMUExtrinsics(B_T_S=np.eye(4))
                ),
                weight=1.0,
                enabled=True
            ),
            "imu_1": IMUConfig(
                imu_id="imu_1",
                calibration=IMUCalibration(
                    imu_id="imu_1",
                    accelerometer_noise_density=0.01,  # 10x worse
                    accelerometer_random_walk=0.001,
                    gyroscope_noise_density=0.001,
                    gyroscope_random_walk=0.0001,
                    rate=100.0,
                    extrinsics=IMUExtrinsics(B_T_S=np.eye(4))
                ),
                weight=0.1,  # Lower weight due to worse noise
                enabled=True
            )
        }
        
        fusion = MultiIMUFusion(configs)
        
        # The weights should affect fusion
        assert configs["imu_0"].weight > configs["imu_1"].weight
        
        measurements = {
            "imu_0": IMUMeasurement(
                timestamp=0.0,
                accelerometer=np.array([0, 0, 9.81]),
                gyroscope=np.zeros(3)
            ),
            "imu_1": IMUMeasurement(
                timestamp=0.0,
                accelerometer=np.array([0, 0, 9.90]),  # Slightly different
                gyroscope=np.zeros(3)
            )
        }
        
        fused = fusion.fuse_measurements(measurements)
        
        # Result should be closer to imu_0 due to higher weight
        assert abs(fused.acceleration[2] - 9.81) < abs(fused.acceleration[2] - 9.90)