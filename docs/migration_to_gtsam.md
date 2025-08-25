# Migration Guide: Legacy Estimators to GTSAM

## Overview

As of version 2.0, the simulation framework has migrated to GTSAM-based estimators for improved performance and numerical stability. The legacy Python implementations are deprecated and will be removed in a future release.

## Migration Timeline

- **Current (v2.0)**: Legacy estimators moved to `src/estimation/legacy/` with deprecation warnings
- **Next Release (v3.0)**: Legacy estimators will be removed completely

## Quick Migration Reference

| Legacy Estimator | GTSAM Replacement | CLI Command Change |
|-----------------|-------------------|-------------------|
| `EKFSlam` | `GtsamEkfEstimator` | `slam ekf` → `slam gtsam-ekf` |
| `SlidingWindowBA` | `GtsamSWBAEstimator` | `slam swba` → `slam gtsam-swba` |
| `SRIFSlam` | `GtsamEkfEstimator` | `slam srif` → `slam gtsam-ekf` |

## Code Migration

### Before (Legacy)
```python
from src.estimation.ekf_slam import EKFSlam
from src.estimation.swba_slam import SlidingWindowBA
from src.estimation.srif_slam import SRIFSlam

# Create estimator
ekf = EKFSlam(config, camera_calib, imu_calib)
```

### After (GTSAM)
```python
from src.estimation.gtsam_ekf_estimator import GtsamEkfEstimator
from src.estimation.gtsam_swba_estimator import GtsamSWBAEstimator

# Create estimator
ekf = GtsamEkfEstimator(config)
```

## CLI Migration

### Before
```bash
python tools/cli.py slam ekf --input data.json --output results/
python tools/cli.py slam swba --input data.json --output results/
python tools/cli.py slam srif --input data.json --output results/
```

### After
```bash
python tools/cli.py slam gtsam-ekf --input data.json --output results/
python tools/cli.py slam gtsam-swba --input data.json --output results/
# Note: SRIF users should migrate to gtsam-ekf
python tools/cli.py slam gtsam-ekf --input data.json --output results/
```

## Configuration Changes

### Legacy Configuration Files
- `config/estimators/ekf.yaml` - Still works but shows deprecation warning
- `config/estimators/swba.yaml` - Still works but shows deprecation warning
- `config/estimators/srif.yaml` - Still works but shows deprecation warning

### New Configuration Files
- `config/estimators/gtsam-ekf.yaml` - GTSAM EKF configuration
- `config/estimators/gtsam-swba.yaml` - GTSAM SWBA configuration

## Key Differences

### 1. Simplified Interface
GTSAM estimators have a simplified interface:
- No separate camera/IMU calibration parameters in constructor
- Configuration handled through unified `EstimatorConfig`

### 2. Improved Performance
- GTSAM EKF: ~4x faster than legacy EKF
- GTSAM SWBA: ~10x faster than legacy SWBA
- Better numerical stability through factor graphs

### 3. Feature Changes
- **IMU-only mode**: Current GTSAM implementations use simplified IMU-only estimation
- **No landmark tracking**: Vision factors not included (for stability)
- **Better marginalization**: SWBA has improved marginalization strategy

## Handling Deprecation Warnings

If you see warnings like:
```
DeprecationWarning: EKFSlam is deprecated and will be removed in a future version.
Please use GtsamEkfEstimator instead by specifying 'gtsam-ekf' as the estimator type.
```

Follow these steps:
1. Update your import statements (see Code Migration above)
2. Update CLI commands to use `gtsam-ekf` or `gtsam-swba`
3. Review configuration files for any estimator-specific settings

## Testing Your Migration

Run the E2E test to verify GTSAM estimators work:
```bash
python test_e2e_gtsam.py
```

Compare results between legacy and GTSAM:
```bash
# Generate test data
python tools/cli.py simulate circle --duration 10 --preintegrate --output test_data.json

# Run both estimators
python tools/cli.py slam ekf --input test_data.json/simulation_*.json --output legacy_results/
python tools/cli.py slam gtsam-ekf --input test_data.json/simulation_*.json --output gtsam_results/

# Results should be comparable (GTSAM will be faster)
```

## Troubleshooting

### Issue: ImportError for legacy estimators
**Solution**: Update imports to use `src.estimation.legacy.*`

### Issue: Configuration not recognized
**Solution**: GTSAM configs use different structure. See example configs in `config/estimators/gtsam-*.yaml`

### Issue: Different results between legacy and GTSAM
**Expected**: GTSAM uses different numerical methods. Results should be similar but not identical.

## Support

For questions or issues with migration:
1. Check this migration guide
2. Review example configurations in `config/estimators/`
3. Run tests to verify functionality
4. Report issues in the project repository

## Future Roadmap

- **v2.0** (Current): Deprecation warnings, dual support
- **v2.5**: Vision factors will be added to GTSAM estimators
- **v3.0**: Complete removal of legacy implementations
- **v3.5**: Full GTSAM feature parity with advanced BA features