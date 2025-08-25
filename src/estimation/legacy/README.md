# Legacy Estimator Implementations

## ⚠️ DEPRECATED

This directory contains legacy SLAM estimator implementations that are **deprecated** and will be removed in a future version.

## Files in this directory:

- `ekf_slam.py` - Legacy Extended Kalman Filter implementation
- `swba_slam.py` - Legacy Sliding Window Bundle Adjustment implementation  
- `srif_slam.py` - Legacy Square Root Information Filter implementation

## Migration

Please migrate to the GTSAM-based implementations:

| Legacy File | Replacement | Import Path |
|------------|-------------|------------|
| `ekf_slam.py` | `gtsam_ekf_estimator.py` | `src.estimation.gtsam_ekf_estimator` |
| `swba_slam.py` | `gtsam_swba_estimator.py` | `src.estimation.gtsam_swba_estimator` |
| `srif_slam.py` | Use GTSAM EKF | `src.estimation.gtsam_ekf_estimator` |

## Deprecation Timeline

- **Current Release**: Deprecation warnings added, files moved to `legacy/` subdirectory
- **Next Major Release (v3.0)**: Complete removal of this directory and all legacy implementations

## Why Deprecate?

1. **Performance**: GTSAM implementations are 4-10x faster
2. **Numerical Stability**: Factor graph approach provides better numerical properties
3. **Maintenance**: Single codebase to maintain (GTSAM) instead of multiple custom implementations
4. **Features**: GTSAM provides more advanced features and better extensibility

## For Developers

If you need to maintain these files temporarily:
1. All imports have been updated to use `src.estimation.legacy.*`
2. Deprecation warnings are shown when classes are instantiated
3. Tests still pass but show deprecation warnings
4. CLI shows warnings when legacy estimators are used

## Testing Legacy Code

```bash
# Legacy tests still work but show warnings
python -m pytest tests/test_ekf_slam.py -W ignore::DeprecationWarning
python -m pytest tests/test_swba_slam.py -W ignore::DeprecationWarning
python -m pytest tests/test_srif_slam.py -W ignore::DeprecationWarning
```

## Removal Checklist (for v3.0)

When removing this directory, also update:
- [ ] Remove `src/estimation/legacy/` directory completely
- [ ] Update `tools/slam.py` to remove legacy imports and options
- [ ] Update `src/evaluation/comparison.py` to remove legacy imports
- [ ] Remove or update tests: `test_ekf_slam.py`, `test_swba_slam.py`, `test_srif_slam.py`
- [ ] Update CLI help text to remove legacy options
- [ ] Remove legacy config files: `config/estimators/ekf.yaml`, `swba.yaml`, `srif.yaml`
- [ ] Update documentation to remove references to legacy estimators