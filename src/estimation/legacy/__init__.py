"""
Legacy SLAM estimator implementations.

DEPRECATED: These implementations are maintained for backward compatibility only.
Please use the GTSAM-based implementations instead:
- Use GtsamEkfEstimator instead of EKFSlam
- Use GtsamSWBAEstimator instead of SlidingWindowBA
- SRIF is deprecated without replacement

The legacy implementations will be removed in a future version.
"""

import warnings
from typing import Optional

def _deprecation_warning(old_class: str, new_class: str) -> None:
    """Issue a deprecation warning for legacy estimators."""
    warnings.warn(
        f"{old_class} is deprecated and will be removed in a future version. "
        f"Please use {new_class} instead. "
        f"Update your code to use 'gtsam-ekf' or 'gtsam-swba' as the estimator type.",
        DeprecationWarning,
        stacklevel=3
    )

# Re-export with deprecation warnings
def get_ekf_slam():
    """Get legacy EKF SLAM with deprecation warning."""
    _deprecation_warning("EKFSlam", "GtsamEkfEstimator")
    from src.estimation.ekf_slam import EKFSlam
    return EKFSlam

def get_swba_slam():
    """Get legacy SWBA with deprecation warning."""
    _deprecation_warning("SlidingWindowBA", "GtsamSWBAEstimator")
    from src.estimation.swba_slam import SlidingWindowBA
    return SlidingWindowBA

def get_srif_slam():
    """Get legacy SRIF with deprecation warning."""
    _deprecation_warning("SRIFSlam", "No GTSAM replacement (use GtsamEkfEstimator)")
    from src.estimation.srif_slam import SRIFSlam
    return SRIFSlam