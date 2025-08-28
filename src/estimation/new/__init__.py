"""
Camera-model-independent estimators.

This module contains new generation estimators that are decoupled from camera models.
They work with pre-processed measurements instead of performing projections internally.
"""

from .swba_estimator import NewSWBAEstimator, NewSWBAConfig

__all__ = ['NewSWBAEstimator', 'NewSWBAConfig']