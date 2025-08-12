"""
Utility modules for SLAM simulation.
"""

# Math utilities are always available
from .math_utils import *

# Optional imports for dataset conversion
__all__ = []

try:
    from .tumvi_converter import convert_tumvi_dataset
    __all__.append('convert_tumvi_dataset')
except ImportError:
    # OpenCV not available, conversion utilities won't work
    pass