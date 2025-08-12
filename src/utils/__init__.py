"""
Utility modules for SLAM simulation.
"""

# Math utilities are always available
from .math_utils import *

# Optional imports for dataset conversion
__all__ = []

try:
    from .tumvie_converter import convert_tumvie_dataset
    __all__.append('convert_tumvie_dataset')
except ImportError:
    # OpenCV not available, conversion utilities won't work
    pass