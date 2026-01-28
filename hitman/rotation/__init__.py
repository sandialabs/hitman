"""rotation module for special orthogonal groups, e.g. :math:`SO(3)`"""

import sys

from .cartesian import Cartesian
from .exponential import LieExponential

__all__ = ["Cartesian", "LieExponential"]

# Imported items in __all__ appear to originate in top-level module
for name in __all__:
    getattr(sys.modules[__name__], name).__module__ = __name__
