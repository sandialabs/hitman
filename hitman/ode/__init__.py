"""ode module for numerically solving ordinary differential equations"""

import sys

from .butcher import butcher_tableaux, ButcherTableau
from .runge_kutta import rk_fixed

from .rkmk import (
    rkmk_fixed,
    rkmk_fixed_discrete,
    bortz_equation,
)
from .solve import verify_ivp_solve, solve_numeric

__all__ = [
    "butcher_tableaux",
    "ButcherTableau",
    "rk_fixed",
    "rkmk_fixed",
    "bortz_equation",
    "rkmk_fixed_discrete",
    "verify_ivp_solve",
    "solve_numeric",
]

# # Imported items in __all__ appear to originate in top-level module
for name in __all__:
    obj = getattr(sys.modules[__name__], name)
    if hasattr(obj, "__module__"):
        # variables do not have __module__ (e.g. butcher_tableaux)
        obj.__module__ = __name__
