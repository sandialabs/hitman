import numpy as np
from scipy import integrate
from typing import Callable


def delta_functional(
    t: float, tau: float, f: Callable[[float], np.ndarray], rng=None, **kwargs
) -> np.ndarray:
    r"""
    Compute lagged integrated measurements (:math:`\Delta f`) from functional :math:`f`
    numerically.

    .. math::
        \Delta f[k] = \int_{t-\tau (k+1)}^{t-\tau k} f(t) dt

    Args:
        t : reference time
        tau : integration interval
        f : functional providing instantaneous function value
            :math:`\mathbb{R}\rightarrow \mathbb{R}^n`
        range: integrated samples to collect about t. Default, :math:`k = [0]`

    Returns:
        :math:`\Delta f[k]` integrated measurement

    """
    if rng is None:
        rslt = integrate.quad_vec(f, t - tau, t, **kwargs)[0]
    else:
        assert isinstance(rng, range), "rng must be range"
        rslt = np.vstack(
            [
                # j indicates END of integration window
                delta_functional(t + j * tau, tau, f, **kwargs)
                for j in rng
            ]
        )
    return rslt
