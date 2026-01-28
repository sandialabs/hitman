"""Runge-Kutta Fixed-Step Solvers"""

import numpy as np
from typing import Callable
from .butcher import ButcherTableau, butcher_tableaux
from .runge_kutta import rk_fixed
from ..rotation import LieExponential as SO


def bortz_equation(
    t: float,
    phi: np.ndarray,
    omega: Callable[[float], np.ndarray],
    Jinv: Callable[[np.ndarray], np.ndarray] = SO.J_r_inv,
) -> np.ndarray:
    r"""Differential equation for rotation groups

    Differential equation of the form

    .. math::
       \dot{\phi} = J_{\phi}^{-1} \omega

    We emphasize `Jinv` must be consistent with `omega`. For example,
    :math:`\left(J^r\right)^{-1}` and :math:`\omega^r`. See
    :cite:`chirikjian2012stochastic`.

    When functional `omega` is available, this function can be turned into an ODE using
    :class:`functools.partial`. For example

    .. code-block:: python

        bortz_ode = partial(bortz_equation, omega=eval_omega)

    Args:
        t: independent variable
        phi: dependent variable in :math:`\mathbb{R}^N`
        omega: functional :math:`\omega : \mathbb{R}\rightarrow \mathbb{R}^N`
        Jinv: functional :math:`J^{-1} : \mathbb{R}^N\rightarrow \mathbb{R}^{N\times N}`

    Returns:
        :math:`\dot{\phi}\in\mathbb{R}^N`.

    """
    return Jinv(phi) @ omega(t)


def rkmk_fixed(
    omega: Callable[[float], np.ndarray],
    t_n: float,
    step_size: float,
    y_n: np.ndarray,
    tableau: ButcherTableau = butcher_tableaux["RK4"],
    Jinv: Callable[[np.ndarray], np.ndarray] = SO.J_r_inv,
) -> np.ndarray:
    r"""
    Fixed step-size Runge-Kutta-Munthe-Kaas solver for Lie-group differential equations

    Solve differential equation of the form:

    .. math::
        \dot{y} = J_{y}^{-1} \omega

    Similar to :func:`rk_fixed` for Lie-group differential equations. Instead of
    supplying the ODE :math:`\mathbb{R}\times\mathbb{R}^N\rightarrow \mathbb{R}^N`, the
    functional `omega` represents :math:`\omega : \mathbb{R}\rightarrow \mathbb{R}^N`.

    Args:
        omega : functional :math:`\omega : \mathbb{R}\rightarrow \mathbb{R}^N`
        t_n: initial time
        step_size: step size :math:`\tau`
        y_n: initial value
        tableau: method-specific tableau
        Jinv: functional :math:`J^{-1} : \mathbb{R}^N\rightarrow \mathbb{R}^{N\times N}`

    Returns:
        Solution :math:`y(t_n + \tau)`
    """

    def ode(t, phi, omega=omega, Jinv=Jinv):
        return bortz_equation(t, phi, omega, Jinv)

    return rk_fixed(ode, t_n, step_size, y_n, tableau)


def rkmk_fixed_discrete(
    tauomega: np.ndarray,
    y_n: np.ndarray,
    tableau: ButcherTableau = butcher_tableaux["RK4"],
    c_to_index: dict = {0.0: 0, 0.5: 1, 1.0: 2},
    Jinv: Callable[[np.ndarray], np.ndarray] = SO.J_r_inv,
) -> np.ndarray:
    r"""

    Runge-Kutta-Munthe-Kaas solver for discrete measurements :math:`\omega`

    Instead of supplying analytic function :math:`\omega : \mathbb{R}\rightarrow
    \mathbb{R}^N`, supply matrix of scaled samples :math:`\omega`.

    Args:
        tauomega : discrete samples of :math:`\omega`, scaled by step size :math:`\tau`
        y_n: initial value
        tableau: method-specific tableau
        c_to_index: convert tableau :math:`c` to index for `homega`
        Jinv : inverse Jacobian (user-defined left vs right corresponding with
            `fun`)
    """

    a, b, c = tableau.expand()

    d = len(np.atleast_1d(y_n))
    nk = len(np.atleast_1d(a))
    f_all = np.zeros([nk, d])
    for k in range(nk):
        theta_k = np.sum(np.expand_dims(a[k, :], axis=(1,)) * f_all, axis=0)

        idx = c_to_index.get(c[k])
        if idx is None:
            raise KeyError(f"The value {c[k]} could not be mapped to tauomega index.")

        f_k = Jinv(theta_k + y_n) @ tauomega[idx]
        f_all[k] = f_k

    theta = np.sum(np.expand_dims(b, axis=(1,)) * f_all, axis=0)
    return y_n + theta
