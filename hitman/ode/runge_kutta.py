"""Runge-Kutta Fixed-Step Solvers"""

import numpy as np
from typing import Callable
from .butcher import ButcherTableau, butcher_tableaux


def rk_fixed(
    fun: Callable[[float, np.ndarray], np.ndarray],
    t_n: float,
    step_size: float,
    y_n: np.ndarray,
    tableau: ButcherTableau = butcher_tableaux["RK4"],
) -> np.ndarray:
    r"""
    Fixed step-size Runge-Kutta solver for ordinary differential equations

    Solve differential equation of the form:

    .. math::
        \dot{y}(t) = f(t, y)


    Similar format to :func:`scipy.integrate.solve_ivp`.

    Args:
        fun: function handle for ordinary differential equation
        t_n: initial time
        step_size: step size :math:`\tau`
        y_n: initial value
        tableau: method-specific tableau

    Returns:
        Solution :math:`y(t_n + \tau)`
    """
    a, b, c = tableau.expand()

    d = len(np.atleast_1d(y_n))
    nk = len(np.atleast_1d(a))
    f_all = np.zeros([nk, d])
    for k in range(nk):
        theta_k = np.sum(np.expand_dims(a[k, :], axis=(1,)) * f_all, axis=0)
        f_k = step_size * fun(t_n + c[k] * step_size, theta_k + y_n)
        f_all[k] = f_k

    theta = np.sum(np.expand_dims(b, axis=(1,)) * f_all, axis=0)
    return y_n + theta
