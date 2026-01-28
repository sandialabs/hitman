"""Solver utilities"""

import numpy as np
from typing import Callable, Tuple
from .runge_kutta import rk_fixed
from scipy.integrate import solve_ivp as scipy_solve


def verify_ivp_solve(
    ode: Callable[[float, np.ndarray], np.ndarray],
    solution: Callable[[float], np.ndarray],
    t: np.ndarray = np.random.uniform(0, 1, 10),
    step_size: np.ndarray = np.logspace(-8, -1, num=15),
    solver: Callable = rk_fixed,
    norm: Callable = np.linalg.norm,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate solver error over step size

    Contrast ivp_solve against the true solution as a function of step size

    Args:
        ode: function handle for ordinary differential equation
        solution: true solution
        t: Random vector of start points for integration.
        step_size: Step sizes to evaluate
        solver : functional accepting `ode`, `t`, `step_size`, and `y_t`
        norm : norm function

    Returns:
        Summary dataframe
    """

    # Loop through each step size step_size
    average_errors = np.zeros(len(step_size))
    for i, step_size_ in enumerate(step_size):
        errors = np.zeros(len(t))
        for j, t_ in enumerate(t):
            y_ = solution(t_)
            t_end = t_ + step_size_  # End time for the integration

            y_hat = solver(ode, t_, step_size_, y_)

            y_star = solution(t_end)  # True solution at t + step_size

            errors[j] = norm(y_star - y_hat)  # Compute the error

        # Average the errors for the current step size step_size
        average_errors[i] = np.mean(errors)

    # Create a DataFrame from the dictionary
    return average_errors, step_size


def solve_numeric(
    ode: Callable[[float, np.ndarray], np.ndarray],
    t: float = 0.0,
    step_size: float = 1,
    y_t=np.zeros(3),
    **kwargs
):
    r"""Numeric ODE IVP solver

    High-fidelity, variable step-size, solver for ODE. This is a wrapper for
    :func:`scipy.integrate.solve_ivp` with adjusted signature and default arguments.

    solves for :math:`y(t + \tau)\in \mathbb{R}^N` from provided :math:`y(t)=0` and
    :math:`\dot{y} : \mathbb{R}\rightarrow\mathbb{R}^N`

    Args:
        ode: ODE function
        t: initial value of independent variable
        step_size: step size :math:`\tau\in \mathbb{R}^+`
        y_t: initial value of dependent variable
        **kwargs: additional keyword arguments passed to
          :func:`scipy.integrate.solve_ivp`

    Returns:
       Final value :math:`y(t + \tau)`
    """

    defaults = {
        "max_step": 1e-2,
        "rtol": 1e-10,
        "atol": 1e-10,
    }
    # user values override defaults
    defaults.update(kwargs)

    y = y_t.copy()
    if np.abs(step_size) > 1e-10:
        sol = scipy_solve(ode, [t, t + step_size], y, **defaults)
        y = sol.y[:, -1]
    return y
