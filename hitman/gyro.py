r"""
The gyro module collects gyro measurement and solution functions

Direct parametric relationship between instantaneous rate and attitude are elusive.
Consider the relationships

.. math::
    \begin{align}
    \dot{\phi} &= \left(J_\phi^r\right)^{-1} \omega^r\\
    \Delta R(t) &= \exp\big(-\phi(t-\tau)\big)\exp\big(\phi(t)\big) \\
    \Delta \theta &= \int_{t-\tau}^t \omega^r(s) ds
    \end{align}

If :math:`\omega` is defined as a polynomial, then :math:`\Delta \theta` is available
directly. However, :math:`\phi` must be solved numerically. Conversely, if :math:`\phi`
is defined as a polynomial then :math:`\Delta R` is available directly. However,
:math:`\Delta \theta` must then be solved numerically.

The following functions provide consistent expressions for :math:`\Delta R` and
:math:`\Delta \theta`. They can be specified from a single polynomial expression for
either :math:`\omega` or :math:`\phi`. The approach is wrapped in
:func:`hitman.gyro.functional_factory`.
"""

import copy
import numpy as np
from scipy.interpolate import BPoly
from functools import partial
from typing import Callable, Tuple

from hitman.ode import bortz_equation, solve_numeric
from hitman.rotation import LieExponential as SO
from hitman.integrate import delta_functional


def delta_theta_from_omega(
    t: float, tau: float, omega: Callable[[float], np.ndarray] = None, **kwargs
) -> np.ndarray:
    r"""Compute delta-theta from omega functional

    Integrated rate is defined as the lagged integral

    .. math::
       \Delta \theta = \int_{t-\tau}^{t} \omega(s) ds

    Will compute integral directly if omega is a :class:`scipy.interpolate.BPoly`.
    Othwerwise, will compute numerically using
    :func:`hitman.integrate.delta_functional`.

    Args:
      t : final time
      tau : integration period
      omega : functional of instantaneous rate
        :math:`\omega : \mathbb{R}\rightarrow \mathbb{R}^3`
      **kwargs : keyword arguments passed to :func:`hitman.integrate.delta_functional`.

    """
    if isinstance(omega, BPoly):
        # Compute analytically
        rslt = omega.integrate(t - tau, t)
    else:
        # Compute numerically
        rslt = delta_functional(t, tau, f=omega, **kwargs)

    return rslt


def omega_from_phi(
    t,
    phi: Callable[[float], np.ndarray] = None,
    phidot: Callable[[float], np.ndarray] = None,
    J=SO.J_r,
):
    r"""functional representation of instantaneous rate

    Instantaneous rate is obtained by inverting the Bortz equation

    .. math::
        \omega = J_\phi \dot{\phi}

    Args:
        t : time to evaluate
        phi : functional of attitude
            :math:`\phi : \mathbb{R}\rightarrow \mathbb{R}^3`
        phidot : functional of attitude rate
            :math:`\dot{\phi} : \mathbb{R}\rightarrow \mathbb{R}^3`
        J : Jacobian associated with omega (right vs. left). Default: right.

    Returns:
       instantaneous rate functional
         :math:`\omega : \mathbb{R}\rightarrow \mathbb{R}^3`

    """

    # omega is computed by inverting the Bortz equation
    omega = partial(bortz_equation, omega=phidot, Jinv=J)

    if isinstance(t, float):
        rslt = omega(t, phi(t))
    elif len(t) == 1:
        rslt = omega(t[0], phi(t[0]))
    else:
        rslt = np.vstack([omega(t, phi(t)) for t in t])

    return rslt


def delta_phi_from_omega(
    t,
    tau,
    omega: Callable[[float], np.ndarray] = None,
    Jinv: Callable[[np.ndarray], np.ndarray] = SO.J_r_inv,
    **kwargs
):
    r"""Numerically solve attitude update

    Solve the ODE

    .. math::
        \Delta \dot{\phi} = J_\phi^{-1} \omega

    for :math:`\Delta \phi(t)` from :math:`\Delta \phi(t-\tau) = 0`

    Args:
      t : final time
      tau : integration period
      omega : functional of instantaneous rate
        :math:`\omega : \mathbb{R}\rightarrow \mathbb{R}^3`
      Jinv: functional :math:`J^{-1} : \mathbb{R}^N\rightarrow \mathbb{R}^{N\times N}`

    Returns:
        :math:`\Delta R \in SO(3)`
    """
    ode = partial(bortz_equation, omega=omega, Jinv=Jinv)
    defaults = {
        "max_step": 1e-4,
    }
    defaults.update(kwargs)
    return solve_numeric(ode, t - tau, tau, **defaults)


def delta_R_from_phi(t, tau, phi: Callable[[float], np.ndarray] = None):
    r"""Direct evaluation of :math:`\Delta R` from :math:`\phi` functional

    .. math::
       \Delta R = \exp\left(-\phi(t - \tau)\right) \exp\left(\phi(t)\right)

    Args:
        t : time (end of integration)
        tau : integration interval
        phi : functional of attitude :math:`\phi : \mathbb{R}\rightarrow \mathbb{R}^3`

    Returns:
        :math:`\Delta R \in SO(3)`.

    """
    return SO.exp(-phi(t - tau)) @ SO.exp(phi(t))


def functional_factory(f: BPoly, is_omega: bool) -> Tuple[
    Callable[[float, float], np.ndarray],
    Callable[[float, float], np.ndarray],
    Callable[[float], np.ndarray],
]:
    r"""Generate functionals for :math:`\Delta R`, :math:`\Delta \theta` and
     :math:`\omega`

    Input is a polynomial expression for either :math:`\omega` or :math:`\phi`. Returns
    a consistent set of functionals for integrated values.

    Args:
       f : polynomial expression for either :math:`\omega` or :math:`\phi`
       is_omega : if true, interpret `f` as :math:`\omega`

    Returns:
       functionals :math:`\Delta \theta (t,\tau)`, :math:`\Delta R (t,\tau)`,
          :math:`\omega (t)`

    """
    assert isinstance(f, BPoly), "Expecting Bpoly"
    if is_omega:
        # omega is analytic, must numerically integrate phi
        eval_omega = copy.deepcopy(f)
        eval_delta_theta = partial(delta_theta_from_omega, omega=eval_omega)
        eval_delta_phi = partial(delta_phi_from_omega, omega=eval_omega)

        # exponentiate result
        def eval_delta_R(*args, f=eval_delta_phi, **kwargs):
            return SO.exp(f(*args, **kwargs))

    else:
        # phi is analytic, must numerically integrate omega
        eval_omega = partial(omega_from_phi, phi=f, phidot=f.derivative(1))
        eval_delta_theta = partial(delta_theta_from_omega, omega=eval_omega)
        eval_delta_R = partial(delta_R_from_phi, phi=f)

    return eval_delta_theta, eval_delta_R, eval_omega


def phi_factory(
    phi0: np.ndarray,
    t0: float,
    omega: Callable[[float], np.ndarray],
    Jinv: Callable[[np.ndarray], np.ndarray] = SO.J_r_inv,
    **kwargs
) -> Callable[[float], np.ndarray]:
    r"""Generate functional for rotation in exponential coordinates

    Provide :math:`\phi(t)` which solves the Bortz equation :cite:`Bortz:1971`
    numerically.

    Args:
        phi0 : Initial attitude
        t0 : Initial time
        omega : functional :math:`\omega^r : \mathbb{R} \rightarrow \mathbb{R}^3`
        Jinv: functional :math:`J^{-1} : \mathbb{R}^N\rightarrow \mathbb{R}^{N\times N}`
        max_step : maximum step size for numeric solver
        **kwargs : additional keyword arguments passed to
          :func:`hitman.ode.solve_numeric`

    Returns:
        functional providing attitude
        :math:`\phi : \mathbb{R} \rightarrow \mathbb{R}^3`
    """
    ode = partial(bortz_equation, omega=omega, Jinv=Jinv)

    # Define functional
    def eval_phi(t, t0=t0, phi=ode):
        return solve_numeric(phi, t=t0, step_size=t - t0, y_t=phi0, **kwargs)

    return eval_phi
