from typing import Callable, Tuple
import numpy as np

# Define a type hint for a callable that takes and returns numpy.ndarray
ArrayFunction = Callable[[np.ndarray], np.ndarray]


def directional_derivative_numeric(
    f: ArrayFunction, x: np.ndarray, d: np.ndarray, tau: float
) -> float:
    r"""Numerical directional derivative using centered finite differences

    .. math::
        \nabla_d f(x) = \lim_{\tau\rightarrow 0}\frac{f(x + \tau d)-f(x)}{\tau}

    Args:
        f : functional :math:`\mathbb{R}^n \rightarrow \mathbb{R}^m`
        x : starting point :math:`x\in\mathbb{R}^n`
        d : direction :math:`d\in\mathbb{R}^n`
        tau : step size

    Returns:
        Finite difference (right-continuous)
    """
    if not isinstance(x, float):
        assert x.shape == d.shape, "direction must have same shape as initial point"
    assert isinstance(tau, float), "tau must be scalar float"

    rslt = (f(x + 0.5 * tau * d) - f(x - 0.5 * tau * d)) / tau
    return rslt


def jacobian_numeric(fctn, x, tau, m=None):
    r"""Numerical directional derivative using finite differences

    Args:
        fctn : functional :math:`\mathbb{R}^n \rightarrow \mathbb{R}^m`
        x : starting point :math:`x\in\mathbb{R}^n`
        tau : step size

    Returns:
        Jacobian :math:`J\in\mathbb{R}^{m\times n}`
    """

    n = len(x)  # Number of dimensions
    if m is None:
        m = len(fctn(x))

    jacobian = np.zeros((m, n))

    # Perturb each dimension
    for i in range(n):
        # Create a small perturbation
        perturbation = np.zeros(n)
        perturbation[i] = 0.5 * tau

        # Compute the function value at the perturbed point
        fp = fctn(x + perturbation)
        fm = fctn(x - perturbation)

        # Numerical derivative
        jacobian[:, i] = (fp - fm) / tau

    return jacobian


def verify_derivative(
    f: ArrayFunction,
    fdot: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_generator: Callable = None,
    taus: Tuple[float, ...] = np.logspace(-9, 2, num=10),
    num_samples: int = 100,
    norm: Callable = np.linalg.norm,
    fdot_gets_xdot: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Compare directional derivative numerically

    Randomly generate both starting points and directions, then compute residual between
    analytic and numeric derivatives.

    Args:
        f: functional :math:`\mathbb{R}^n \rightarrow \mathbb{R}^m`
        fdot: analytic derivative from
            :math:`\dot{f}\left(x,\dot{x}\right) :
            \mathbb{R}^n \times \mathbb{R}^n \rightarrow \mathbb{R}^m`
        x_generator: generator for initial values, or specify shape of x
        taus: range of step sizes. (Default = ``np.logspace(-9,2,num=10)``)
        num_samples: number of samples at each step size
        norm : norm function
        fdot_gets_xdot : if true, then xdot will be supplied to fdot as second argument.

    Returns:
        Arrays of errors and step sizes
    """

    if not isinstance(x_generator, int):
        """Assume generator is a functional producing random values with shape
        consistent with function."""
        xgen = x_generator
    else:
        # x_generator specifies size of function inputs
        if x_generator == 1:

            # if scalar, don't wrap in np.ndarray...
            def xgen(shape=x_generator):
                return np.random.randn(shape)[0]

        else:

            def xgen(shape=x_generator):
                return np.random.randn(shape)

    # Preallocate the errors array
    errors = np.zeros((num_samples, len(taus)))

    for sample in range(num_samples):
        # Generate random x and xdot vectors
        x = xgen()
        if isinstance(x, float):
            xdot = 1
        else:
            xdot = np.random.randn(*x.shape)
            xdot = xdot / norm(xdot)

        for tau_index, tau in enumerate(taus):
            # Compute numeric and analytic derivatives
            d_numeric = directional_derivative_numeric(f, x, xdot, tau)
            if fdot_gets_xdot:
                d_analytic = fdot(x, xdot)
            else:
                d_analytic = fdot(x)

            # Compute the error norm
            errors[sample, tau_index] = norm(d_numeric - d_analytic)

    return errors, taus


def fit_stats_loglog(errors, deltas, bounds=None, verbose=False):
    """
    Compute linear fit of log-log data and return statistics

    Designed to provide unit-test metrics

    Args:
       errors: (R, M) array
       deltas: (M) array
       bounds: (2,) array

    Returns:
        min_delta : minimum delta for which fit was applied
        error_at_min_delta: error at minimum delta
        slope : line slope (loglog scale)
        var_resid : line fit residual (loglog scale)
    """

    assert len(deltas) == errors.shape[1]

    # build a boolean mask
    if bounds is None:
        mask = np.s_[:]  # Equivalent to selecting all elements
    else:
        mask = (deltas >= np.min(bounds)) & (deltas <= np.max(bounds))

    # 1) log‐transform
    x = np.log(deltas[mask])  # shape (M,)
    y = np.log(errors[:, mask].T)  # shape (M, R)

    # 2) flatten over all realizations
    #    so we have a single (M*R)-point regression
    x_flat = np.repeat(x, y.shape[1])  # or: np.tile(x, y.shape[1])
    x_flat = np.repeat(x, y.shape[1])  # [ x0,x0,…,x0, x1,x1,…,x1, … ]
    y_flat = y.ravel(order="C")  # [ y[0,0],y[0,1],…,y[0,R-1], y[1,0],… ]

    # 3) fit a line y_flat ≃ m * x_flat + b
    (m, b) = np.polyfit(x_flat, y_flat, 1)

    # 4) the quantities you asked for:

    #   slope
    slope = m

    #   predicted log‐error at the minimum delta
    min_delta = np.min(deltas[mask])
    x0 = np.log(min_delta)
    log_error_at_min_delta = m * x0 + b
    # if you want the predicted error (not log‐error):
    error_at_min_delta = np.exp(log_error_at_min_delta)

    #   residual variance = SSE / (N_points – 2)
    Npts = x_flat.size
    dof = Npts - 2
    y_fit = m * x_flat + b
    sse = np.sum((y_flat - y_fit) ** 2)
    var_resid = sse / dof

    if verbose:
        # ====== print them out ======
        print(f"          δmin = {min_delta:.2e}")
        print(f"error (@ δmin) = {error_at_min_delta:.2e}")
        print(f"         power = {slope:.2f}")
        print(f"  residual var = {var_resid:.2e}")

    return min_delta, error_at_min_delta, slope, var_resid
