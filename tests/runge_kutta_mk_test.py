import unittest
import numpy as np
from functools import partial

from hitman.rotation import LieExponential as SO
from hitman.gyro import phi_factory
from hitman.ode import rkmk_fixed, butcher_tableaux, verify_ivp_solve


A = np.array([[1, 2, -1, 0.5], [0, 1, 1 / 3, 0.0], [-1, 0, -np.pi, -3]]).T


def eval_omega(t, t0=0.0, A=A):
    """Omega function

    :param t: N-vector of timesteps
    :return: Nx3 matrix of omega(t)
    """
    return np.squeeze(np.vander(np.ravel(t + t0), A.shape[0], increasing=True) @ A)


# Create dictionary of ode solvers
rkmk_solver_dict = {}
tableau = ["FwdElr", "ExpMid", "RK3", "RK4"]
for tableau_ in tableau:
    # Define solver for this tableau
    if tableau_ in butcher_tableaux:

        def ivp_solve(f, t, step_size, y_t, tableau=tableau_) -> np.ndarray:
            y = rkmk_fixed(f, t, step_size, y_t, tableau=butcher_tableaux[tableau])
            return y

    else:
        raise ValueError("Unknown tableau: %s" % tableau_)

    rkmk_solver_dict[tableau_] = ivp_solve


class TestRkMkFixed(unittest.TestCase):
    def test_solution_consistency(self):
        tau = 0.1
        t = 0.2
        tol = 1e-11
        solution_global = phi_factory(np.zeros(3), 0.0, omega=eval_omega)
        Rstar = SO.exp(solution_global(t + tau))

        def norm(x):
            return np.linalg.norm(x, ord=np.inf)

        # Local solution
        phi_t = solution_global(t)
        solution_local = phi_factory(phi_t, 0.0, partial(eval_omega, t0=t))
        phi_hat = solution_local(tau)
        Rhat = SO.exp(phi_hat)
        err = norm(Rstar - Rhat)
        print("Error updating incrementally : %e" % err)
        self.assertLess(err, tol, "Error updating from non-zero phi, incrementally")

        # Local delta-solution
        solution_local = phi_factory(np.zeros(3), 0.0, partial(eval_omega, t0=t))
        delta_phi = solution_local(tau)
        DeltaR = SO.exp(delta_phi)
        R = SO.exp(phi_t)
        Rhat = R @ DeltaR
        err = norm(Rstar - Rhat)
        print("Error updating delta-rotation: %e" % err)
        self.assertLess(err, tol, "Error computing delta-rotation")

    def test_errors_decrease_with_solver_order(self):
        """Ensure solver error decreases orders of magnitude at fixed step size"""

        # Reference ODE with known solution
        solution = phi_factory(np.zeros(3), 0.0, eval_omega)
        f = eval_omega
        solver_dict = rkmk_solver_dict

        # Fix random start points and log-scale step sizes
        t = np.random.uniform(0, 1, 10)
        step_size = np.array([1e-3])

        # Initialize a dictionary to store average errors for each tableau
        errors = np.array([])
        for name, ivp_solve in solver_dict.items():
            print(name)
            # Compute average error for each step size
            err, _ = verify_ivp_solve(
                f,
                solution,
                t,
                step_size,
                solver=ivp_solve,
            )
            errors = np.append(errors, err)

        print(errors)
        print(np.diff(np.log10(errors)))
        self.assertTrue((np.diff(np.log10(errors)) < -1.5).all())

        # RK4 error should be small
        self.assertLess(errors[-1], 1e-12)
