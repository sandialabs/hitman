import unittest
import numpy as np
from scipy.integrate import solve_ivp as scipy_solve
from hitman.ode import rk_fixed, butcher_tableaux, verify_ivp_solve


class OdeExponential:
    """Exponential ODE"""

    def __init__(self, C: float = 0.1):
        self.C = C

    def sol(self, t):
        """Solution to the ODE"""
        return self.C * np.exp(t)

    def ode(self, t, y):
        """ODE"""
        return y


# Create dictionary of ode solvers
rk_solver_dict = {}
tableau = ["FwdElr", "ExpMid", "RK3", "RK4", "Variable"]
for tableau_ in tableau:
    # Define solver for this tableau
    def ivp_solve(ode, t, step_size, y_t, tableau=tableau_) -> np.ndarray:
        if tableau in butcher_tableaux:

            y = rk_fixed(ode, t, step_size, y_t, tableau=butcher_tableaux[tableau])
        else:
            # Unknown tableau, use scipy built-in solver (variable step-size)
            t_end = t + step_size
            rslt = scipy_solve(
                ode, np.array([t, t_end]), np.array([y_t]), t_eval=np.array([t_end])
            )
            y = rslt.y

        return y

    rk_solver_dict[tableau_] = ivp_solve


class TestRkFixed(unittest.TestCase):
    def test_errors_decrease_with_solver_order(self):
        """Ensure solver error decreases orders of magnitude at fixed step size"""

        # Reference ODE with known solution
        ref_ode = OdeExponential()
        solution = ref_ode.sol
        f = ref_ode.ode
        solver_dict = rk_solver_dict

        # Fix random start points and log-scale step sizes
        t = np.random.uniform(0, 1, 10)
        step_size = np.array([1e-2])

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

        self.assertTrue((np.diff(np.log10(errors)) < -2).all())

        # RK4 error should be small
        self.assertLess(errors[-2], 1e-12)
