import numpy as np
import unittest
from scipy.interpolate import BPoly

from hitman.rotation import LieExponential as SO
from hitman.gyro import functional_factory


class TestBPolyIntegration(unittest.TestCase):
    def test_delta_theta_consistency(self):

        t = 0.5  # time (curve sample point)
        tau = 1e-5  # step size
        tol = 1e-14
        norm = np.linalg.norm

        A = np.random.randn(4, 3)
        f = BPoly(A[:, np.newaxis, :], [0.0, 1.0])
        for is_omega in (True, False):
            eval_delta_theta, _, eval_omega = functional_factory(f, is_omega)

            delta_theta = 0.5 * (eval_omega(t) + eval_omega(t - tau)) * tau
            err = norm(delta_theta - eval_delta_theta(t, tau))

            self.assertLess(
                err, tol, f"Delta omega error for is_omega is {is_omega}, {err}"
            )

    def test_delta_R_consistency(self):
        t = 0.5  # time (curve sample point)
        tau = 1e-5  # step size
        tol = 1e-13
        norm = np.linalg.norm

        A = np.random.randn(4, 3)
        f = BPoly(A[:, np.newaxis, :], [0.0, 1.0])
        for is_omega in (True, False):
            _, eval_delta_R, eval_omega = functional_factory(f, is_omega)

            delta_R = SO.exp(0.5 * (eval_omega(t) + eval_omega(t - tau)) * tau)
            err = norm(delta_R - eval_delta_R(t, tau))

            self.assertLess(err, tol, f"Delta R error for is_omega is {is_omega}")
