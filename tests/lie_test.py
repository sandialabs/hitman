import numpy as np
import unittest

from hitman.lie import LieGeneric
from hitman.rotation import LieExponential
from hitman.derivative import directional_derivative_numeric

from common import print_comparison, generate_random_vectors


class TestLie(unittest.TestCase):

    def test_dexp(self):
        """Test LieGeneric.dexp using numerical directional derivative"""

        # Generate two random 3-vectors
        theta, theta_dot = generate_random_vectors()

        # Convert theta and theta_dot to 3x3 matrices using LieExponential.hat
        Theta = LieExponential.hat(theta)
        Theta_dot = LieExponential.hat(theta_dot)

        # Compute LieGeneric.dexp(Theta, Theta_dot)
        Omega_brute_force = LieGeneric.dexp(Theta, Theta_dot, order=20)

        tau = 1e-8
        Omega_numerical = directional_derivative_numeric(
            LieGeneric.exp, Theta, Theta_dot, tau
        ) @ LieGeneric.exp(-Theta)

        print_comparison(Omega_brute_force, Omega_numerical)

        # Assert that the results are all close
        self.assertTrue(np.allclose(Omega_brute_force, Omega_numerical, atol=1e-7))
