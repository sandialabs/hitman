import numpy as np
import unittest

from hitman.rotation import LieExponential

from hitman.rotation import Cartesian as C
from hitman.lie import LieGeneric as Numeric
from hitman.derivative import verify_derivative

from common import generate_random_vectors


class TestLieExponential(unittest.TestCase):

    def test_construction(self):
        """Test LieExponential construction"""

        # Construct from exponential coordinates
        theta = np.array([np.pi / 6, 0, 0])
        G = LieExponential(theta)
        np.allclose(G.M, LieExponential.exp(theta), atol=1e-15)

        # Construct from SO(3)
        G2 = LieExponential(M=G.M)
        np.allclose(G.M, G2.M, atol=1e-15)

    def test_exp_cartesian(self):
        """Test rotations about Cartesian axes against LieExponential.exp"""

        # Number of tests to run
        num_tests = 10

        # Axes to test
        axes = ["x", "y", "z"]
        exp_axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        rotation_methods = [
            C.x,
            C.y,
            C.z,
        ]

        # Run test for multiple random realizations of theta
        for axis, exp_axis, rotation_method in zip(axes, exp_axes, rotation_methods):
            for _ in range(num_tests):
                # Generate random scalar theta over [-pi, pi]
                theta = np.random.uniform(-np.pi, np.pi)

                # Compute LieSO.exp and Cartesian
                lie_so_exp = LieExponential.exp(theta * exp_axis)
                cartesian_rotation = rotation_method(theta)

                # Check if LieSO.exp is equal to Cartesian to machine precision
                equal = np.allclose(lie_so_exp, cartesian_rotation, atol=1e-15)
                self.assertTrue(equal)

    def test_J_l(self):
        """Verify left Jacobian"""

        # Number of tests to run
        num_tests = 10

        # Run test for multiple random realizations of theta, wt, and theta_dot
        for _ in range(num_tests):
            theta, theta_dot = generate_random_vectors()

            # Compute omega using the Jacobian
            J_l = LieExponential.J_l(theta)
            omega_analytic = J_l @ theta_dot

            # Compute omega numerically using Numeric.dexp
            Theta = LieExponential.hat(theta)
            Theta_dot = LieExponential.hat(theta_dot)
            omega_numeric = LieExponential.vee(Numeric.dexp(Theta, Theta_dot, order=20))

            # Assert that the results are all close
            self.assertTrue(
                np.allclose(omega_numeric, omega_analytic, atol=1e-5, rtol=1e-5)
            )

    def test_J_l_inverse(self):
        """Verify left Jacobian is consistent with its inverse"""

        # Number of tests to run
        num_tests = 10

        # Run test for multiple random realizations of theta, wt, and theta_dot
        for _ in range(num_tests):
            # Generate two random 3-vectors
            theta, _ = generate_random_vectors()

            # Compute omega using the Jacobian
            J_l = LieExponential.J_l(theta)
            J_l_inv = LieExponential.J_l_inv(theta)

            for eye_analytic in (J_l @ J_l_inv, J_l_inv @ J_l):

                # Assert that the results are all close
                self.assertTrue(
                    np.allclose(eye_analytic, np.eye(3), atol=1e-5, rtol=1e-5)
                )

    def test_J_r_consistency(self):
        """Verify right Jacobian consistency with left and its inverse"""

        # Number of tests to run
        num_tests = 10

        # Run test for multiple random realizations of theta, wt, and theta_dot
        for _ in range(num_tests):
            theta, _ = generate_random_vectors()

            # Compute omega using the Jacobian
            J_l = LieExponential.J_l(theta)

            J_r = LieExponential.J_r(theta)
            R = LieExponential.exp(theta)

            # Assert that the results are all close
            self.assertTrue(np.allclose(R @ J_r, J_l, atol=1e-5, rtol=1e-5))

            J_r_inv = LieExponential.J_r_inv(theta)

            for eye_analytic in (J_r @ J_r_inv, J_r_inv @ J_r):

                # Assert that the results are all close
                self.assertTrue(
                    np.allclose(eye_analytic, np.eye(3), atol=1e-5, rtol=1e-5)
                )

    def test_J_dot(self):
        """Test derivative of Jacobians"""

        # Right Jacobian
        f = LieExponential.J_r
        fdot = LieExponential.J_r_dot
        errors, _ = verify_derivative(
            f, fdot, 3, taus=np.array([1e-6]), fdot_gets_xdot=True
        )
        self.assertTrue(
            np.allclose(errors, 0, atol=1e-6),
            "Right Jacobian derivative does not match numeric.",
        )

        # Right Jacobian
        f = LieExponential.J_l
        fdot = LieExponential.J_l_dot
        errors, _ = verify_derivative(
            f, fdot, 3, taus=np.array([1e-6]), fdot_gets_xdot=True
        )
        self.assertTrue(
            np.allclose(errors, 0, atol=1e-6),
            "Left Jacobian derivative does not match numeric.",
        )
