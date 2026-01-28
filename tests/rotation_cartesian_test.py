import numpy as np
import unittest


from hitman.rotation import Cartesian


class TestCartesian(unittest.TestCase):

    def test_30_deg(self):
        """Test right-hand rotations about Cartesian axes"""
        atol = 1e-15
        theta = 30 / 180 * np.pi
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # rotation about x
        C = Cartesian.x(theta)
        x = np.array([0, 1, 0])
        y = C @ x
        expected_y = np.array([0, cos_theta, sin_theta])
        np.testing.assert_allclose(y, expected_y, atol=atol)

        # rotation about y
        C = Cartesian.y(theta)
        x = np.array([0, 0, 1])
        y = C @ x
        expected_y = np.array([sin_theta, 0, cos_theta])
        np.testing.assert_allclose(y, expected_y, atol=atol)

        # rotation about z
        C = Cartesian.z(theta)
        x = np.array([1, 0, 0])
        y = C @ x
        expected_y = np.array([cos_theta, sin_theta, 0])
        np.testing.assert_allclose(y, expected_y, atol=atol)

    def test_identity_at_zero_theta(self):
        """Test identity matrix at theta = 0"""
        identity_matrix = np.eye(3)
        for axis in ["x", "y", "z"]:
            theta = 0
            C = getattr(Cartesian, axis)(theta)
            np.testing.assert_allclose(C, identity_matrix, atol=1e-15)

    def test_rotation_leaves_axis_unchanged(self):
        """Test rotation about an axis leaves a unit vector in that axis unchanged"""
        for i, axis in enumerate(["x", "y", "z"]):
            theta = 30 / 180 * np.pi
            C = getattr(Cartesian, axis)(theta)
            unit_vector = np.zeros([3])
            unit_vector[i] = 1.0
            rotated_vector = C @ unit_vector
            np.testing.assert_allclose(rotated_vector, unit_vector, atol=1e-15)
