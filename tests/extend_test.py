import unittest
import numpy as np

from hitman.extend import apply_along_axis_multi


class TestApplyAlongAxisMulti(unittest.TestCase):

    def test_basic_functionality(self):
        # Define a simple function to test
        def add(x, y):
            return x + y

        A = np.array([[1, 2, 3], [4, 5, 6]])
        B = np.array([[10, 20, 30], [40, 50, 60]])

        # Apply the function along axis 0
        result = apply_along_axis_multi(add, 0, A, B)
        expected_result = np.array([[11, 22, 33], [44, 55, 66]])

        np.testing.assert_array_equal(result, expected_result)

    def test_different_axis(self):
        def multiply(x, y):
            return x * y

        A = np.array([[1, 2, 3], [4, 5, 6]])
        B = np.array([[10, 20, 30], [40, 50, 60]])

        # Apply the function along axis 1
        result = apply_along_axis_multi(multiply, 1, A, B)
        expected_result = np.array([[10, 40, 90], [160, 250, 360]])

        np.testing.assert_array_equal(result, expected_result)

    def test_shape_mismatch(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6, 7], [8, 9, 10]])  # Different shape

        with self.assertRaises(ValueError):
            apply_along_axis_multi(np.add, 0, A, B)

    def test_invalid_axis(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])

        with self.assertRaises(IndexError):
            apply_along_axis_multi(np.sum, 2, A, B)  # Axis out of bounds


if __name__ == "__main__":
    unittest.main()
