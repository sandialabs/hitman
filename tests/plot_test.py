import unittest

import numpy as np
import matplotlib.pyplot as plt

from hitman.plot import plot_hermite_spline
from scipy.interpolate import BPoly


class TestPlot(unittest.TestCase):

    def test_plot_generation(self):

        random_matrix = np.random.randn(5, 3) * 0.1

        poly_phi = BPoly(random_matrix[:, np.newaxis, :], [0.0, 1.0])

        fig = plt.figure()
        fig.clf()
        ax = fig.add_subplot(121, projection="3d")
        _, _ = plot_hermite_spline(poly_phi, fig=fig, ax=ax)

        # plt.close()
