import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BPoly


def plot_hermite_spline(spline: BPoly, **kwargs):

    fig = kwargs.pop("fig", None)
    ax = kwargs.pop("ax", None)
    x = kwargs.pop("x", None)
    title = kwargs.pop("title", None)

    # set up 3D plot
    figp = fig
    if fig is None:
        fig = plt.figure(figsize=(8, 6))
        fig.clf()
        ax = fig.add_subplot(111, projection="3d")
    elif ax is None:
        ax = fig.gca()

    # dense sampling of the parameter
    n_samples = 200
    if isinstance(spline, BPoly):
        x = spline.x
    assert x is not None, "If spline not provided, x must be set"

    x_vals = np.linspace(x[0], x[-1], n_samples)

    # evaluate spline at each sample point
    # note: __call__ only takes scalar x_in
    curve = np.array([spline(xi) for xi in x_vals])  # shape (n_samples, 3)

    # plot the Hermite spline curve
    ax.plot(
        curve[:, 0],
        curve[:, 1],
        curve[:, 2],
        color="C0",
        linewidth=2,
        label="curve",
    )

    # plot start and end
    draw_arrows = False
    if isinstance(spline, BPoly):
        Y = np.vstack([spline(t) for t in spline.x[[0, -1]]])
        dspline = spline.derivative(1)
        dYdx = np.vstack([dspline(t) for t in spline.x[[0, -1]]])
        draw_arrows = True

    if draw_arrows:
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], color="C1", s=60, label="knots")

        wt = (
            0.05
            * np.max(np.linalg.vector_norm(curve - np.mean(curve)))
            / np.max(np.linalg.vector_norm(dYdx))
        )

        # draw tangent arrows at each knot
        for i, (p, m) in enumerate(zip(Y, dYdx)):

            # Choose arrow size
            arrow_vec = wt * m

            ax.quiver(
                p[0],
                p[1],
                p[2],  # start point
                arrow_vec[0],
                arrow_vec[1],
                arrow_vec[2],  # direction
                color="C2",
                linewidth=1.5,
                arrow_length_ratio=0.2,  # shrink head relative to vector
                normalize=False,
            )

    # labels & legend
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title(title)

    if figp is None:
        plt.show()

    return fig, ax
