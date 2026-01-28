"""Butcher Tableaux for Runge-Kutta Solvers

See :cite:`butcherNumericalMethodsOrdinary2016` for context.
"""

import numpy as np


class ButcherTableau:
    """
    Class for storing a Butcher Tableau with error-checking on construction.
    """

    def __init__(self, a: np.ndarray, b: np.ndarray, c: np.ndarray):
        """
        Initialize the Butcher tableau :cite:`butcherNumericalMethodsOrdinary2016` with
        given coefficients.

        Args:
            a : The matrix of coefficients for the stages.
            b : The weights for the stages.
            c : The nodes (or time levels) for the stages.

        Raises:
            ValueError: If the sizes of a, b, or c are not compatible.
        """
        # Validate the input sizes
        if (
            not isinstance(a, np.ndarray)
            or not isinstance(b, np.ndarray)
            or not isinstance(c, np.ndarray)
        ):
            raise ValueError("a, b, and c must be NumPy arrays.")

        # Check the dimensions of a, b, and c
        num_stages = c.size  # Number of stages is determined by the size of c

        if a.shape != (num_stages, num_stages):
            raise ValueError(
                f"Matrix a must be of shape ({num_stages}, {num_stages}).\
                    Current shape: {a.shape}"
            )

        if b.size != num_stages:
            raise ValueError(
                f"Array b must have size {num_stages}. Current size: {b.size}"
            )

        if c.size != num_stages:
            raise ValueError(
                f"Array c must have size {num_stages}. Current size: {c.size}"
            )

        # Store the validated parameters as private attributes
        self._a = a
        self._b = b
        self._c = c

    @property
    def a(self):
        """Get a copy of the matrix of coefficients."""
        return self._a.copy()

    @property
    def b(self):
        """Get a copy of the weights for the stages."""
        return self._b.copy()

    @property
    def c(self):
        """Get a copy of the nodes for the stages."""
        return self._c.copy()

    def expand(self):
        """expand tableau as parameters a,b,c"""
        return (self.a, self.b, self.c)

    def __repr__(self):
        return str(
            "ButcherTableau(\n"
            + self._parm2str("a")
            + ",\n"
            + self._parm2str("b")
            + ",\n"
            + self._parm2str("c")
            + ")"
        )

    def _parm2str(self, property_name):
        return str(
            " %3s=" % property_name
            + str(getattr(self, "_" + property_name)).replace("\n", "\n     ")
        )


class ButcherTableauEmbedded(ButcherTableau):
    """
    Embedded Butcher Tableau extends class for adaptive step-size methods.
    """

    def __init__(self, a: np.ndarray, bf: np.ndarray, b: np.ndarray, c: np.ndarray):
        """
        Initialize the embedded Butcher tableau with given coefficients.

        Args:
            a : The matrix of coefficients for the stages.
            bf : The full-order weights for the stages.
            b : The weights for the stages with order reduced by one.
            c : The nodes (or time levels) for the stages.

        Note, the default is for b to return the reduced order solution. The
        full order solution is generally used to assess solution accuracy for
        adaptive step-size selection.

        Raises:
            ValueError: If the sizes of a, bf, b, or c are not compatible.
        """
        super().__init__(a, b, c)
        # Validate the input sizes
        if not isinstance(bf, np.ndarray):
            raise ValueError("breduced must bf a NumPy array.")

        # Check the dimensions
        num_stages = c.size  # Number of stages is determined by the size of c
        if bf.size != num_stages:
            raise ValueError(
                f"Array bf must have size {num_stages}. Current size: {bf.size}"
            )

        self._bf = bf

    @property
    def bf(self):
        """Get a copy of the full-order weights for the stages."""
        return self._bf.copy()

    def __repr__(self):
        return str(
            "ButcherTableau(\n"
            + self._parm2str("a")
            + ",\n"
            + self._parm2str("bf")
            + ",\n"
            + self._parm2str("b")
            + ",\n"
            + self._parm2str("c")
            + ")"
        )


butcher_tableaux = {
    "FwdElr": ButcherTableau(  # Forward Euler
        a=np.array([[0]]), b=np.array([1]), c=np.array([0])  # 1 stage
    ),
    "ExpMid": ButcherTableau(  # Explicit Midpoint
        a=np.array([[0, 0], [1 / 2, 0]]),  # 2 stages
        b=np.array([0, 1]),
        c=np.array([0, 1 / 2]),
    ),
    "Heun3": ButcherTableau(  # Heun's third-order method
        a=np.array([[0, 0, 0], [1 / 3, 0, 0], [0, 2 / 3, 0]]),  # 3 stages
        b=np.array([1 / 4, 0, 3 / 4]),
        c=np.array([0, 1 / 3, 2 / 3]),
    ),
    "RK3": ButcherTableau(  # RK third-order method
        a=np.array([[0, 0, 0], [1 / 2, 0, 0], [-1, 2, 0]]),  # 3 stages
        b=np.array([1 / 6, 2 / 3, 1 / 6]),
        c=np.array([0, 1 / 2, 1]),
    ),
    "RK4": ButcherTableau(  # Runge-Kutta "original" fouth-order method
        a=np.array(
            [[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 1 / 2, 0, 0], [0, 0, 1, 0]]
        ),  # 4 stages
        b=np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6]),
        c=np.array([0, 1 / 2, 1 / 2, 1]),
    ),
    "RK45": ButcherTableauEmbedded(
        a=np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [1 / 4, 0, 0, 0, 0, 0],
                [3 / 32, 9 / 32, 0, 0, 0, 0],
                [1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0, 0],
                [439 / 216, -8, 3680 / 513, -845 / 4104, 0, 0],
                [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40, 0],
            ]
        ),  # 6 stages
        bf=np.array([16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55]),
        b=np.array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0]),
        c=np.array([0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2]),
    ),
}
""" Dictionary containing common :class:`ButcherTableau`

    Examples include forward-Euler, explicit midpoint, and RK4.

"""
