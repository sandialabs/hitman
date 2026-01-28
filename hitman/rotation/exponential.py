from typing import Tuple
import numpy as np
from numpy import sin, cos
from numpy.linalg import norm
import warnings

from hitman.lie import LieGroup


class LieExponential(LieGroup):
    """Rotation group with exponential coordinates

    Retains group element in matrix form as property M

    Note, if constructed using a matrix, no error-checking is applied to ensure
    the input is indeed a group element (othonormal).

    Several operations are retained as static methods which can be called
    without construction.
    """

    def __init__(self, theta=None, M=None, **kwargs):
        super().__init__(**kwargs)
        if theta is not None and M is None:
            self.__M = LieExponential.exp(theta)
        elif M is not None:
            self.__M = M
        else:
            raise RuntimeError("Must specify either theta or M")

    def __call__(self, p):
        return self.__M @ p

    def __matmul__(self, p):
        if isinstance(p, LieExponential):
            rslt = LieExponential(M=self.__M @ p.M)
        else:
            rslt = self.__M @ p
        return rslt

    @property
    def M(self):
        return self.__M

    # STATIC METHODS
    # ---------------------------------------------------------------------------------
    @staticmethod
    def hat(theta: np.array) -> np.matrix:
        r"""hat operator mapping :math:`\mathbb{R}^n \rightarrow so(n)`

        Args:
            theta : exponential coordinate parameterization

        Returns:
            Element of Lie algebra  (:math:`so(2)` or :math:`so(3)`)
        """
        theta_ = np.atleast_1d(theta)
        if len(theta_) == 1:
            # Scalar value, so(2)
            skewsymm = np.array([[0, -theta_[0]], [theta_[0], 0]])
        elif len(theta_) == 3:
            # R3 -> so(3)
            skewsymm = np.array(
                [
                    [0, -theta_[2], theta_[1]],
                    [theta_[2], 0, -theta_[0]],
                    [-theta_[1], theta_[0], 0],
                ]
            )
        else:
            raise ValueError(f"unexpected length: {len(theta_)}")
        return skewsymm

    @staticmethod
    def vee(M: np.matrix, tol: float = 1e-15) -> np.array:
        r"""vee operator mapping :math:`so(n) \rightarrow \mathbb{R}^n`

        Args:
            M : Element of Lie algebra  (:math:`so(2)` or :math:`so(3)`)
            tol: tolerance for checking skew-symmetry

        Returns:
            vector parameterization
        """

        # User should input skew-symmetric matrix. Sanity check off-diagonals.
        val = np.max(np.abs(np.diag(M)))
        if val > tol:
            warnings.warn(
                "Input is not skew-symmetric, off-diagonals are non-zero: %e" % val,
                UserWarning,
            )

        if M.size == 4:
            theta = 0.5 * (M[1, 0] - M[0, 1])
        elif M.size == 9:
            # Scalar value, so(2)
            theta = 0.5 * (
                np.array([M[2, 1], M[0, 2], M[1, 0]])
                - np.array([M[1, 2], M[2, 0], M[0, 1]])
            )
        else:
            raise ValueError(f"unexpected length: {M.size}")
        return theta

    @staticmethod
    def exp(theta: np.array) -> np.matrix:
        """Exponential operation for so in vector form, returns group element

        Args:
            theta: Element of Lie algebra of :math:`so(2)` or :math:`so(3)` as a vector

        Raises:
            Exception: if dimensions are incorrect

        Returns:
            Group element of :math:`SO(2)`, :math:`SO(3)`
        """
        theta_ = np.atleast_1d(theta)
        if len(theta_) == 1:
            M = np.array(
                [[cos(theta_[0]), -sin(theta_[0])], [sin(theta_[0]), cos(theta_[0])]]
            )
        elif len(theta_) == 3:
            # Compute matrix exponential via Rodriguez's formula
            #  Note, sinc should avoid divide by 0. numpy uses normalized sinc (ick!)
            thetanorm = np.linalg.norm(theta_) / np.pi
            f1 = np.sinc(thetanorm)
            f2 = 0.5 * np.sinc(0.5 * thetanorm) ** 2
            Omega = LieExponential.hat(theta_)
            M = np.eye(3) + f1 * Omega + f2 * Omega @ Omega
        else:
            raise ValueError(f"unexpected length: {len(theta_)}")
        return M

    @staticmethod
    def log(X: np.array) -> np.ndarray:
        r"""Log of group element, :math:`SO(3)\rightarrow \mathbb{R}^3`
            or :math:`SO(2)\rightarrow \mathbb{R}`

        Args:
           X: group element

        Returns:
            vector
        """
        if isinstance(X, np.ndarray):
            M = X
        else:
            M = X.M
        if M.shape == (2, 2):
            v = np.arctan(M[1, 0], M[0, 0])
        elif M.shape == (3, 3):
            # Compute matrix log
            tmp = 0.5 * (np.trace(M) - 1)
            # Clamp at +/-1 despite numeric stability errors
            tmp = tmp if np.abs(tmp) < 1 else np.sign(tmp)
            theta = np.arccos(tmp)
            if np.abs(theta) < 1e-10:
                sf = 0.5
            else:
                sf = theta / (2 * np.sin(theta))
            v = sf * LieExponential.vee(M - M.T)
        else:
            raise ValueError(f"unexpected length: {M.shape}")
        return v

    # LEFT Jacobian Methods
    # ------------------------------------------------------------------------------

    @staticmethod
    def J_l(x: np.ndarray, right=False) -> np.ndarray:
        r"""left-Jacobian for exponential coordinates

        (See :cite:`chirikjian2012stochastic` eq. 10.86)

        Args:
            x: vector representation of Lie algebra
            right: optionally compute right-Jacobian

        Returns:
            General linear group, :math:`\mathbb{R}^{3\times 3}`

        See Also:
          - Inverse: :meth:`J_l_inv`
          - Right-Jacobian and inverse:  :meth:`J_r`, :meth:`J_r_inv`
        """
        theta = norm(x)
        x_hat = LieExponential.hat(x)
        c1, c2 = LieExponential.__J_scalars(theta, right)

        return np.identity(3) + c1 * x_hat + c2 * (x_hat @ x_hat)

    @staticmethod
    def J_l_dot(x: np.ndarray, xdot: np.ndarray, right: bool = False) -> np.matrix:
        r"""time-derivative of Jacobian for exponential coordinates

        Args:
            x: vector representation of Lie algebra
            xdot: time-derivative of Lie algebra exponential coordinates
            right: optionally compute right-Jacobian

        Returns:
            General linear group :math:`\mathbb{R}^{3\times 3}`
        """
        x_hat = LieExponential.hat(x)
        xdot_hat = LieExponential.hat(xdot)
        theta = norm(x)
        c1, c2 = LieExponential.__J_scalars(theta, right)
        c1dot, c2dot = LieExponential.__J_scalars_dot(theta, x, xdot, right)
        return (
            c1dot * x_hat
            + c1 * xdot_hat
            + c2dot * (x_hat @ x_hat)
            + c2 * (xdot_hat @ x_hat + x_hat @ xdot_hat)
        )

    @staticmethod
    def J_l_inv(x: np.array, right=False) -> np.matrix:
        """inverse left-Jacobian for exponential coordinates

        See :cite:`chirikjian2012stochastic` eq. 10.86

        Args:
            x: vector representation of Lie algebra
            right: optionally compute right-Jacobian

        Returns:
            inverse-Jacobian of exp at x

        See Also:
          - Left-Jacobian: :meth:`J_l`
          - Right-Jacobian and inverse:  :meth:`J_r`, :meth:`J_r_inv`
        """
        theta = norm(x)
        x_hat = LieExponential.hat(x)
        c1, c2 = LieExponential.__J_inv_scalars(theta, right)
        return np.identity(3) + c1 * x_hat + c2 * (x_hat @ x_hat)

    # RIGHT Jacobian Methods (simply call left-Jacobian methods with right=True)
    # ------------------------------------------------------------------------------
    @staticmethod
    def J_r(theta: np.array) -> np.matrix:
        """Right-Jacobian for exponential coordinates

        See :cite:`chirikjian2012stochastic` eq. 10.86

        Args:
            x: vector representation of Lie algebra

        Returns:
            Jacobian of exp at x

        See Also:
          - Inverse: :meth:`J_r_inv`
          - Left-Jacobian and inverse:  :meth:`J_l`, :meth:`J_l_inv`
        """
        return LieExponential.J_l(theta, right=True)

    @staticmethod
    def J_r_dot(x: np.ndarray, xdot: np.ndarray) -> np.matrix:
        r"""time-derivative of right-Jacobian for exponential coordinates

        Args:
            x: vector representation of Lie algebra
            xdot: time-derivative of Lie algebra exponential coordinates
            right: optionally compute right-Jacobian

        Returns:
            General linear group :math:`\mathbb{R}^{3\times 3}`
        """
        return LieExponential.J_l_dot(x, xdot, True)

    @staticmethod
    def J_r_inv(theta: np.array) -> np.matrix:
        """Inverse right-Jacobian at exponential coordinates

        See :cite:`chirikjian2012stochastic` eq. 10.86

        Args:
            theta: exponential coordinates (vector)

        Returns:
            Inverse right-Jacobian

        See Also:
          - Right-Jacobian: :meth:`J_r`
          - Left-Jacobian and inverse:  :meth:`J_l`, :meth:`J_l_inv`

        """
        return LieExponential.J_l_inv(theta, right=True)

    @staticmethod
    def Ad(R: np.matrix, X: np.matrix) -> np.matrix:
        """Adjoint operator is a group-parameterized linear operator of the
        algebra

        Args:
            R: group element :math:`SO(3)`
            X: algebra element :math:`so(3)`

        Returns: algebra element :math:`so(3)`
        """
        if X.shape == (3, 3):
            return R @ X @ R.T
        elif X.shape == (3,):
            return R @ X

    @staticmethod
    def __J_scalars(theta: float, right=False) -> Tuple[float, float]:
        r"""
        Compute scalars required for :math:`J_\theta`

        Checks for small theta values to avoid divide-by-zero. Returns limiting values
        instead.

        (See :cite:`chirikjian2012stochastic` eq. 10.86)

        Args:
           theta: magnitude of exponential coordinate vector
           right: if True, computing right-Jacobian (default False)
        """
        if np.abs(theta) < 1e-10:
            c1 = 0.5
            c2 = 1 / 6
        else:
            c1 = (1 - np.cos(theta)) / (theta**2)
            c2 = (theta - np.sin(theta)) / (theta**3)

        if right:
            c1 *= -1

        return c1, c2

    @staticmethod
    def __J_inv_scalars(theta: float, right=False) -> Tuple[float, float]:
        r"""
        Compute scalars required for :math:`J^{-1}_\theta`

        Checks for small theta values to avoid divide-by-zero. Returns limiting values
        instead.

        (See :cite:`chirikjian2012stochastic` eq. 10.86)

        Args:
           theta: magnitude of exponential coordinate vector
           right: if True, computing right-Jacobian (default False)
        """
        if np.abs(theta) < 1e-10:
            c2 = 1 / 12.0
        else:
            c2 = 1 / (theta * theta) - (1 + np.cos(theta)) / (2 * theta * np.sin(theta))

        if right:
            c1 = 0.5
        else:
            c1 = -0.5

        return c1, c2

    @staticmethod
    def __J_scalars_dot(
        theta: float, x: np.ndarray, xdot: np.ndarray, right=False
    ) -> Tuple[float, float]:
        r"""
        Compute derivative of scalars required for :math:`J_\theta`

        Checks for small theta values to avoid divide-by-zero. Returns limiting values
        instead.

        (See :cite:`chirikjian2012stochastic` eq. 10.86)

        Args:
           theta: norm of exponential coordinate vector
           x: exponential coordinate vector
           xdot: time-derivative of exponential coordinate vector
           right: if True, computing right-Jacobian (default False)
        """

        if np.abs(theta) < 1e-10:
            c1 = 0.0
            c2 = 0.0
        else:
            c1 = (
                (x @ xdot)
                / (theta**3)
                * (np.sin(theta) + 2 * (np.cos(theta) - 1) / theta)
            )
            c2 = (
                (x @ xdot)
                / (theta**4)
                * (3 * np.sin(theta) / theta - np.cos(theta) - 2)
            )

        if right:
            c1 *= -1

        return c1, c2
