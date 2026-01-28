"""General Lie-theoretic tools for associating Lie groups with Lie algebras.

Note, Lie groups extend beyond rotations :cite:`chirikjian2012stochastic`.

See also:
  :mod:`rotation`
"""

import numpy as np
from abc import ABC, abstractmethod
from scipy.linalg import expm


def comm(A, B):
    """Matrix commutator, or (little) adjoint"""
    return A @ B - B @ A


def cot(x: float) -> float:
    """Cotangent, because not defined in numpy?!?"""
    return -np.tan(x + np.pi / 2)


class LieGeneric:
    r"""Generic, brute-force, methods for Lie groups

    Analytic expressions are available for specific Lie groups (e.g.
    :math:`LieSO(3)`) which are typically more efficient computationally.
    """

    @staticmethod
    def exp(A: np.matrix) -> np.matrix:
        r"""Matrix exponential"""
        return expm(A)

    @staticmethod
    def dexp(A: np.matrix, C: np.matrix, order: int = 20) -> np.matrix:
        r"""Left-Jacobian at :math:`A` applied to :math:`C`

        :math:`A,C` are in the Lie algebra.

        Left indicates derivative appears on the left (c.f. (2.42)
        :cite:t:`iserles2000lie`)

        .. math::
            \operatorname{dexp}(A,C) = \left(\frac{d}{dt} \exp(A)\right) \exp(A)^{-1}

        where

        .. math:: C = A' (t)

        For the commutator expansion, see (2.44) in :cite:t:`iserles2000lie`.

        Args:
          A : algebra elements
          C : algebra elements
          order : series order to evaluate

        Returns:
          algebra element.
        """
        result = C.copy()
        comm_result = C.copy()
        factorial_denominator = 1

        for i in range(2, order + 1):
            factorial_denominator *= i
            comm_result = comm(A, comm_result)
            result += (1 / factorial_denominator) * comm_result

        return result


class LieGroup(ABC):
    """Lie groups base for building efficient, coordinate-depenedent implementations

    See also :class:`hitman.rotation.LieExponential`
    """

    def __init__(self):
        pass

    def __call__(self):
        raise AttributeError("not implemented")

    @staticmethod
    @abstractmethod
    def hat(q: np.array) -> np.matrix:
        r"""hat operator mapping vectors to the Lie algebra element

        Args:
            q : vector coordinates

        Returns:
            algebra element
        """
        pass

    @staticmethod
    @abstractmethod
    def vee(Omega: np.matrix) -> np.array:
        r"""vee operator mapping Lie algebra element to vector coordinates

        Args:
            Omega : algebra element

        Returns:
            vector coordinates
        """
        pass

    @staticmethod
    @abstractmethod
    def Ad(R: np.array, X: np.array) -> np.array:
        """Adjoint operator is a group-parameterized linear operator of the
        algebra

        Args:
            R: group element
            X: algebra element

        Returns: algebra element
        """
        pass
