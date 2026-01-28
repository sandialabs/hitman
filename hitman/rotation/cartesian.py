import numpy as np


class Cartesian:
    """Rotations about Cartesian axes

    Active, right-hand rotations :cite:`Sommer2018Why`.
    """

    @staticmethod
    def x(theta: float) -> np.matrix:
        r"""Return the rotation matrix about the x-axis

        .. math::
            R_x(\theta) = \begin{pmatrix}
                1 & 0 & 0 \\
                0 & \cos(\theta) & -\sin(\theta) \\
                0 & \sin(\theta) & \cos(\theta)
                \end{pmatrix}

        Args:
           theta: radians

        Returns:
            Rotation matrix, :math:`SO(3)`
        """
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        return np.array(
            [[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]]
        )

    @staticmethod
    def y(theta: float) -> np.matrix:
        r"""Return the rotation matrix about the y-axis

        .. math::
            R_y(\theta) = \begin{pmatrix}
                \cos(\theta) & 0 & \sin(\theta) \\
                0 & 1 & 0 \\
                -\sin(\theta) & 0 & \cos(\theta)
                \end{pmatrix}

        Args:
           theta: radians

        Returns:
            Rotation matrix, :math:`SO(3)`
        """
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        return np.array(
            [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]
        )

    @staticmethod
    def z(theta: float) -> np.matrix:
        r"""Return the rotation matrix about the z-axis

        .. math::
            R_z(\theta) = \begin{pmatrix}
                \cos(\theta) & -\sin(\theta) & 0 \\
                \sin(\theta) & \cos(\theta) & 0 \\
                0 & 0 & 1
                \end{pmatrix}

        Args:
           theta: radians

        Returns:
            Rotation matrix, :math:`SO(3)`
        """
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        return np.array(
            [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
        )
