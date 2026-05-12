"""
Measurement models for mapping the system state vector to sensor measurements.
"""

import numpy as np

from src.rotations import Rotations


class MeasurementModels:
    """
    Collection of static measurement models for different sensor types.

    The system state is defined as x = [nu(6), eta(6)].T

    where:
        - nu = [u, v, w, p, q, r] represents the body fixed translational and angular velocities of the body.
        - eta = [x, y, z, phi, theta, psi] represents position and orientation in the North-East-Down (NED) coordinate frame

    State vector layout (indices):
        x[0:6]   -> nu  (velocities)
        x[6:12]  -> eta (position and orientation)
    """

    @staticmethod
    def gps(x: np.ndarray) -> np.ndarray:
        """
        GPS measurement model.

        Maps the system state to a GPS measurement consisting of position in the NED frame.

        :param x: State vector of shape (12,), containing velocities and pose.
        :type x: np.ndarray
        :return: Position [x, y, z] in NED coordinates.
        :rtype: np.ndarray
        """
        print("GPS measurement model called with state: ", x)
        print("Returning GPS measurement: ", np.concatenate([x[6:9], x[3:6]]))
        return np.concatenate([x[6:9], x[3:6]])

    @staticmethod
    def baro(x: np.ndarray) -> np.ndarray:
        """
        TODO: Barometric pressure (altitude) measurement model.

        Converts the vertical position (Down component in NED) into an
        altitude-like measurement. Since NED uses positive-down convention,
        the sign is inverted to represent height above a reference level.

        :param x: State vector of shape (12,), containing velocities and pose.
        :type x: np.ndarray
        :return: Estimated altitude (positive upward).
        :rtype: np.ndarray
        """
        return np.array([-x[8]])

    @staticmethod
    def mag(x: np.ndarray) -> np.ndarray:
        """
        Magnetometer measurement model.

        Extracts the yaw angle from the state.

        :param x: State vector of shape (12,), containing velocities and pose.
        :type x: np.ndarray
        :return: Heading (yaw angle) in radians.
        :rtype: np.ndarray
        """
        return np.array([x[11]])

    @staticmethod
    def gyroscope(x: np.ndarray) -> np.ndarray:
        """
        TODO: Review
        Gyroscope measurement model.

        Maps the state to gyroscope measurements consisting of angular velocities in the body frame.

        :param x: State vector of shape (12,), containing velocities and pose.
        :type x: np.ndarray
        :return: The predicted gyroscope measurement.
        :rtype: np.ndarray
        """

        return x[3:6]

    @staticmethod
    def accelerometer(x: np.ndarray) -> np.ndarray:
        """
        TODO:
        """

        eta = x[6:12]
        Rnb = Rotations.rpy2rot(eta[3:6])
        gn = np.array([0, 0, 9.81])

        return Rnb.R.T @ gn
