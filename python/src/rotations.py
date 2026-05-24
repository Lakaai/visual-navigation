"""
TODO:

Note the notation of Rab denotes the rotation matrix that takes the basis {a} into the basis {b}.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RotationMatrix:
    """
    TODO:

    """

    R: np.ndarray
    dRdx: Optional[np.ndarray] = None


class Rotations:
    """
    TODO:

    If Rab is the matrix that rotates the vectors of the basis {a} into the vectors of the basis {b}, 
    then Rab is the coordinate tranformation matrix that relates the coordinates of a vector in {b} to its components in {a}:
    ua = Rab @ ub
    """

    @staticmethod
    def rotx(x: float, jacobian: bool = False):
        """
        TODO:
        """
        c = np.cos(x)
        s = np.sin(x)
        R = np.eye(3)
        R[1, 1] = c
        R[1, 2] = -s
        R[2, 1] = s
        R[2, 2] = c

        dRdx = None
        if jacobian:
            dRdx = np.zeros((3, 3))

            dRdx[1, 1] = -np.sin(x)
            dRdx[2, 1] = np.cos(x)
            dRdx[1, 2] = -np.cos(x)
            dRdx[2, 2] = -np.sin(x)

        return RotationMatrix(R, dRdx)

    @staticmethod
    def roty(x, jacobian: bool = False):
        """
        TODO:
        """
        c = np.cos(x)
        s = np.sin(x)
        R = np.eye(3)
        R[0, 0] = c
        R[0, 2] = s
        R[2, 0] = -s
        R[2, 2] = c

        dRdx = None
        if jacobian:
            dRdx = np.zeros((3, 3))

            dRdx[0, 0] = -np.sin(x)
            dRdx[2, 0] = -np.cos(x)
            dRdx[0, 2] = np.cos(x)
            dRdx[2, 2] = -np.sin(x)

        return RotationMatrix(R, dRdx)

    @staticmethod
    def rotz(x: float, jacobian: bool = False):
        """
        TODO:
        """
        c = np.cos(x)
        s = np.sin(x)
        R = np.eye(3)
        R[0, 0] = c
        R[0, 1] = -s
        R[1, 0] = s
        R[1, 1] = c

        dRdx = None
        if jacobian:
            dRdx = np.zeros((3, 3))

            dRdx[0, 0] = -np.sin(x)
            dRdx[1, 0] = np.cos(x)

            dRdx[0, 1] = -np.cos(x)
            dRdx[1, 1] = -np.sin(x)

        return RotationMatrix(R, dRdx)

    @staticmethod
    def rpy2rot(rpy: np.ndarray, order: str = "zyx"):
        """
        TODO:
        """
        if order == 'zxy':

            return RotationMatrix(Rotations.rotz(rpy[2]).R @ Rotations.rotx(rpy[0]).R @ Rotations.roty(rpy[1]).R, None)
        
        return RotationMatrix(Rotations.rotz(rpy[2]).R @ Rotations.roty(rpy[1]).R @ Rotations.rotx(rpy[0]).R, None)
        # if order == "zyx":
        #     return RotationMatrix(Rotations.rotz(rpy[2]).R @ Rotations.roty(rpy[1]).R @ Rotations.rotx(rpy[0]).R, None)
        # else:
        #     raise ValueError("Unsupported rotation order")

    @staticmethod
    def rot2rpy(R: RotationMatrix):
        """
        TODO:
        """

        Rm = R.R

        theta = np.arctan2(-Rm[2, 0], np.hypot(Rm[0, 0], Rm[1, 0]))

        if np.hypot(Rm[0, 0], Rm[1, 0]) > 1e-10:
            phi = np.arctan2(Rm[2, 1], Rm[2, 2])
            psi = np.arctan2(Rm[1, 0], Rm[0, 0])
        else:
            # Gimbal lock
            phi = 0.0
            psi = np.arctan2(-Rm[0, 1], Rm[1, 1])

        return np.array([phi, theta, psi])
