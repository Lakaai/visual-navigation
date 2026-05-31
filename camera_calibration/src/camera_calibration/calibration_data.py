"""
TODO:
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class CalibrationData:
    """
    A data class to store the results of camera calibration.
    """
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    reprojection_error: float
    image_resolution: tuple = None