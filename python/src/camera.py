from dataclasses import dataclass
import numpy as np

@dataclass 
class Camera:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray