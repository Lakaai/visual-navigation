from dataclasses import dataclass
import numpy as np
from src.rotations import Rotations, RotationMatrix
import cv2 

@dataclass
class Camera():
    matrix: np.ndarray
    distortion_coeffs: np.ndarray
    translation_vector: np.ndarray
    rotation_matrix: RotationMatrix


    def undistort_features(self, features):
        """
        TODO: 
        """
        undistorted_features = cv2.undistortPoints(features, self.matrix, self.distortion_coeffs, P=self.matrix)
        return undistorted_features

CAMERA = Camera(
matrix=np.array([
    [2230.0302561729463, 0.0, 1326.9024252144795],
    [0.0, 2249.3015295213522, 483.88958360970497],
    [0.0, 0.0, 1.0]
    ]), 
    distortion_coeffs=np.array([
        0.010524588721904289, 0.048537210858010528, -0.059423525226001202,
        -0.0063860373221385318, 0.11992593543150673, -0.016224313722952306,
        0.1179079716819904, 0.06468044932840758, 0.015143083729458221,
        -0.011402957916733165, 0.052841065220381671, 0.0074396821766865989
    ]), 
    translation_vector=np.array([0.0, 0.0, 0.0]),
    rotation_matrix=RotationMatrix(np.array([
            [0, 1, 0],   # c1 = b2
            [0, 0, 1],   # c2 = b3
            [1, 0, 0]    # c3 = b1
        ]))
    )