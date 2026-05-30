"""
TODO:
"""

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
import cv2
import rerun as rr
import time

@dataclass
class ChessboardImage:
    """
    TODO:
    """
    filename: str
    image: np.ndarray
    resolution: tuple = None # TODO: Fix this, currrently require argument
    corners_found: bool = False
    corners: np.ndarray = None
    rvec: Rotation = None
    tvec: np.ndarray = None

    def construct_grid_points(pattern_size, square_size):
        """
        Construct the 3D coordinates of the chessboard corners in the chessboard's coordinate frame.
        :param pattern_size: The number of internal corners per chessboard row and column (e.g., (10, 7)).
        :type pattern_size: tuple
        :param square_size: The side length of each square in the chessboard.
        :type square_size: float
        :return: The 3D coordinates of the chessboard corners in the chessboard's coordinate frame.
        :rtype: list[list[float]]
        """
        x, y = np.meshgrid(np.arange(pattern_size[0]) * square_size, np.arange(pattern_size[1]) * square_size)

        grid_points = np.stack([x.ravel(), y.ravel(), np.zeros_like(x.ravel())], axis=1)
        
        return grid_points

    def recover_pose(self, pattern_size, square_size):
        """
        TODO: 
        """
        grid_points = self.construct_grid_points(pattern_size, square_size)
        success, rvec, tvec = cv2.solvePnP(grid_points, self.corners, camera_matrix, dist_coeffs)
        print("rvec: ", rvec.flatten())
        print("tvec: ", tvec.flatten())

            # TODO: SPEND SOME MORE TIME HERE TO FIGURE OUT WHATS HAPPENING .
        # === 1. Convert OpenCV output to proper numpy arrays ===
        rvec = rvec.ravel()          # shape (3,)
        tvec = tvec.ravel()          # shape (3,)

        # === 2. Compute the INVERSE pose → camera in world coordinates ===
        R_board_to_cam = Rotation.from_rotvec(rvec).as_matrix()   # R = world → camera
        t_board_to_cam = tvec

        # AI Camera pose in world = inverse(T_board_to_cam)
        R_cam_to_world = R_board_to_cam.T
        t_cam_to_world = -R_cam_to_world @ t_board_to_cam

        # AI Store both for later use if needed
        self.rvec = Rotation.from_matrix(R_cam_to_world)   # camera orientation
        self.tvec = t_cam_to_world                         # camera position in world

        # self.rvec = Rotation.from_rotvec(rvec.flatten())
        # self.tvec = tvec.flatten()

        # Compute the inverse pose to get the camera's position and orientation in the world frame

        if success:
            # Log the camera extrinsics as a transform, this will move the camera frustrum to the correct position in the world
            rr.log("world/camera", rr.Transform3D(translation=self.tvec, rotation=rr.Quaternion(xyzw=self.rvec.as_quat())))

            # Log a visible point at the camera origin
            # rr.log("world/camera/origin", rr.Points3D(np.zeros((1, 3)), colors=np.array([[255, 0, 0]]), radii=0.005))

            # R = self.rvec.as_matrix()
            # corners_3d_world = (R @ grid_points.T).T + self.tvec

            rr.log("world/chessboard_corners", rr.Points3D(grid_points, colors=np.array([[0, 255, 0]]), radii=0.0015))

            rr.log("world/camera/pinhole", rr.Pinhole(image_from_camera=camera.camera_matrix, resolution=(self.resolution[1], self.resolution[0]), camera_xyz=rr.ViewCoordinates.RDF))

             # Draw and display the corners
            cv2.drawChessboardCorners(self.image, pattern_size, self.corners, self.corners_found)

            cv2.namedWindow('Annotated Chessboard', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Annotated Chessboard', 1280, 720)
            cv2.imshow('Annotated Chessboard', self.image)

            if cv2.waitKey(500) & 0xFF == ord('q'):          
                return

            time.sleep(10)
        else: 
            raise RuntimeError(f"Failed to recover pose for image {self.filename}")