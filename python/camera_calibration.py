"""
TODO: 
"""

from typing import List
from dataclasses import dataclass, field
import os
import cv2
import numpy as np
import glob
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation
import argparse
import time
from src.camera import Camera

rr.init("Camera Calibration", spawn=True)

blueprint = rrb.Horizontal(
    rrb.Spatial3DView(origin="/world", name="World"),
    rrb.Spatial2DView(origin="/world/camera", name="Camera", contents=["/world/**"]),
)

# Create empty arg parser for rerun
parser = argparse.ArgumentParser(description="argparser")

rr.script_add_args(parser)

args = parser.parse_args()

rr.script_setup(args, "Camera Calibration", default_blueprint=blueprint)


@dataclass
class ChessboardImage:
    filename: str
    resolution: tuple
    image: np.ndarray
    corners_found: bool
    corners: np.ndarray
    rvec: Rotation = None
    tvec: np.ndarray = None

    def recover_pose(self, pattern_size, square_size, camera: Camera):
        """
        TODO: 
        """
        grid_points = construct_grid_points(pattern_size, square_size)
        success, rvec, tvec = cv2.solvePnP(grid_points, self.corners, camera.camera_matrix, camera.dist_coeffs)
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


@dataclass
class ChessboardData:
    """
    TODO:
    """
    chessboard_images: List[ChessboardImage] = field(default_factory=list)
    square_size: float = 0.022
    pattern_size: tuple = (10, 7)

    def draw_corners(self):
        for chessboard_image in self.chessboard_images:
            chessboard_image.draw_chessboard_corners()


    def recover_poses(self, camera: Camera):
        for chessboard_image in self.chessboard_images:
            chessboard_image.recover_pose(self.pattern_size, self.square_size, camera)

# the convention in the computer vision field is to have the X-axis of the camera frame pointing to the right,
#  the Y-axis downward and the Z-axis forward

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


def calibrate_camera(path: str, visualise: bool = False, export: bool = False, pattern_size=(10,7)):

    """

    Coordinate grid for chessboard layout
    (10x7 internal corners):

    (0,0)---(1,0)---(2,0)---...---(9,0)
    |       |       |             |
    (0,1)---(1,1)---(2,1)---...---(9,1)
    |       |       |             |
    (0,2)---(1,2)---(2,2)---...---(9,2)
    |       |       |             |
    ...     ...     ...           ...
    |       |       |             |
    (0,6)---(1,6)---(2,6)---...---(9,6)

    """

    chessboard_data = ChessboardData(pattern_size=pattern_size)

    # square_size = 1.0 # Size of squares in your units (mm, cm, etc.) 

    # Termination criteria for sub-pixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    # Note corner position are not vectors from camera to points, they are corner positions relative to a coordinate system attached to the chessboard.
    corner_position = np.zeros((pattern_size[1]*pattern_size[0], 3), np.float32) # n x m corners x 3 coordinates (X, Y, Z)

    # Create coordinate grids, i.e two 2D arrays for X coordinates and Y coordinates to represent each corner 
    grid = np.mgrid[0:pattern_size[0], 0:pattern_size[1]]   # Each array has shape pattern_size[0] x pattern_size[1] 
    grid_transposed = grid.T  # Transpose to get [Y coordinates (grid), X coodinates (grid)]
    grid_reshaped = grid_transposed.reshape(-1, 2) # Set number of columns to 2 and automatically determine what the number of rows will be 

    corner_position[:, :2] = grid_reshaped

    # Arrays to store object points and image points from all the images
    object_points = [] # 3D point in space
    image_points = [] # 2D points in image plane

    image_paths = glob.glob(path)
    
    if not image_paths:
        raise FileNotFoundError(f"No images found at: {path}")

    # construct the chessbaord data object and populate it with chessboard images
    for path in image_paths:

        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        print("Processing image:", path)
        
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        # If found, add object points, image points (after refining them)
        if ret:
            object_points.append(corner_position)

            refined_corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            image_points.append(refined_corners)

            chessboard_image = ChessboardImage(filename=os.path.basename(path), image=image, corners_found=ret, corners=refined_corners, resolution=image.shape[:2])
            chessboard_data.chessboard_images.append(chessboard_image)

            if visualise:
                # Draw and display the corners
                cv2.drawChessboardCorners(image, pattern_size, refined_corners, ret)

                cv2.namedWindow('Annotated Chessboard', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Annotated Chessboard', 1280, 720)
                cv2.imshow('Annotated Chessboard', image)

                if cv2.waitKey(500) & 0xFF == ord('q'):          
                    break
   


    cv2.destroyAllWindows()

    # Perform camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

    camera = Camera(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

    # Recover camera pose for each chessboard image
    chessboard_data.recover_poses(camera)


    def print_calibration():
        print("Camera Matrix:")
        print(camera_matrix)
        print("Distortion Coefficients:")
        print(dist_coeffs)

    # # Undistortion
    # image = cv.imread(chessboard_data[0])
    # h,  w = image.shape[:2]
    # new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
    # undistored_image = cv.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    # # Crop the image
    # x, y, w, h = roi
    # undistored_image = undistored_image[y:y+h, x:x+w]

    # Compute reprojection error
    mean_error = 0
    for i in range(len(object_points)):
        projected_image_points, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(image_points[i], projected_image_points, cv2.NORM_L2)/len(projected_image_points)
        mean_error += error
    
    print("Total reprojection error: {} (pixels)".format(mean_error/len(object_points)))

    if export:
        export_camera_calibration_data(camera_matrix, dist_coeffs, mean_error, object_points)
        
def export_camera_calibration_data(camera_matrix, dist_coeffs, mean_error, object_points):
    """
    Export the camera calibration data to a JSON file.

    :param camera_matrix: The intrinsic camera matrix obtained from calibration.
    :type camera_matrix: np.ndarray
    :param dist_coeffs: The distortion coefficients obtained from calibration.
    :type dist_coeffs: np.ndarray
    :param mean_error: The mean reprojection error calculated from the calibration.
    :type mean_error: float
    :param object_points: The object points used for calibration.
    :type object_points: list[np.ndarray]
    """
    import json
    data = {
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.tolist(),
        'reprojection_error': mean_error/len(object_points)
    }
    with open('camera.json', 'w') as f:
        json.dump(data, f)

# Get the directory where the current Python script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build absolute path relative to this script
data_path = os.path.join(script_dir, '../../data/images/*.JPG')

calibrate_camera(data_path, visualise=False, export=False, pattern_size=(10,7))


def draw_corners():
    # for chessboardImage in chessboardImages:
    #     chessboardImage.drawCorners(chessboard)
    # void ChessboardData::drawCorners()
    # {
    #     for (auto & chessboardImage : chessboardImages)
    #     {
    #         chessboardImage.drawCorners(chessboard);
    #     }
    # }
    return 



def calc_field_of_view(camera):
    assert camera.camera_matrix.shape == (3, 3)
    assert camera.camera_matrix.dtype == np.float64

    return




def body_to_camera(Tnb, Tbc):
    # Pose Camera::bodyToCamera(const Pose & Tnb) const
# {
#     // Tnc = Tnb*Tbc
#     return Tnb*Tbc;
# }
    return Tnb * Tbc

def world_to_vector(rPNn, Tnb, Tbc):

    Tnc = body_to_camera(Tnb, Tbc)
    rPCc = Tnc.rotation_matrix @ (rPNn - Tnc.translation_vector)
    uPCc = rPCc / np.linalg.norm(rPCc)

    return uPCc