"""
TODO:

Default chessboard layout

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


from dataclasses import dataclass, field
from typing import List
import numpy as np
from .chessboard_image import ChessboardImage
from .calibration_data import CalibrationData
import cv2
import os
import glob 
from pathlib import Path

VIDEO_EXTENSIONS = {'.mp4', '.mov'}

@dataclass
class ChessboardData:
    """
    TODO:
    """
    square_size: float
    pattern_size: tuple[int, int]
    resolution: tuple[int, int] = None
    chessboard_images: List[ChessboardImage] = field(default_factory=list)

    def draw_corners(self):
        """
        TODO:
        """
        for chessboard_image in self.chessboard_images:
            chessboard_image.draw_chessboard_corners()

    def recover_poses(self):
        """
        TODO:
        """
        for chessboard_image in self.chessboard_images:
            chessboard_image.recover_pose(self.pattern_size, self.square_size)
        
    def read_images_from_directory(self, directory_path: str) -> List[ChessboardImage]:
        """
        TODO:
        """
        image_paths = glob.glob(directory_path)

        if not image_paths:
            raise FileNotFoundError(f"No images found at: {directory_path}")
        
        self.chessboard_images = []
        
        for path in image_paths:
            image = cv2.imread(path)
            chessboard_image = ChessboardImage(filename=os.path.basename(path), image=image, resolution=image.shape[:2])
            self.chessboard_images.append(chessboard_image)

        return self.chessboard_images

    def read_images_from_video(self, path: str) -> List[ChessboardImage]:
        """
        TODO:

        Cycle through every 20 frames in the video and return the images 
        :param path: The file path to the video containing the chessboard images.
        :type path: str
        :return: TODO
        :rtype: TODO
        """
        cap = cv2.VideoCapture(path)
        
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {path}")

        frame_count = 0
        self.chessboard_images = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 250 == 0:  # Process every 5th frame
                chessboard_image = ChessboardImage(filename=f"frame_{frame_count}", image=frame)
                self.chessboard_images.append(chessboard_image)

            frame_count += 1

        cap.release()
        return self.chessboard_images


    def calibrate_from_images(self, chessboard_images: List[ChessboardImage], visualise: bool = False) -> None:
        """
        TODO: Broken in current state
        """

        # Termination criteria for sub-pixel refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

        # Note corner position are not vectors from camera to points, they are corner positions relative to a coordinate system attached to the chessboard.
        corner_position = np.zeros((self.pattern_size[1]*self.pattern_size[0], 3), np.float32) # n x m corners x 3 coordinates (X, Y, Z)

        # Create coordinate grids, i.e two 2D arrays for X coordinates and Y coordinates to represent each corner 
        grid = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]]   # Each array has shape pattern_size[0] x pattern_size[1] 
        grid_transposed = grid.T  # Transpose to get [Y coordinates (grid), X coodinates (grid)]
        grid_reshaped = grid_transposed.reshape(-1, 2) # Set number of columns to 2 and automatically determine what the number of rows will be 

        corner_position[:, :2] = grid_reshaped

        # Arrays to store object points and image points from all the images
        object_points = [] # 3D point in space
        image_points = [] # 2D points in image plane

        print("Starting calibration using {} images...".format(len(chessboard_images)))
        
        self.resolution = chessboard_images[0].resolution
        print("Image resolution: ", self.resolution)

        for chessboard_image in chessboard_images:
            
            print("Processing image:", chessboard_image.filename)
            gray = cv2.cvtColor(chessboard_image.image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)

            # TODO: Check current image has same resolution if not then throw error 

            # If found, add object points, image points (after refining them)
            if ret:
                object_points.append(corner_position)

                refined_corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                image_points.append(refined_corners)

                # chessboard_image = ChessboardImage(filename=os.path.basename(path), image=image, corners_found=ret, corners=refined_corners, resolution=image.shape[:2])

                if visualise:
                    # Draw and display the corners
                    cv2.drawChessboardCorners(chessboard_image.image, self.pattern_size, refined_corners, ret)

                    cv2.namedWindow('Annotated Chessboard', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Annotated Chessboard', 1280, 720)
                    cv2.imshow('Annotated Chessboard', chessboard_image.image)

                    if cv2.waitKey(500) & 0xFF == ord('q'):          
                        break

        cv2.destroyAllWindows()

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
        
        mean_error = 0
        for i in range(len(object_points)):
            projected_image_points, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv2.norm(image_points[i], projected_image_points, cv2.NORM_L2)/len(projected_image_points)
            mean_error += error
        
        reprojection_error = mean_error / len(object_points)

        print("Total reprojection error: {} (pixels)".format(reprojection_error))
        print("Camera matrix:\n", camera_matrix)
        print("Distortion coefficients:\n", dist_coeffs)
        
        return CalibrationData(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, reprojection_error=reprojection_error)


    def calibrate_camera(self, path: str, visualise: bool = False, export: bool = False):
        """
        TODO:
        :param path: The file path to the directory containing the chessboard images or a video file.
        :type path: str
        :param visualise: Whether to visualise the detected corners on the chessboard images.
        :type visualise: bool
        :param export: Whether to export the camera calibration data to a JSON file.
        :type export: bool
        """
        suffix = Path(path).suffix.lower()
        
        if suffix in VIDEO_EXTENSIONS:
            images = self.read_images_from_video(path)
        else:
            images = self.read_images_from_directory(path)

        calibration_result = self.calibrate_from_images(images, visualise=visualise)

        if export:
            self.export_to_json(calibration_result)


    def export_to_json(self, calibration_result: CalibrationData) -> None:
        """
        Export the camera calibration data to a JSON file.

        Includes the camera matrix, distortion coefficients, and reprojection error.

        :param calibration_result: The camera calibration data to be exported.
        :type calibration_result: CalibrationData
        """
        import json
        data = {
            'camera_matrix': calibration_result.camera_matrix.tolist(),
            'dist_coeffs': calibration_result.dist_coeffs.tolist(),
            'reprojection_error': calibration_result.reprojection_error
        }
        with open('camera.json', 'w') as f:
            json.dump(data, f)

        
        