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
        if ret == True:
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

data_path = '../../data/images/*.JPG'

calibrate_camera(data_path, visualise=False, export=False, pattern_size=(10,7))


# // Group operation of SE(3)
# //
# // Tac = Tab*Tbc
# //
# // [ Rac, rCAa ] = [ Rab, rBAa ] * [ Rbc, rCBb ] = [ Rab*Rbc, Rab*rCBb + rBAa ]
# // [   0,    1 ]   [   0,    1 ]   [   0,    1 ]   [       0,               1 ]
# /

# Pose Pose::operator*(const Pose & other) const
# {
#     Pose result;
#     result.rotationMatrix = rotationMatrix * other.rotationMatrix;
#     result.translationVector = rotationMatrix * other.translationVector + translationVector;
#     return result;
# }

# // Inverse element in SE(3)
# //
# // Tab^-1 = Tba
# //
# // [ Rab, rBAa ]^-1 = [ Rba, rABb ] = [ Rab^T, -Rab^T*rBAa ]
# // [   0,    1 ]      [   0,    1 ]   [     0,           1 ]
# //
# Pose Pose::inverse() const
# {
#     Pose result;
#     result.rotationMatrix = rotationMatrix.t();
#     result.translationVector = -result.rotationMatrix * translationVector;
#     return result;
# }

# void Chessboard::write(cv::FileStorage & fs) const
# {
#     fs << "{"
#        << "grid_width"  << boardSize.width
#        << "grid_height" << boardSize.height
#        << "square_size" << squareSize
#        << "}";
# }

# void Chessboard::read(const cv::FileNode & node)
# {
#     node["grid_width"]  >> boardSize.width;
#     node["grid_height"] >> boardSize.height;
#     node["square_size"] >> squareSize;
# }

# std::vector<cv::Point3f> Chessboard::gridPoints() const
# {
#     std::vector<cv::Point3f> rPNn_all;
#     rPNn_all.reserve(boardSize.height*boardSize.width);
#     for (int i = 0; i < boardSize.height; ++i)
#         for (int j = 0; j < boardSize.width; ++j)
#             rPNn_all.push_back(cv::Point3f(j*squareSize, i*squareSize, 0));   
#     return rPNn_all; 
# }

# std::ostream & operator<<(std::ostream & os, const Chessboard & chessboard)
# {
#     return os << "boardSize: " << chessboard.boardSize << ", squareSize: " << chessboard.squareSize;
# }

# ChessboardImage::ChessboardImage(const cv::Mat & image_, const Chessboard & chessboard, const std::filesystem::path & filename_)
#     : image(image_)
#     , filename(filename_)
#     , isFound(false)
# {
#     // Convert image to grayscale
#     //std::cout << "image:" << image << std::endl;
#     cv::Mat gray;
#     cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

#     // Detect chessboard corners
#     cv::Size patternSize(chessboard.boardSize.width, chessboard.boardSize.height);
#     std::cout << "Pattern Size:" << patternSize << std::endl;
#     isFound = cv::findChessboardCorners(gray, patternSize, corners,
#         cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

#     // If corners are found, do subpixel refinement
#     if (isFound)
#     {
#         std::cout << "Corners found, doing subpixel refinement..." << std::endl;
#         cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1);
#         cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria);
#     }
#     else
#     {
#         std::cout << "Corners not found!" << std::endl;
#     }
# }

# void ChessboardImage::drawCorners(const Chessboard & chessboard)
# {
#     cv::drawChessboardCorners(image, chessboard.boardSize, corners, isFound);
# }

# void ChessboardImage::drawBox(const Chessboard &chessboard, const Camera &camera)
# {
#     const double boxHeight = -0.23;  // Height of the box in meters
#     const int numSegmentsPerEdge = 200;  // Number of segments per edge for smooth curves
#     const double offsetX = 0.0;  // Offset for interior corners (currently not used)
#     const double offsetY = 0.0;  // Offset for interior corners (currently not used)

#     // Define the 8 vertices of the box in world coordinates
#     std::vector<cv::Vec3d> vertices(8);
#     double width = (chessboard.boardSize.width - 1) * chessboard.squareSize;
#     double height = (chessboard.boardSize.height - 1) * chessboard.squareSize;
    
#     // Base vertices (bottom of the box)
#     vertices[0] = cv::Vec3d(offsetX, offsetY, 0);  // Bottom-left
#     vertices[1] = cv::Vec3d(width - offsetX, offsetY, 0);  // Bottom-right
#     vertices[2] = cv::Vec3d(width - offsetX, height - offsetY, 0);  // Top-right
#     vertices[3] = cv::Vec3d(offsetX, height - offsetY, 0);  // Top-left

#     // Top vertices
#     for (int i = 0; i < 4; ++i)
#     {
#         vertices[i + 4] = vertices[i] + cv::Vec3d(0, 0, boxHeight);
#     }

#     // Define the edges of the box with their corresponding colors
#     const std::vector<std::tuple<int, int, cv::Scalar>> edges = {
#         {0, 1, cv::Scalar(0, 255, 0)},  // X direction (Green)
#         {1, 2, cv::Scalar(255, 0, 0)},  // Y direction (Blue)
#         {2, 3, cv::Scalar(0, 255, 0)},  // X direction (Green)
#         {3, 0, cv::Scalar(255, 0, 0)},  // Y direction (Blue)
#         {4, 5, cv::Scalar(0, 255, 0)},  // X direction (Green)
#         {5, 6, cv::Scalar(255, 0, 0)},  // Y direction (Blue)
#         {6, 7, cv::Scalar(0, 255, 0)},  // X direction (Green)
#         {7, 4, cv::Scalar(255, 0, 0)},  // Y direction (Blue)
#         {0, 4, cv::Scalar(0, 0, 255)},  // Z direction (Red)
#         {1, 5, cv::Scalar(0, 0, 255)},  // Z direction (Red)
#         {2, 6, cv::Scalar(0, 0, 255)},  // Z direction (Red)
#         {3, 7, cv::Scalar(0, 0, 255)},  // Z direction (Red)
#     };

#     Pose Tcn = Tnc.inverse();

#     // Lambda function to check if a point is within the camera's field of view
#     auto isWithinFOV = [&](const cv::Vec3d& worldPoint) -> bool {
#         cv::Vec3d cameraCoords = camera.worldToVector(worldPoint, Tcn);
#         return camera.isVectorWithinFOV(cameraCoords);
#     };

#     // Draw the edges of the box
#     for (const auto &[startIdx, endIdx, color] : edges)
#     {
#         cv::Vec3d startVertex = vertices[startIdx];
#         cv::Vec3d endVertex = vertices[endIdx];
#         cv::Point2f prevImagePoint;
#         bool prevPointValid = false;

#         for (int i = 0; i <= numSegmentsPerEdge; ++i)
#         {
#             double t = static_cast<double>(i) / numSegmentsPerEdge;
#             cv::Vec3d worldPoint = startVertex * (1.0 - t) + endVertex * t;

#             if (isWithinFOV(worldPoint)) {
#                 try {
#                     cv::Vec2d imagePoint = camera.worldToPixel(worldPoint, Tcn);
#                     cv::Point2f currentImagePoint(imagePoint[0], imagePoint[1]);

#                     if (prevPointValid) {
#                         // Determine line thickness based on color
#                         int thickness = (color == cv::Scalar(0, 0, 255)) ? 7 : 
#                                         (color == cv::Scalar(0, 255, 0)) ? 10 : 2;

#                         cv::line(image, prevImagePoint, currentImagePoint, color, thickness, cv::LINE_AA);
#                     }

#                     prevImagePoint = currentImagePoint;
#                     prevPointValid = true;
#                 }
#                 catch (const std::exception& e) {
#                     std::cerr << "Error projecting point: " << e.what() << std::endl;
#                 }
#             } else {
#                 prevPointValid = false;
#             }
#         }
#     }

#     // Draw circles at the base vertices for debugging
#     for (int i = 0; i < 4; ++i) {
#         if (isWithinFOV(vertices[i])) {
#             cv::Vec2d imagePoint = camera.worldToPixel(vertices[i], Tcn);
#             cv::circle(image, cv::Point(imagePoint[0], imagePoint[1]), 5, cv::Scalar(255, 0, 255), -1);
#         }
#     }
# }

# def recover_pose(todo, camera: Camera):
#     rPNn = todo  # Placeholder for actual chessboard grid points

#     cv2.solvePnP()
#     thetacn, rNCc = None, None
#     Tcn = Pose(thetacn, rNCc)
#     return Tcn.inverse()
# void ChessboardImage::recoverPose(const Chessboard & chessboard, const Camera & camera)
# {
#     std::vector<cv::Point3f> rPNn_all = chessboard.gridPoints();

#     cv::Mat Thetacn, rNCc;
#     cv::solvePnP(rPNn_all, corners, camera.cameraMatrix, camera.distCoeffs, Thetacn, rNCc);

#     Pose Tcn(Thetacn, rNCc);
#     Tnc = Tcn.inverse();
# }

# ChessboardData::ChessboardData(const std::filesystem::path & configPath)
# {
#     // Ensure the config file exists
#     assert(std::filesystem::exists(configPath));

#     // Open the config file
#     cv::FileStorage fs(configPath.string(), cv::FileStorage::READ);
#     assert(fs.isOpened());

#     // Read chessboard configuration
#     cv::FileNode node = fs["chessboard_data"];
#     node["chessboard"] >> chessboard;
#     std::cout << "Chessboard: " << chessboard << std::endl;

#     // Read file pattern for chessboard images
#     std::string pattern;
#     node["file_regex"] >> pattern;
#     fs.release();

#     // Create regex object from pattern
#     std::regex re(pattern, std::regex_constants::basic | std::regex_constants::icase);
    
#     // Get the directory containing the config file
#     std::filesystem::path root = configPath.parent_path();
#     std::cout << "Scanning directory " << root.string() << " for file pattern \"" << pattern << "\"" << std::endl;

#     // Populate chessboard images from regex
#     chessboardImages.clear();
#     if (std::filesystem::exists(root) && std::filesystem::is_directory(root))
#     {
#         // Iterate through all files in the directory and its subdirectories
#         for (const auto & p : std::filesystem::recursive_directory_iterator(root))
#         {
#             if (std::filesystem::is_regular_file(p))
#             {
#                 // Check if the file matches the regex pattern
#                 if (std::regex_match(p.path().filename().string(), re))
#                 {
#                     std::cout << "Loading " << p.path().filename().string() << "..." << std::flush;

#                     // Try to load the file as an image
#                     cv::Mat image = cv::imread(p.path().string(), cv::IMREAD_COLOR);

#                     bool isImage = !image.empty();
#                     if (isImage)
#                     {
#                         // If it's an image, detect chessboard
#                         std::cout << " done, detecting chessboard..." << std::flush;
#                         ChessboardImage ci(image, chessboard, p.path().filename());
#                         std::cout << (ci.isFound ? " found" : " not found") << std::endl;
#                         if (ci.isFound)
#                         {
#                             chessboardImages.push_back(ci);
#                         }
#                     }
#                     else
#                     {
#                         // If it's not an image, try to load it as a video
#                         cv::VideoCapture cap(p.path().string());
#                         bool isVideo = cap.isOpened();
#                         if (isVideo)
#                         {
#                             // Get number of video frames
#                             int nFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT)); // TODO
#                             std::cout << " done, found " << nFrames << " frames" << std::endl;

#                             // Loop through selected frames
#                             for (int idxFrame = 0; idxFrame < nFrames; idxFrame += std::max(1, nFrames / 10))
#                             {
#                                 // Read frame
#                                 std::cout << "Reading " << p.path().filename().string() << " frame " << idxFrame << "..." << std::flush;
#                                 cv::Mat frame;
#                                 // TODO
#                                 cap.set(cv::CAP_PROP_POS_FRAMES, idxFrame);
#                                 cap.read(frame);
#                                 if (frame.empty())
#                                 {
#                                     std::cout << " end of file found" << std::endl;
#                                     break;
#                                 }

#                                 // Detect chessboard in frame
#                                 std::cout << " done, detecting chessboard..." << std::flush;
#                                 std::string baseName = p.path().stem().string();
#                                 std::string frameFilename = std::format("{}_{:05d}.jpg", baseName, idxFrame);
#                                 ChessboardImage ci(frame, chessboard, frameFilename);
#                                 std::cout << (ci.isFound ? " found" : " not found") << std::endl;
#                                 if (ci.isFound)
#                                 {
#                                     chessboardImages.push_back(ci);
#                                 }
#                             }
#                             cap.release();
#                         }
#                     }
#                 }
#             }
#         }
#     }
# }

# void ChessboardData::drawCorners()
# {
#     for (auto & chessboardImage : chessboardImages)
#     {
#         chessboardImage.drawCorners(chessboard);
#     }
# }

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

# void ChessboardData::drawBoxes(const Camera & camera)
# {
#     for (auto & chessboardImage : chessboardImages)
#     {
#         chessboardImage.drawBox(chessboard, camera);
#     }
# }

# void ChessboardData::recoverPoses(const Camera & camera)
# {
#     for (auto & chessboardImage : chessboardImages)
#     {
#         chessboardImage.recoverPose(chessboard, camera);
#     }
# }

# void Camera::calibrate(ChessboardData & chessboardData)
# {
#     std::vector<cv::Point3f> rPNn_all = chessboardData.chessboard.gridPoints();

#     std::vector<std::vector<cv::Point2f>> rQOi_all;
#     std::vector<std::vector<cv::Point3f>> rPNn_allImages;
#     std::vector<ChessboardImage*> validChessboardImages;

#     for (auto & chessboardImage : chessboardData.chessboardImages)
#     {
#         if (chessboardImage.isFound)
#         {
#             rQOi_all.push_back(chessboardImage.corners);
#             rPNn_allImages.push_back(rPNn_all);
#             validChessboardImages.push_back(&chessboardImage);
#         }
#     }
    
#     if (rQOi_all.empty()) {
#         throw std::runtime_error("No valid chessboard images found for calibration");
#     }

#     imageSize = validChessboardImages[0]->image.size();
    
#     flags = cv::CALIB_RATIONAL_MODEL | cv::CALIB_THIN_PRISM_MODEL;

#     // Find intrinsic and extrinsic camera parameters
#     cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
#     distCoeffs = cv::Mat::zeros(12, 1, CV_64F);
#     std::vector<cv::Mat> Thetacn_all, rNCc_all;
#     std::cout << "Calibrating camera..." << std::flush;
    
#     double rms = cv::calibrateCamera(rPNn_allImages, rQOi_all, imageSize, cameraMatrix, distCoeffs, 
#                                      Thetacn_all, rNCc_all, flags);
    
#     std::cout << " done" << std::endl;
    
#     // Calculate horizontal, vertical and diagonal field of view
#     calcFieldOfView();

#     // Write extrinsic camera parameters for each valid chessboard image
#     assert(validChessboardImages.size() == rNCc_all.size());
#     assert(validChessboardImages.size() == Thetacn_all.size());
#     for (std::size_t k = 0; k < validChessboardImages.size(); ++k)
#     {
#         // Set the camera orientation and position (extrinsic camera parameters)
#         Pose & Tnc = validChessboardImages[k]->Tnc;
#         cv::Mat R;
#         cv::Rodrigues(Thetacn_all[k], R);
#         Tnc.rotationMatrix = cv::Matx33d(R);
#         Tnc.translationVector = cv::Vec3d(rNCc_all[k]);
#     }
    
#     printCalibration();
#     std::cout << std::setw(30) << "RMS reprojection error: " << rms << std::endl;

#     assert(cv::checkRange(cameraMatrix));
#     assert(cv::checkRange(distCoeffs));
# }

# void Camera::printCalibration() const
# {
#     std::bitset<8*sizeof(flags)> bitflag(flags);
#     std::cout << std::endl << "Calibration data:" << std::endl;
#     std::cout << std::setw(30) << "Bit flags: " << bitflag << std::endl;
#     std::cout << std::setw(30) << "cameraMatrix:\n" << cameraMatrix << std::endl;
#     std::cout << std::setw(30) << "distCoeffs:\n" << distCoeffs.t() << std::endl;
#     std::cout << std::setw(30) << "Focal lengths: " 
#               << "(fx, fy) = "
#               << "("<< cameraMatrix.at<double>(0, 0) << ", "<< cameraMatrix.at<double>(1, 1) << ")"
#               << std::endl;       
#     std::cout << std::setw(30) << "Principal point: " 
#               << "(cx, cy) = "
#               << "("<< cameraMatrix.at<double>(0, 2) << ", "<< cameraMatrix.at<double>(1, 2) << ")"
#               << std::endl;     
#     // std::cout << std::setw(30) << "Field of view (horizontal): " << 180.0/CV_PI*hFOV << " deg" << std::endl; // Origin print statements 
#     // std::cout << std::setw(30) << "Field of view (vertical): " << 180.0/CV_PI*vFOV << " deg" << std::endl;
#     // std::cout << std::setw(30) << "Field of view (diagonal): " << 180.0/CV_PI*dFOV << " deg" << std::endl;
#     std::cout << std::setw(30) << "Field of view (horizontal): " << hFOV << " deg" << std::endl;
#     std::cout << std::setw(30) << "Field of view (vertical): " << vFOV << " deg" << std::endl; 
#     std::cout << std::setw(30) << "Field of view (diagonal): " << dFOV << " deg" << std::endl;
# }

def calc_field_of_view(camera):
    assert camera.camera_matrix.shape == (3, 3)
    assert camera.camera_matrix.dtype == np.float64

    return

# void Camera::calcFieldOfView()
# {


#     // Calculate horizontal FOV
#     cv::Vec3d leftVector = pixelToVector(cv::Vec2d(0, imageSize.height / 2));
#     cv::Vec3d rightVector = pixelToVector(cv::Vec2d(imageSize.width - 1, imageSize.height / 2));
#     hFOV = std::acos(leftVector.dot(rightVector));

#     // Calculate vertical FOV
#     cv::Vec3d topVector = pixelToVector(cv::Vec2d(imageSize.width / 2, 0));
#     cv::Vec3d bottomVector = pixelToVector(cv::Vec2d(imageSize.width / 2, imageSize.height - 1));
#     vFOV = std::acos(topVector.dot(bottomVector));

#     // Calculate diagonal FOV
#     cv::Vec3d topLeftVector = pixelToVector(cv::Vec2d(0, 0));
#     cv::Vec3d bottomRightVector = pixelToVector(cv::Vec2d(imageSize.width - 1, imageSize.height - 1));
#     dFOV = std::acos(topLeftVector.dot(bottomRightVector));

#     // Convert radians to degrees
#     hFOV = hFOV * 180.0 / CV_PI;
#     vFOV = vFOV * 180.0 / CV_PI;
#     dFOV = dFOV * 180.0 / CV_PI;
# }

# Pose Camera::cameraToBody(const Pose & Tnc) const
# {
#     // Tnb = Tnc*Tcb
#     return Tnc*Tbc.inverse();
# }


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

# cv::Vec3d Camera::worldToVector(const cv::Vec3d & rPNn, const Pose & Tnb) const
# {
#     // Camera pose Tnc (i.e., Rnc, rCNn)
#     Pose Tnc = bodyToCamera(Tnb); // Tnb*Tbc

#     // Compute the unit vector uPCc from the world position rPNn and camera pose Tnc
#     cv::Vec3d uPCc;
    
#     // TODO
#     // Compute the vector from the camera to the point in camera coordinates
#     cv::Vec3d rPCc = Tnc.rotationMatrix.t() * (rPNn - Tnc.translationVector);

#     // Normalize the resulting vector to ensure it represents only direction (unit vector)
#     uPCc = rPCc / cv::norm(rPCc);

#     return uPCc;
# }

# cv::Vec2d Camera::worldToPixel(const cv::Vec3d & rPNn, const Pose & Tnb) const
# {
#     return vectorToPixel(worldToVector(rPNn, Tnb));
# }

# cv::Vec2d Camera::vectorToPixel(const cv::Vec3d & rPCc) const

#     // Project the point
#     std::vector<cv::Point3f> objectPoints = {cv::Point3f(uPCc[0], uPCc[1], uPCc[2])};
#     std::vector<cv::Point2f> imagePoints;

#     cv::projectPoints(objectPoints, cv::Vec3d::zeros(), cv::Vec3d::zeros(), 
#                       cameraMatrix, distCoeffs, imagePoints);

#     // Return the pixel coordinates
#     return cv::Vec2d(imagePoints[0].x, imagePoints[0].y);
# }

def vector_to_pixel(rPCc):
    """Compute the pixel location (rQOi) for the given unit vector (uPCc)."""

    # Normalise the input vector
    uPCc = rPCc / np.linalg.norm(rPCc)

    # Project the point
    object_points = np.array([[uPCc[0], uPCc[1], uPCc[2]]], dtype=np.float32)

    return

def pixel_to_vector(rQOi, camera): 
    """Compute unit vector (uPCc) for the given pixel location (rQOi)."""

    image_points = np.array([[rQOi[0], rQOi[1]]], dtype=np.float32)

    # Undistort and normalise the point
    normalised_points = cv2.undistortPoints(image_points, camera.camera_matrix, camera.dist_coeffs)

    # The normalised point is now [x, y, 1] in camera coordinates
    uPCc = np.array([normalised_points[0][0][0], normalised_points[0][0][1], 1.0])

    # Normalise to unit vector
    return uPCc / np.linalg.norm(uPCc)

# WHEN("Calling vectorToPixel")
#             {
#                 cv::Vec2d rQOi = cam.vectorToPixel(uPCc);

#                 THEN("Vector maps to centre of image")
#                 {
#                     // cv::Vec3d uPCc(0.0, 0.0, 1.0);
#                     const double & cx = cam.cameraMatrix.at<double>(0, 2);
#                     const double & cy = cam.cameraMatrix.at<double>(1, 2);
#                     std::cout << "Expected pixel location: (" << cx << ", " << cy << ")" << std::endl;
#                     CHECK(rQOi(0) == doctest::Approx(cx));
#                     CHECK(rQOi(1) == doctest::Approx(cy));
#                 }


def is_vector_within_fov(rPCc): 
    """Check if a 3D vector in camera coordinates is within the camera's field of view."""

    # Normalise the input vector
    uPCc = rPCc / np.linalg.norm(rPCc)

    # Principal direction vector (typically [0, 0, 1] for a camera looking along its z-axis)
    principalDirection = np.array([0, 0, 1])

    # Compute horizontal angle
    horizontalProjection = np.array([uPCc[0], 0, uPCc[2]])
    horizontalProjection = horizontalProjection / np.linalg.norm(horizontalProjection)
    # horizontalAngle = np.arccos(np.clip(np.dot(horizontalProjection, principalDirection), 

# bool Camera::isVectorWithinFOV(const cv::Vec3d & rPCc) const
# {
#     // Normalize the input vector
#     cv::Vec3d uPCc = rPCc / cv::norm(rPCc);

#     // Principal direction vector (typically [0, 0, 1] for a camera looking along its z-axis)
#     cv::Vec3d principalDirection(0, 0, 1);

#     // Compute horizontal angle
#     cv::Vec3d horizontalProjection(uPCc[0], 0, uPCc[2]);
#     horizontalProjection = horizontalProjection / cv::norm(horizontalProjection);
#     double horizontalAngle = std::acos(horizontalProjection.dot(principalDirection));

#     // Compute vertical angle
#     cv::Vec3d verticalProjection(0, uPCc[1], uPCc[2]);
#     verticalProjection = verticalProjection / cv::norm(verticalProjection);
#     double verticalAngle = std::acos(verticalProjection.dot(principalDirection));

#     // Convert angles to degrees
#     double horizontalAngleDeg = horizontalAngle * 180.0 / CV_PI;
#     double verticalAngleDeg = verticalAngle * 180.0 / CV_PI;

#     // Project the vector onto the image plane
#     cv::Vec2d pixel = vectorToPixel(uPCc);

#     // Check if the projected point is within the image bounds
#     bool insideImage = (pixel[0] >= 0 && pixel[0] < imageSize.width &&
#                         pixel[1] >= 0 && pixel[1] < imageSize.height);

#     // Check if both angles are within their respective FOV limits
#     bool insideHorizontalFOV = horizontalAngleDeg <= hFOV / 2.0;
#     bool insideVerticalFOV = verticalAngleDeg <= vFOV / 2.0;

#     return insideImage && insideHorizontalFOV && insideVerticalFOV;

def is_world_within_fov(rPNn, Tnb, camera):
    """ TODO """
    return is_vector_within_fov(world_to_vector(rPNn, Tnb, camera))

# void Camera::write(cv::FileStorage & fs) const
# {
#     fs << "{"
#        << "camera_matrix"           << cameraMatrix
#        << "distortion_coefficients" << distCoeffs
#        << "flags"                   << flags
#        << "imageSize"               << imageSize
#        << "}";
# }

# void Camera::read(const cv::FileNode & node)
# {
#     node["camera_matrix"]           >> cameraMatrix;
#     node["distortion_coefficients"] >> distCoeffs;
#     node["flags"]                   >> flags;
#     node["imageSize"]               >> imageSize;

#     calcFieldOfView();

#     assert(cameraMatrix.cols == 3);
#     assert(cameraMatrix.rows == 3);
#     assert(cameraMatrix.type() == CV_64F);
#     assert(distCoeffs.cols == 1);
#     assert(distCoeffs.type() == CV_64F);
# }


