#include <cassert>
#include <cstddef>
#include <cmath>
#include <limits>
#include <iostream>
#include <format>
#include <vector>
#include <filesystem>
#include <regex>
#include <algorithm>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include "rotation.hpp"
#include "Pose.hpp"
#include "Camera.h"

void Chessboard::write(cv::FileStorage & fs) const
{
    fs << "{"
       << "grid_width"  << boardSize.width
       << "grid_height" << boardSize.height
       << "square_size" << squareSize
       << "}";
}

void Chessboard::read(const cv::FileNode & node)
{
    node["grid_width"]  >> boardSize.width;
    node["grid_height"] >> boardSize.height;
    node["square_size"] >> squareSize;
}

std::vector<cv::Point3f> Chessboard::gridPoints() const
{
    std::vector<cv::Point3f> rPNn_all;
    rPNn_all.reserve(boardSize.height*boardSize.width);
    for (int i = 0; i < boardSize.height; ++i)
        for (int j = 0; j < boardSize.width; ++j)
            rPNn_all.push_back(cv::Point3f(j*squareSize, i*squareSize, 0));   
    return rPNn_all; 
}

std::ostream & operator<<(std::ostream & os, const Chessboard & chessboard)
{
    return os << "boardSize: " << chessboard.boardSize << ", squareSize: " << chessboard.squareSize;
}

ChessboardImage::ChessboardImage(const cv::Mat & image_, const Chessboard & chessboard, const std::filesystem::path & filename_)
    : image(image_)
    , filename(filename_)
    , isFound(false)
{
    // Convert image to grayscale
    //std::cout << "image:" << image << std::endl;
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Detect chessboard corners
    cv::Size patternSize(chessboard.boardSize.width, chessboard.boardSize.height);
    std::cout << "Pattern Size:" << patternSize << std::endl;
    isFound = cv::findChessboardCorners(gray, patternSize, corners,
        cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

    // If corners are found, do subpixel refinement
    if (isFound)
    {
        std::cout << "Corners found, doing subpixel refinement..." << std::endl;
        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1);
        cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria);
    }
    else
    {
        std::cout << "Corners not found!" << std::endl;
    }
}

void ChessboardImage::drawCorners(const Chessboard & chessboard)
{
    cv::drawChessboardCorners(image, chessboard.boardSize, corners, isFound);
}

void ChessboardImage::drawBox(const Chessboard & chessboard, const Camera & camera)
{
    const double boxHeight = -0.23;  // Height of the box in meters
    const int numSegmentsPerEdge = 200;  // Number of segments per edge for smooth curves
    const double offsetX = 0.0;  // Offset for interior corners (currently not used)
    const double offsetY = 0.0;  // Offset for interior corners (currently not used)

    // Define the 8 vertices of the box in world coordinates
    std::vector<cv::Vec3d> vertices(8);
    double width = (chessboard.boardSize.width - 1) * chessboard.squareSize;
    double height = (chessboard.boardSize.height - 1) * chessboard.squareSize;
    
    // Base vertices (bottom of the box)
    vertices[0] = cv::Vec3d(offsetX, offsetY, 0);  // Bottom-left
    vertices[1] = cv::Vec3d(width - offsetX, offsetY, 0);  // Bottom-right
    vertices[2] = cv::Vec3d(width - offsetX, height - offsetY, 0);  // Top-right
    vertices[3] = cv::Vec3d(offsetX, height - offsetY, 0);  // Top-left

    // Top vertices
    for (int i = 0; i < 4; ++i)
    {
        vertices[i + 4] = vertices[i] + cv::Vec3d(0, 0, boxHeight);
    }

    // Define the edges of the box with their corresponding colors
    const std::vector<std::tuple<int, int, cv::Scalar>> edges = {
        {0, 1, cv::Scalar(0, 255, 0)},  // X direction (Green)
        {1, 2, cv::Scalar(255, 0, 0)},  // Y direction (Blue)
        {2, 3, cv::Scalar(0, 255, 0)},  // X direction (Green)
        {3, 0, cv::Scalar(255, 0, 0)},  // Y direction (Blue)
        {4, 5, cv::Scalar(0, 255, 0)},  // X direction (Green)
        {5, 6, cv::Scalar(255, 0, 0)},  // Y direction (Blue)
        {6, 7, cv::Scalar(0, 255, 0)},  // X direction (Green)
        {7, 4, cv::Scalar(255, 0, 0)},  // Y direction (Blue)
        {0, 4, cv::Scalar(0, 0, 255)},  // Z direction (Red)
        {1, 5, cv::Scalar(0, 0, 255)},  // Z direction (Red)
        {2, 6, cv::Scalar(0, 0, 255)},  // Z direction (Red)
        {3, 7, cv::Scalar(0, 0, 255)},  // Z direction (Red)
    };

    Pose Tcn = Tnc.inverse();

    // Lambda function to check if a point is within the camera's field of view
    auto isWithinFOV = [&](const cv::Vec3d& worldPoint) -> bool {
        cv::Vec3d cameraCoords = camera.worldToVector(worldPoint, Tcn);
        return camera.isVectorWithinFOV(cameraCoords);
    };

    // Draw the edges of the box
    for (const auto &[startIdx, endIdx, color] : edges)
    {
        cv::Vec3d startVertex = vertices[startIdx];
        cv::Vec3d endVertex = vertices[endIdx];
        cv::Point2f prevImagePoint;
        bool prevPointValid = false;

        for (int i = 0; i <= numSegmentsPerEdge; ++i)
        {
            double t = static_cast<double>(i) / numSegmentsPerEdge;
            cv::Vec3d worldPoint = startVertex * (1.0 - t) + endVertex * t;

            if (isWithinFOV(worldPoint)) {
                try {
                    cv::Vec2d imagePoint = camera.worldToPixel(worldPoint, Tcn);
                    cv::Point2f currentImagePoint(imagePoint[0], imagePoint[1]);

                    if (prevPointValid) {
                        // Determine line thickness based on color
                        int thickness = (color == cv::Scalar(0, 0, 255)) ? 7 : 
                                        (color == cv::Scalar(0, 255, 0)) ? 10 : 2;

                        cv::line(image, prevImagePoint, currentImagePoint, color, thickness, cv::LINE_AA);
                    }

                    prevImagePoint = currentImagePoint;
                    prevPointValid = true;
                }
                catch (const std::exception& e) {
                    std::cerr << "Error projecting point: " << e.what() << std::endl;
                }
            } else {
                prevPointValid = false;
            }
        }
    }

    // Draw circles at the base vertices for debugging
    for (int i = 0; i < 4; ++i) {
        if (isWithinFOV(vertices[i])) {
            cv::Vec2d imagePoint = camera.worldToPixel(vertices[i], Tcn);
            cv::circle(image, cv::Point(imagePoint[0], imagePoint[1]), 5, cv::Scalar(255, 0, 255), -1);
        }
    }
}

void ChessboardImage::recoverPose(const Chessboard & chessboard, const Camera & camera)
{
    std::vector<cv::Point3f> rPNn_all = chessboard.gridPoints();

    cv::Mat Thetacn, rNCc;
    cv::solvePnP(rPNn_all, corners, camera.cameraMatrix, camera.distCoeffs, Thetacn, rNCc);

    Pose<double> Tcn(Thetacn, rNCc);
    Tnc = Tcn.inverse();
}

ChessboardData::ChessboardData(const std::filesystem::path & configPath)
{
    // Ensure the config file exists
    assert(std::filesystem::exists(configPath));

    // Open the config file
    cv::FileStorage fs(configPath.string(), cv::FileStorage::READ);
    assert(fs.isOpened());

    // Read chessboard configuration
    cv::FileNode node = fs["chessboard_data"];
    node["chessboard"] >> chessboard;
    std::cout << "Chessboard: " << chessboard << std::endl;

    // Read file pattern for chessboard images
    std::string pattern;
    node["file_regex"] >> pattern;
    fs.release();

    // Create regex object from pattern
    std::regex re(pattern, std::regex_constants::basic | std::regex_constants::icase);
    
    // Get the directory containing the config file
    std::filesystem::path root = configPath.parent_path();
    std::cout << "Scanning directory " << root.string() << " for file pattern \"" << pattern << "\"" << std::endl;

    // Populate chessboard images from regex
    chessboardImages.clear();
    if (std::filesystem::exists(root) && std::filesystem::is_directory(root))
    {
        // Iterate through all files in the directory and its subdirectories
        for (const auto & p : std::filesystem::recursive_directory_iterator(root))
        {
            if (std::filesystem::is_regular_file(p))
            {
                // Check if the file matches the regex pattern
                if (std::regex_match(p.path().filename().string(), re))
                {
                    std::cout << "Loading " << p.path().filename().string() << "..." << std::flush;

                    // Try to load the file as an image
                    cv::Mat image = cv::imread(p.path().string(), cv::IMREAD_COLOR);

                    bool isImage = !image.empty();
                    if (isImage)
                    {
                        // If it's an image, detect chessboard
                        std::cout << " done, detecting chessboard..." << std::flush;
                        ChessboardImage ci(image, chessboard, p.path().filename());
                        std::cout << (ci.isFound ? " found" : " not found") << std::endl;
                        if (ci.isFound)
                        {
                            chessboardImages.push_back(ci);
                        }
                    }
                    else
                    {
                        // If it's not an image, try to load it as a video
                        cv::VideoCapture cap(p.path().string());
                        bool isVideo = cap.isOpened();
                        if (isVideo)
                        {
                            // Get number of video frames
                            int nFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT)); // TODO
                            std::cout << " done, found " << nFrames << " frames" << std::endl;

                            // Loop through selected frames
                            for (int idxFrame = 0; idxFrame < nFrames; idxFrame += std::max(1, nFrames / 10))
                            {
                                // Read frame
                                std::cout << "Reading " << p.path().filename().string() << " frame " << idxFrame << "..." << std::flush;
                                cv::Mat frame;
                                // TODO
                                cap.set(cv::CAP_PROP_POS_FRAMES, idxFrame);
                                cap.read(frame);
                                if (frame.empty())
                                {
                                    std::cout << " end of file found" << std::endl;
                                    break;
                                }

                                // Detect chessboard in frame
                                std::cout << " done, detecting chessboard..." << std::flush;
                                std::string baseName = p.path().stem().string();
                                std::string frameFilename = std::format("{}_{:05d}.jpg", baseName, idxFrame);
                                ChessboardImage ci(frame, chessboard, frameFilename);
                                std::cout << (ci.isFound ? " found" : " not found") << std::endl;
                                if (ci.isFound)
                                {
                                    chessboardImages.push_back(ci);
                                }
                            }
                            cap.release();
                        }
                    }
                }
            }
        }
    }
}

void ChessboardData::drawCorners()
{
    for (auto & chessboardImage : chessboardImages)
    {
        chessboardImage.drawCorners(chessboard);
    }
}

void ChessboardData::drawBoxes(const Camera & camera)
{
    for (auto & chessboardImage : chessboardImages)
    {
        chessboardImage.drawBox(chessboard, camera);
    }
}

void ChessboardData::recoverPoses(const Camera & camera)
{
    for (auto & chessboardImage : chessboardImages)
    {
        chessboardImage.recoverPose(chessboard, camera);
    }
}

void Camera::calibrate(ChessboardData & chessboardData)
{
    std::vector<cv::Point3f> rPNn_all = chessboardData.chessboard.gridPoints();

    std::vector<std::vector<cv::Point2f>> rQOi_all;
    std::vector<std::vector<cv::Point3f>> rPNn_allImages;
    std::vector<ChessboardImage*> validChessboardImages;

    for (auto & chessboardImage : chessboardData.chessboardImages)
    {
        if (chessboardImage.isFound)
        {
            rQOi_all.push_back(chessboardImage.corners);
            rPNn_allImages.push_back(rPNn_all);
            validChessboardImages.push_back(&chessboardImage);
        }
    }
    
    if (rQOi_all.empty()) {
        throw std::runtime_error("No valid chessboard images found for calibration");
    }

    imageSize = validChessboardImages[0]->image.size();
    
    flags = cv::CALIB_RATIONAL_MODEL | cv::CALIB_THIN_PRISM_MODEL;

    // Find intrinsic and extrinsic camera parameters
    cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    distCoeffs = cv::Mat::zeros(12, 1, CV_64F);
    std::vector<cv::Mat> Thetacn_all, rNCc_all;
    
    std::cout << "Calibrating camera... " << std::flush;
    // TODO: Merge from Lab 3
    double rms = cv::calibrateCamera(rPNn_allImages, rQOi_all, imageSize, cameraMatrix, distCoeffs, 
                                     Thetacn_all, rNCc_all, flags);
    
    std::cout << " done" << std::endl;
    
    // Calculate horizontal, vertical and diagonal field of view
    std::cout << "done" << std::endl;
    
    // Pre-compute constants used in isVectorWithinFOV
    calcFieldOfView();

    // Write extrinsic camera parameters for each chessboard image
    assert(chessboardData.chessboardImages.size() == rNCc_all.size());
    assert(chessboardData.chessboardImages.size() == Thetacn_all.size());
    for (std::size_t k = 0; k < chessboardData.chessboardImages.size(); ++k)
    {
        // Set the camera orientation and position (extrinsic camera parameters)
        Pose<double> & Tnc = chessboardData.chessboardImages[k].Tnc;
        // TODO: Merge from Lab 3
        cv::Mat R;
        cv::Rodrigues(Thetacn_all[k], R);
        // Tnc.rotationMatrix = cv::Matx33d(R); FROM LAB 3 
        // Tnc.translationVector = cv::Vec3d(rNCc_all[k]);
        // Convert OpenCV rotation matrix to Eigen
        Eigen::Matrix3d R_eigen;
        cv::cv2eigen(R, R_eigen);

        // Convert OpenCV translation vector to Eigen
        Eigen::Vector3d t_eigen;
        cv::cv2eigen(rNCc_all[k], t_eigen);

        // Assign to Pose<double>
        Tnc = Pose<double>(R_eigen, t_eigen);
    }
    
    printCalibration();
    std::cout << std::setw(30) << "RMS reprojection error: " << rms << std::endl;

    assert(cv::checkRange(cameraMatrix));
    assert(cv::checkRange(distCoeffs));
}

void Camera::printCalibration() const
{
    std::bitset<8*sizeof(flags)> bitflag(flags);
    std::cout << std::endl << "Calibration data:" << std::endl;
    std::cout << std::setw(30) << "Bit flags: " << bitflag << std::endl;
    std::cout << std::setw(30) << "cameraMatrix:\n" << cameraMatrix << std::endl;
    std::cout << std::setw(30) << "distCoeffs:\n" << distCoeffs.t() << std::endl;
    std::cout << std::setw(30) << "Focal lengths: " 
              << "(fx, fy) = "
              << "("<< cameraMatrix.at<double>(0, 0) << ", "<< cameraMatrix.at<double>(1, 1) << ")"
              << std::endl;       
    std::cout << std::setw(30) << "Principal point: " 
              << "(cx, cy) = "
              << "("<< cameraMatrix.at<double>(0, 2) << ", "<< cameraMatrix.at<double>(1, 2) << ")"
              << std::endl;     
    // std::cout << std::setw(30) << "Field of view (horizontal): " << 180.0/CV_PI*hFOV << " deg" << std::endl; // Origin print statements 
    // std::cout << std::setw(30) << "Field of view (vertical): " << 180.0/CV_PI*vFOV << " deg" << std::endl;
    // std::cout << std::setw(30) << "Field of view (diagonal): " << 180.0/CV_PI*dFOV << " deg" << std::endl;
    std::cout << std::setw(30) << "Field of view (horizontal): " << hFOV << " deg" << std::endl;
    std::cout << std::setw(30) << "Field of view (vertical): " << vFOV << " deg" << std::endl; 
    std::cout << std::setw(30) << "Field of view (diagonal): " << dFOV << " deg" << std::endl;
}

void Camera::calcFieldOfView()
{
    assert(cameraMatrix.rows == 3);
    assert(cameraMatrix.cols == 3);
    assert(cameraMatrix.type() == CV_64F);

    // Calculate horizontal FOV
    cv::Vec3d leftVector = pixelToVector(cv::Vec2d(0, imageSize.height / 2));
    cv::Vec3d rightVector = pixelToVector(cv::Vec2d(imageSize.width - 1, imageSize.height / 2));
    hFOV = std::acos(leftVector.dot(rightVector));

    // Calculate vertical FOV
    cv::Vec3d topVector = pixelToVector(cv::Vec2d(imageSize.width / 2, 0));
    cv::Vec3d bottomVector = pixelToVector(cv::Vec2d(imageSize.width / 2, imageSize.height - 1));
    vFOV = std::acos(topVector.dot(bottomVector));

    // Calculate diagonal FOV
    cv::Vec3d topLeftVector = pixelToVector(cv::Vec2d(0, 0));
    cv::Vec3d bottomRightVector = pixelToVector(cv::Vec2d(imageSize.width - 1, imageSize.height - 1));
    dFOV = std::acos(topLeftVector.dot(bottomRightVector));

    // Convert radians to degrees
    hFOV = hFOV * 180.0 / CV_PI;
    vFOV = vFOV * 180.0 / CV_PI;
    dFOV = dFOV * 180.0 / CV_PI;
}

cv::Vec3d Camera::worldToVector(const cv::Vec3d & rPNn, const Pose<double> & Tnb) const
{
    // Camera pose Tnc (i.e., Rnc, rCNn)
    Pose Tnc = bodyToCamera(Tnb); // Tnb*Tbc

    Eigen::Vector3d rPNn_eigen(rPNn[0], rPNn[1], rPNn[2]);

    // Compute the vector from the camera to the point in camera coordinates
    Eigen::Vector3d rPCc_eigen = Tnc.rotationMatrix.transpose() * (rPNn_eigen - Tnc.translationVector);

    // Normalize the resulting vector to ensure it represents only direction (unit vector)
    Eigen::Vector3d uPCc_eigen = rPCc_eigen.normalized();

    // Convert back to cv::Vec3d
    return cv::Vec3d(uPCc_eigen[0], uPCc_eigen[1], uPCc_eigen[2]);
}

cv::Vec2d Camera::worldToPixel(const cv::Vec3d & rPNn, const Pose<double> & Tnb) const
{
    return vectorToPixel(worldToVector(rPNn, Tnb));
}

cv::Vec2d Camera::vectorToPixel(const cv::Vec3d & rPCc) const
{
    // Normalize the input vector
    cv::Vec3d uPCc = rPCc / cv::norm(rPCc);

    // Project the point
    std::vector<cv::Point3f> objectPoints = {cv::Point3f(uPCc[0], uPCc[1], uPCc[2])};
    std::vector<cv::Point2f> imagePoints;

    cv::projectPoints(objectPoints, cv::Vec3d::zeros(), cv::Vec3d::zeros(), 
                      cameraMatrix, distCoeffs, imagePoints);

    // Return the pixel coordinates
    return cv::Vec2d(imagePoints[0].x, imagePoints[0].y);
}

Eigen::Vector2d Camera::vectorToPixel(const Eigen::Vector3d & rPCc, Eigen::Matrix23d & J) const
{
    Eigen::Vector2d rQOi;
    // TODO: Lab 7 (optional)
    return rQOi;
}

cv::Vec3d Camera::pixelToVector(const cv::Vec2d & rQOi) const
{
    // Compute unit vector (uPCc) for the given pixel location (rQOi)
    std::vector<cv::Point2f> imagePoints(1, cv::Point2f(rQOi[0], rQOi[1]));
    std::vector<cv::Point2f> normalizedPoints;

    // TODO
    // Undistort and normalize the point
    cv::undistortPoints(imagePoints, normalizedPoints, Camera::cameraMatrix, Camera::distCoeffs);

    // The normalized point is now [x, y, 1] in camera coordinates
    cv::Vec3d uPCc(normalizedPoints[0].x, normalizedPoints[0].y, 1.0);
    
    // Normalize to unit vector
    return uPCc / cv::norm(uPCc);
}

bool Camera::isVectorWithinFOV(const cv::Vec3d & rPCc) const
{
    // Normalize the input vector
    cv::Vec3d uPCc = rPCc / cv::norm(rPCc);

    // Principal direction vector (typically [0, 0, 1] for a camera looking along its z-axis)
    cv::Vec3d principalDirection(0, 0, 1);

    // Compute horizontal angle
    cv::Vec3d horizontalProjection(uPCc[0], 0, uPCc[2]);
    horizontalProjection = horizontalProjection / cv::norm(horizontalProjection);
    double horizontalAngle = std::acos(horizontalProjection.dot(principalDirection));

    // Compute vertical angle
    cv::Vec3d verticalProjection(0, uPCc[1], uPCc[2]);
    verticalProjection = verticalProjection / cv::norm(verticalProjection);
    double verticalAngle = std::acos(verticalProjection.dot(principalDirection));

    // Convert angles to degrees
    double horizontalAngleDeg = horizontalAngle * 180.0 / CV_PI;
    double verticalAngleDeg = verticalAngle * 180.0 / CV_PI;

    // Project the vector onto the image plane
    cv::Vec2d pixel = vectorToPixel(uPCc);

    // Check if the projected point is within the image bounds
    bool insideImage = (pixel[0] >= 0 && pixel[0] < imageSize.width &&
                        pixel[1] >= 0 && pixel[1] < imageSize.height);

    // Check if both angles are within their respective FOV limits
    bool insideHorizontalFOV = horizontalAngleDeg <= hFOV / 2.0;
    bool insideVerticalFOV = verticalAngleDeg <= vFOV / 2.0;

    return insideImage && insideHorizontalFOV && insideVerticalFOV;
}

bool Camera::isWorldWithinFOV(const cv::Vec3d & rPNn, const Pose<double> & Tnb) const
{
    return isVectorWithinFOV(worldToVector(rPNn, Tnb));
}

void Camera::write(cv::FileStorage & fs) const
{
    fs << "{"
       << "camera_matrix"           << cameraMatrix
       << "distortion_coefficients" << distCoeffs
       << "flags"                   << flags
       << "imageSize"               << imageSize
       << "}";
}

void Camera::read(const cv::FileNode & node)
{
    node["camera_matrix"]           >> cameraMatrix;
    node["distortion_coefficients"] >> distCoeffs;
    node["flags"]                   >> flags;
    node["imageSize"]               >> imageSize;

    // Pre-compute constants used in isVectorWithinFOV
    calcFieldOfView();

    assert(cameraMatrix.cols == 3);
    assert(cameraMatrix.rows == 3);
    assert(cameraMatrix.type() == CV_64F);
    assert(distCoeffs.cols == 1);
    assert(distCoeffs.type() == CV_64F);
}

