
#include <filesystem>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "Camera.h"
#include "calibrate.h"

void calibrateCamera(const std::filesystem::path & configPath)
{
    // Check if the config file exists
    if (!std::filesystem::exists(configPath))
    {
        throw std::runtime_error("Config file does not exist: " + configPath.string());
    }

    // Read XML configuration file
    cv::FileStorage fs(configPath.string(), cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        throw std::runtime_error("Failed to open config file: " + configPath.string());
    }

    // Parse XML and extract chessboard data
    ChessboardData chessboardData(configPath);

    // Create Camera object
    Camera camera;

    // Perform camera calibration
    camera.calibrate(chessboardData);

    // Write the camera matrix and lens distortion parameters to camera.xml file
    std::filesystem::path cameraPath = configPath.parent_path() / "camera.xml";
    cv::FileStorage fs_out(cameraPath.string(), cv::FileStorage::WRITE);
    if (!fs_out.isOpened())
    {
        throw std::runtime_error("Failed to open output file: " + cameraPath.string());
    }

    fs_out << "camera" << camera;
    fs_out.release();

    // Print calibration results
    std::cout << "Camera calibration completed." << std::endl;
    camera.printCalibration();

    // Visualize the camera calibration results (optional)
    if (!chessboardData.chessboardImages.empty())
    {
        for (auto& image : chessboardData.chessboardImages)
        {
            image.drawCorners(chessboardData.chessboard);
            image.drawBox(chessboardData.chessboard, camera);
            
            cv::imshow("Calibration Result", image.image);
            char key = cv::waitKey(0);
            if (key == 27) // ESC key
                break;
        }
        cv::destroyAllWindows();
    }
}
