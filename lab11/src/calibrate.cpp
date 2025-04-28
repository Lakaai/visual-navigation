#include <filesystem>
#include <opencv2/highgui.hpp>
#include "Camera.h"
#include "calibrate.h"
#include <filesystem>

void calibrateCamera(const std::filesystem::path & configPath, bool exportImages, 
                     const std::filesystem::path & outputDirectory)
{
    // Read chessboard data using configuration file
    ChessboardData chessboardData(configPath);

    // Calibrate camera from chessboard data
    Camera cam;
    cam.calibrate(chessboardData);

    // Write camera calibration to file
    std::filesystem::path cameraPath = configPath.parent_path() / "camera.xml";
    cv::FileStorage fs(cameraPath.string(), cv::FileStorage::WRITE);
    fs << "camera" << cam;
    fs.release();

    // Export calibration images if requested
    if (exportImages)
    {
        for (const auto & chessboardImage : chessboardData.chessboardImages)
        {
            std::filesystem::path outputPath = outputDirectory / chessboardImage.filename;
            cv::imwrite(outputPath.string(), chessboardImage.image);
            std::cout << "Exported calibration image: " << outputPath.string() << std::endl;
        }
    }

    // Visualise the camera calibration results
    chessboardData.drawBoxes(cam);
    for (const auto & chessboardImage : chessboardData.chessboardImages)
    {
        cv::imshow("Calibration images", chessboardImage.image);
        char c = static_cast<char>(cv::waitKey(0));
        if (c == 27 || c == 'q' || c == 'Q') // ESC, q or Q to quit, any other key to continue
            break;
    }
}