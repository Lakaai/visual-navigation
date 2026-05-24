#include <cstdlib>
#include <string>  
#include <filesystem>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
// #include <opencv2/core/utility.hpp>
#include "Camera.h"
#include "visual_odometry.h"
#include "visualNavigation.h"


int main(int argc, char* argv [])
{
    const cv::String keys =
        // Argument names | defaults | help message
        "{help h usage ?  |          | print this help message}"
        "{@input          | <none>   | either configuration XML (with --calibrate) or input video path}"
        "{calibrate c     |          | perform camera calibration for given configuration XML}"
        "{export e        |          | export video}"
        "{scenario s      | 4        | run visual navigation on input video with scenario type (4:flight, 5:indoor, 6:duck)}"
        "{interactive i   | 0        | interactivity (0:none, 1:last frame, 2:all frames)}";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("MCHA4400 Assignment 2");

    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    // Parse input arguments
    int scenario = parser.get<int>("scenario");
    int interactive = parser.get<int>("interactive");

    bool hasExport = parser.has("export");
    bool hasCalibrate = parser.has("calibrate");

    std::filesystem::path inputPath = parser.get<std::string>("@input");

    // Check for syntax errors
    if (!parser.check())
    {
        parser.printMessage();
        parser.printErrors();
        return EXIT_FAILURE;
    }

    // Prepare output directory
    std::filesystem::path outputDirectory;
    if (hasExport)
    {
        std::filesystem::path appPath = parser.getPathToApplication();
        outputDirectory = appPath / ".." / "out";

        // Create output directory if we need to
        if (!std::filesystem::exists(outputDirectory))
        {
            std::cout << "Creating directory " << outputDirectory.string() << std::endl;
            std::filesystem::create_directory(outputDirectory);
        }
        std::cout << "Output directory set to " << outputDirectory.string() << std::endl;
    }

    // Check if input path exists
    if (!std::filesystem::exists(inputPath))
    {
        std::cout << "File: " << inputPath << " does not exist" << std::endl;
        return EXIT_FAILURE;
    }

    std::filesystem::path cameraPath = inputPath.parent_path() / "camera.xml";

    Camera cam;
    if (hasCalibrate)
    {
        std::cout << "Calibrating camera..." << std::endl;

        // Read chessboard data using configuration file
        ChessboardData chessboardData(inputPath);

        // Calibrate camera from chessboard data
        cam.calibrate(chessboardData);
        
        // Write camera calibration to file
        cv::FileStorage fs(cameraPath.string(), cv::FileStorage::WRITE);
        fs << "camera" << cam;
        fs.release();

        // Visualise calibration results
        chessboardData.drawBoxes(cam);

        for (const auto & chessboardImage : chessboardData.chessboardImages)
        {
            if (hasExport)
            {
                std::filesystem::path outputPath = outputDirectory / chessboardImage.filename;
                cv::imwrite(outputPath.string(), chessboardImage.image);
            }
            else
            {
                cv::imshow("Calibration images (press ESC to quit, any other key to continue)", chessboardImage.image);
                char c = static_cast<char>(cv::waitKey(0));
                if (c == 27) // ESC to quit, any other key to continue
                    break;
            }
        }
    }
    else
    {
         // Load calibration
        if (!std::filesystem::exists(cameraPath))
        {
            std::cout << "File: " << cameraPath << " does not exist" << std::endl;
            return EXIT_FAILURE;
        }

        cv::FileStorage fs(cameraPath.string(), cv::FileStorage::READ);
        assert(fs.isOpened());
        fs["camera"] >> cam;

        // Display loaded calibration data
        cam.printCalibration();


        assert(4 <= scenario && scenario <= 6);
        assert(0 <= interactive && interactive <= 2);
        std::cout << "Input video: " << inputPath.string() << std::endl;
        std::cout << "Running visual odometry" << std::endl;
        std::filesystem::path cameraPath = inputPath.parent_path() / "camera.xml";
        
        if (scenario == 4)
        {
            std::cout << "Running scenario 4: Outdoor Visual Navigation" << std::endl;
            runVisualOdometryFromVideo(inputPath, cameraPath, outputDirectory);
        }
        else if (scenario == 5)
        {
            std::cout << "Running scenario 5: Indoor Visual Navigation" << std::endl;
            // runVisualNavigationFromVideo(inputPath, cameraPath, scenario, interactive, outputDirectory);
        }
        else if (scenario == 6)
        {
            std::cout << "Running scenario 6: Post-Duck Apocalyptic Visual Navigation" << std::endl;
            // runVisualNavigationFromVideo(inputPath, cameraPath, scenario, interactive, outputDirectory);
        }
        
    }

    return EXIT_SUCCESS;
}
