#include <cstdlib>
#include <string>  
#include <filesystem>
#include <iostream>
#include <opencv2/core/utility.hpp>
#include "calibrate.h"
#include "visual_odometry.h"
#include "calibrate.h"

int main(int argc, char* argv [])
{
    const cv::String keys =
        // Argument names | defaults | help message
        "{help h usage ?  |          | print this help message}"
        "{@input          | <none>   | path to input video or configuration XML}"
        "{calibrate c     |          | perform camera calibration for given configuration XML}"
        "{export e        |          | export video}";
        


    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("MCHA4400 Lab 11");

    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    bool hasExport = parser.has("export");
    bool hasCalibrate = parser.has("calibrate");
    std::filesystem::path inputPath = parser.get<std::string>("@input");
    

    if (!parser.check())
    {
        parser.printMessage();
        parser.printErrors();
        return EXIT_FAILURE;
    }

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
    }

    if (hasCalibrate)
    {
        std::cout << "Calibrating camera" << std::endl;
        std::cout << "Configuration file: " << inputPath.string() << std::endl;
        calibrateCamera(inputPath, hasExport, outputDirectory);
    }

    else
    {
        std::cout << "Input video: " << inputPath.string() << std::endl;
        std::cout << "Running visual odometry" << std::endl;
        std::filesystem::path cameraPath = inputPath.parent_path() / "camera.xml";
        runVisualOdometryFromVideo(inputPath, cameraPath, outputDirectory);
    }

    return EXIT_SUCCESS;
}
