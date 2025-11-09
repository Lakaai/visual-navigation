#include <cstdlib>
#include <string>  
#include <filesystem>
#include <iostream>
#include <opencv2/core/utility.hpp>
#include "calibrate.h"
#include "image_flow.h"

int main(int argc, char* argv [])
{
    const cv::String keys =
        // Argument names | defaults | help message
        "{help h usage ?  |          | print this help message}"
        "{@input          | <none>   | path to input video or configuration XML}"
        "{calibrate c     |          | perform camera calibration for given configuration XML}"
        "{export e        |          | export video}";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("MCHA4400 Lab 10");
    
    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    // Parse input arguments
    bool hasCalibrate = parser.has("calibrate");
    bool hasExport = parser.has("export");
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

    if (hasCalibrate)
    {
        std::cout << "Calibrating camera" << std::endl;
        std::cout << "Configuration file: " << inputPath.string() << std::endl;
        calibrateCamera(inputPath);
    }
    else
    {
        std::cout << "Input video: " << inputPath.string() << std::endl;
        std::cout << "Running image flow" << std::endl;
        std::filesystem::path cameraPath = inputPath.parent_path() / "camera.xml";
        runImageFlow(inputPath, cameraPath, outputDirectory);
    }

    return EXIT_SUCCESS;
}
