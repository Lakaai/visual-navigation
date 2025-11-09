#include <filesystem>
#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "Camera.h"
#include "association_demo.h"

int main(int argc, char* argv [])
{
    const cv::String keys =
        // Argument names | defaults | help message
        "{help h usage ?  |        | print this message}"
        "{@config         | <none> | path to configuration XML}"
        "{export e        |        | export files to the ./out/ directory}"
        ;

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("MCHA4400 Lab 8");
    
    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    // Parse input arguments
    bool hasExport = parser.has("export");
    std::filesystem::path configPath = parser.get<std::string>("@config");

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

    // Check if configuration exists
    if (!std::filesystem::exists(configPath))
    {
        std::cout << "File: " << configPath << " does not exist" << std::endl;
        return EXIT_FAILURE;
    }

    // Read chessboard data using configuration file
    ChessboardData chessboardData(configPath);

    // Camera calibration file
    std::filesystem::path cameraPath = configPath.parent_path() / "camera.xml";

    Camera camera;
    assert(std::filesystem::exists(cameraPath));
    cv::FileStorage fs(cameraPath.string(), cv::FileStorage::READ);
    assert(fs.isOpened());
    fs["camera"] >> camera;

    // Reconstruct extrinsic parameters (camera pose) for each chessboard image
    chessboardData.recoverPoses(camera);

    // ------------------------------------------------------------
    // Run geometric matcher demo
    // ------------------------------------------------------------
    for (const auto & chessboardImage : chessboardData.chessboardImages)
    {
        cv::Mat img = associationDemo(camera, chessboardImage);

        if (hasExport)
        {
            std::string outputFilename = chessboardImage.filename.stem().string()
                                       + "_out"
                                       + chessboardImage.filename.extension().string();
            std::filesystem::path outputPath = outputDirectory / outputFilename;
            cv::imwrite(outputPath.string(), img);
        }
        else
        {
            const double resize_scale = 0.5;
            cv::Mat resized_img;
            cv::resize(img, resized_img, cv::Size(img.cols*resize_scale, img.rows*resize_scale), cv::INTER_LINEAR);
            cv::imshow("Data association demo (press ESC, q or Q to quit)", resized_img);
            char c = static_cast<char>(cv::waitKey(0));
            if (c == 27 || c == 'q' || c == 'Q') // ESC, q or Q to quit, any other key to continue
                break;
        }
    }

    return EXIT_SUCCESS;
}
