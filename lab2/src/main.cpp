#include <set>
#include <cstdlib>
#include <iostream>
#include <string>  
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <nanobench.h>
#include "imagefeatures.h"

int main(int argc, char *argv[])
{
    cv::String keys = 
        // Argument names | defaults | help message
        "{help h usage ?  |          | print this message}"
        "{@input          | <none>   | input can be a path to an image or video (e.g., ../data/lab.jpg)}"
        "{export e        |          | export output file to the ./out/ directory}"
        "{N               | 10       | maximum number of features to find}"
        "{detector d      | fast     | feature detector to use (e.g., harris, shi, aruco, fast)}"
        "{benchmark b     |          | run benchmark for all detectors}"
    ;
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("MCHA4400 Lab 2");

    if (parser.has("help"))
    {
        parser.printMessage();
        return EXIT_SUCCESS;
    }

    // Parse input arguments
    bool doExport = parser.has("export");
    int maxNumFeatures = parser.get<int>("N");
    bool doBenchmark = parser.has("benchmark");
    cv::String detector = parser.get<std::string>("detector");
    std::filesystem::path inputPath = parser.get<std::string>("@input");

    // Check for syntax errors
    if (!parser.check())
    {
        parser.printMessage();
        parser.printErrors();
        return EXIT_FAILURE;
    }

    if (!std::filesystem::exists(inputPath))
    {
        std::cout << "File: " << inputPath.string() << " does not exist" << std::endl;
        return EXIT_FAILURE;
    }

    // Prepare output directory
    std::filesystem::path outputDirectory;
    if (doExport)
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

    // Prepare output file path
    std::filesystem::path outputPath;
    if (doExport)
    {
        std::string outputFilename = inputPath.stem().string()
                                   + "_"
                                   + detector
                                   + inputPath.extension().string();
        outputPath = outputDirectory / outputFilename;
        std::cout << "Output name: " << outputPath.string() << std::endl;
    }

    // Check if input is an image or video (or neither)
    bool isVideo = false; // TODO
    bool isImage = false; // TODO

    // Get the file extension
    std::string extension = inputPath.extension().string();

    // Convert extension to lowercase
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    // List of supported image and video extensions
    std::set<std::string> imageExtensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"};
    std::set<std::string> videoExtensions = {".mp4", ".avi", ".mov", ".mkv"};

    // Check extension against known image and video formats
    if (imageExtensions.find(extension) != imageExtensions.end())
    {
        // Try to open the file as an image
        cv::Mat image = cv::imread(inputPath.string());
        if (!image.empty())
        {
            isImage = true;
        }
        else
        {
            std::cout << "File appears to be an image but could not be opened." << std::endl;
        }
    }
    else if (videoExtensions.find(extension) != videoExtensions.end())
    {
        // Try to open the file as a video
        cv::VideoCapture video(inputPath.string());
        if (video.isOpened())
        {
            isVideo = true;
        }
        else
        {
            std::cout << "File appears to be a video but could not be opened." << std::endl;
        }
    }
    else
    {
        std::cout << "Unknown file type for extension: " << extension << std::endl;
    }

    // Output the results
    if (isImage)
    {
        std::cout << "Input file is an image." << std::endl;
    }
    else if (isVideo)
    {
        std::cout << "Input file is a video." << std::endl;
    }

    if (!isImage && !isVideo)
    {
        std::cout << "Could not read file: " << inputPath.string() << std::endl;
        return EXIT_FAILURE;
    }

    if (doBenchmark)
    {
        if (!isImage)
        {
            std::cout << "Benchmark can only be run on images, not videos." << std::endl;
            return EXIT_FAILURE;
        }

        // Suppress cout during benchmark
        std::streambuf* old_buf = std::cout.rdbuf(nullptr);

        // Create benchmark object
        ankerl::nanobench::Bench bench;
        
        // Enable relative comparison
        bench.relative(true).minEpochIterations(109);

        // Capture benchmark output in a separate stringstream
        std::stringstream bench_output;
        bench.output(&bench_output);

        // Load the image
        cv::Mat img = cv::imread(inputPath.string(), cv::IMREAD_COLOR);
        if (img.empty()) {
        std::cout << "Could not open or find the image." << std::endl;
        return EXIT_FAILURE;
        }   

        // TODO: Run the benchmarks for the 4 feature detectors
        // Define the feature detection functions
        auto detectHarris = [&]() { detectAndDrawHarris(img, maxNumFeatures); };
        auto detectShi = [&]() { detectAndDrawShiAndTomasi(img, maxNumFeatures); };
        auto detectArUco = [&]() { detectAndDrawArUco(img, maxNumFeatures); };
        auto detectFAST = [&]() { detectAndDrawFAST(img, maxNumFeatures); };

        // Run the benchmarks for the 4 feature detectors
        bench.run("Harris Detector", detectHarris);
        bench.run("Shi-Tomasi Detector", detectShi);
        bench.run("ArUco Detector", detectArUco);
        bench.run("FAST Detector", detectFAST);
        
        // Restore cout
        std::cout.rdbuf(old_buf);
        
        // Print benchmark results
        std::cout << "\nBenchmark results:" << std::endl;
        std::cout << bench_output.str();

        return EXIT_SUCCESS;
    }

    if (isImage)
    {
        // TODO: Call one of the detectAndDraw functions from imagefeatures.cpp according to the detector option specified at the command line
        std::cout << "isImage = true" << std::endl;
    
        // Load the image
        cv::Mat image = cv::imread(inputPath.string());
        if (image.empty()) {
            std::cout << "Could not open or find the image." << std::endl;
            return EXIT_FAILURE;
        }

        // Call the feature detection function according to the detector option specified at the command line
        std::cout << "Attempting feature detection with detection algorithm: " << detector << std::endl;

        cv::Mat outputImage;
        if (detector == "harris") {
            outputImage = detectAndDrawHarris(image, maxNumFeatures);

        } else if (detector == "shi") {
            outputImage = detectAndDrawShiAndTomasi(image, maxNumFeatures);           

        } else if (detector == "aruco") {
            outputImage = detectAndDrawArUco(image, maxNumFeatures);
            
        } else if (detector == "fast") {
            outputImage = detectAndDrawFAST(image, maxNumFeatures);

        } else {
            std::cout << "Unknown detector: " << detector << std::endl;
            return EXIT_FAILURE;
        }
        
        if (doExport)
        {
            // TODO: Write image returned from detectAndDraw to outputPath
            cv::imwrite(outputPath.string(), outputImage);
            std::cout << "Exported image to: " << outputPath.string() << std::endl;
        }
        else
        {
            // TODO: Display image returned from detectAndDraw on screen and wait for keypress
            // Display original and processed images side by side
            cv::Mat combinedImage;
            cv::hconcat(image, outputImage, combinedImage); // Concatenate horizontally

            cv::imshow("Detected Features", outputImage);
            cv::waitKey(0);
        }
    }

    if (isVideo)
    {
        cv::VideoCapture video(inputPath.string());
        if (!video.isOpened())
        {
            std::cerr << "Error: Could not open the input video." << std::endl;
            return EXIT_FAILURE;
        }

        double fps = video.get(cv::CAP_PROP_FPS);
        cv::Size frameSize(
            static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH)),
            static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT))
        );

        cv::VideoWriter outputVideo;
        if (doExport)
        {
            int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // Codec for MP4
            outputVideo.open(outputPath.string(), codec, fps, frameSize, true);
            if (!outputVideo.isOpened())
            {
                std::cerr << "Error: Could not open the output video for writing." << std::endl;
                return EXIT_FAILURE;
            }

            std::cout << "Exporting video to " << outputPath.string() << std::endl;
        }

        cv::Mat frame;
        while (true)
        {
            // Get next frame from input video
            bool success = video.read(frame);
            std::cout << "Frame has been read " << std::endl;
            // If frame is empty, break out of the while loop
            if (!success)
                break;

            // Call one of the detectAndDraw functions from imagefeatures.cpp according to the detector option specified at the command line
            cv::Mat processedFrame = detectAndDrawHarris(frame, maxNumFeatures); // Assuming detectAndDrawHarris is the chosen function

            if (doExport)
            {
                // Write image returned from detectAndDraw to frame of output video
                outputVideo.write(processedFrame);
                // Display image returned from detectAndDraw on screen and wait for 1000/fps milliseconds
                cv::imshow("Processed Video", processedFrame);
            }
            else
            {
                // Display image returned from detectAndDraw on screen and wait for 1000/fps milliseconds
                cv::imshow("Processed Video", processedFrame);
                cv::waitKey(1000 / fps);
            }
        }

        // Release the input video object
        video.release();

        if (doExport)
        {
            // Release the output video object
            outputVideo.release();
            std::cout << "Video export completed." << std::endl;
        }
    }
    

    return EXIT_SUCCESS;
}

