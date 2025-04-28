#include <filesystem>
#include <string>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>
#include "BufferedVideo.h"
#include "Camera.h"
#include "image_flow.h"
#include <Eigen/Core>
#include <Eigen/LU>

void runImageFlow(const std::filesystem::path & videoPath, const std::filesystem::path & cameraPath, const std::filesystem::path & outputDirectory)
{
    assert(!videoPath.empty());
    std::filesystem::path outputPath;
    bool doExport = !outputDirectory.empty();

    // TODO: Lab 10
    int divisor         = 2;     // Image scaling factor
    int imgModulus      = 14;     // Take frames divisible by this number
    int maxNumFeatures  = 1000;   // Maximum number of features per frame
    int minNumFeatures  = 700;   // Minimum number of features per frame

    if (doExport)
    {
        std::string outputFilename = videoPath.stem().string()
                                   + "_"
                                   + std::to_string(divisor)
                                   + "_"
                                   + std::to_string(imgModulus)
                                   + videoPath.extension().string();
        outputPath = outputDirectory / outputFilename;
    }

    // Load camera calibration
    Camera camera;
    assert(std::filesystem::exists(cameraPath));
    cv::FileStorage fs(cameraPath.string(), cv::FileStorage::READ);
    assert(fs.isOpened());
    fs["camera"] >> camera;

    // Display loaded calibration data
    camera.printCalibration();

    // Open input video
    cv::VideoCapture cap(videoPath.string());
    assert(cap.isOpened());
    int nFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    assert(nFrames > 0);

    std::cout << "Input video: " << videoPath.string() << std::endl;
    std::cout << "Total number of frames: " << nFrames << std::endl;
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout   << "Input video frame rate: " << fps << std::endl;
    std::cout   << "Input video dimensions: [" 
                << cap.get(cv::CAP_PROP_FRAME_WIDTH) << " x "
                << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << "]"
                << std::endl;

    std::cout << std::setw(20) << "maxNumFeatures: " << maxNumFeatures << std::endl;
    std::cout << std::setw(20) << "minNumFeatures: " << minNumFeatures << std::endl;
    std::cout << std::setw(20) << "divisor: "  << divisor  << std::endl;
    std::cout << std::setw(20) << "imgModulus: " << imgModulus << std::endl;
    std::cout << std::endl;

    BufferedVideoReader bufferedVideoReader(5);
    bufferedVideoReader.start(cap);

    cv::VideoWriter videoOut;
    BufferedVideoWriter bufferedVideoWriter(3);
    if (doExport)
    {
        double outputFps    = fps/imgModulus;
        cv::Size frameSize;
        frameSize.width     = cap.get(cv::CAP_PROP_FRAME_WIDTH)/divisor;
        frameSize.height    = cap.get(cv::CAP_PROP_FRAME_HEIGHT)/divisor;
        int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // manually specify output video codec
        videoOut.open(outputPath.string(), codec, outputFps, frameSize);
        bufferedVideoWriter.start(videoOut);
    }

    // Declare variables to store the current and previous frames
    cv::Mat frame, imgk, imgkm1;

    // Variables for feature tracking
    std::vector<cv::Point2f> rQOik, rQOikm1;
    std::vector<uchar> status;
    std::vector<float> err;

    // Parameters for goodFeaturesToTrack
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;

        // Loop through all frames in the video
    for (int frameCount = 0; frameCount < nFrames; ++frameCount)
    {   
        // Get next input frame
        cv::Mat frame = bufferedVideoReader.read();
        assert(!frame.empty()); 
        
        // Check if this frame should be processed based on imgModulus
        if (frameCount % imgModulus == 0)
        {
            // a) Extract images and create grayscale copy
            cv::Mat frameScaled = frame;
            // Resize the frame if divisor is not 1
            if (divisor != 1)
            {
                cv::resize(frame, frameScaled, cv::Size(), 1.0/divisor, 1.0/divisor, cv::INTER_AREA);
            }

            // Convert the frame to grayscale
            cv::cvtColor(frameScaled, imgk, cv::COLOR_BGR2GRAY);

            // b) Initialize or reinitialize feature set if needed
            if (rQOik.size() < minNumFeatures)
            {
                std::vector<cv::Point2f> corners;
                cv::goodFeaturesToTrack(imgk, corners, maxNumFeatures, qualityLevel, minDistance / divisor, 
                                        cv::Mat(), blockSize, useHarrisDetector, k);

                // Refine the corner locations
                cv::cornerSubPix(imgk, corners, cv::Size(10, 10), cv::Size(-1, -1),
                                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001));

                // Scale up corners to original image space
                rQOik.clear();
                for (const auto& corner : corners)
                {
                    rQOik.push_back(corner * divisor);
                }
            }

            // c) Calculate optical flow for appropriate frames
            if (!imgkm1.empty())
            {
                std::vector<cv::Point2f> rQOikm1Scaled, rQOikScaled;
                // Scale down previous points for optical flow calculation
                for (const auto& point : rQOikm1)
                {
                    rQOikm1Scaled.push_back(point / divisor);
                }

                // i) Calculate optical flow
                cv::calcOpticalFlowPyrLK(imgkm1, imgk, rQOikm1Scaled, rQOikScaled, status, err);

                // Scale up new points to original image space
                rQOik.clear();
                for (const auto& point : rQOikScaled)
                {
                    rQOik.push_back(point * divisor);
                }

                // ii) Filter features by status
                std::vector<cv::Point2f> goodNew, goodOld;
                for (uint i = 0; i < rQOik.size(); i++)
                {
                    if (status[i] == 1)
                    {
                        goodNew.push_back(rQOik[i]);
                        goodOld.push_back(rQOikm1[i]);
                    }
                }

                 // a) Undistort the points
                Eigen::Matrix<double, 2, Eigen::Dynamic> rQOiNew(2, goodNew.size());
                Eigen::Matrix<double, 2, Eigen::Dynamic> rQOiOld(2, goodOld.size());

                for (size_t i = 0; i < goodNew.size(); ++i)
                {
                    rQOiNew.col(i) << goodNew[i].x, goodNew[i].y;
                    rQOiOld.col(i) << goodOld[i].x, goodOld[i].y;
                }

                Eigen::Matrix<double, 2, Eigen::Dynamic> undistortedNew = camera.undistort(rQOiNew);
                Eigen::Matrix<double, 2, Eigen::Dynamic> undistortedOld = camera.undistort(rQOiOld);

                // b) Calculate fundamental matrix
                std::vector<cv::Point2f> cvUndistortedNew, cvUndistortedOld;
                for (int i = 0; i < undistortedNew.cols(); ++i)
                {
                    cvUndistortedNew.push_back(cv::Point2f(undistortedNew(0, i), undistortedNew(1, i)));
                    cvUndistortedOld.push_back(cv::Point2f(undistortedOld(0, i), undistortedOld(1, i)));
                }

                std::vector<uchar> inlierMask;
                cv::Mat F = cv::findFundamentalMat(cvUndistortedOld, cvUndistortedNew, cv::FM_RANSAC, 1.0, 0.99, inlierMask);


                
                // iii) Display the flow field with inlier/outlier coloring
                cv::Mat flowImage = frameScaled.clone();
                for (uint i = 0; i < goodNew.size(); i++)
                {
                    cv::Point2f pt1 = goodOld[i] / divisor;
                    cv::Point2f pt2 = goodNew[i] / divisor;
                    cv::Scalar color = inlierMask[i] ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
                    cv::arrowedLine(flowImage, pt1, pt2, color, 2, cv::LINE_AA);
                }
                cv::imshow("LK Demo", flowImage);
                cv::waitKey(1);

                rQOik = goodNew;
                rQOikm1 = goodOld;

                // v) Reinitialize features if needed
                if (rQOik.size() < minNumFeatures)
                {
                    std::vector<cv::Point2f> corners;
                    cv::goodFeaturesToTrack(imgk, corners, maxNumFeatures, qualityLevel, minDistance / divisor, 
                                            cv::Mat(), blockSize, useHarrisDetector, k);
                    cv::cornerSubPix(imgk, corners, cv::Size(10, 10), cv::Size(-1, -1),
                                    cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001));

                    // Scale up corners to original image space
                    rQOik.clear();
                    for (const auto& corner : corners)
                    {
                        rQOik.push_back(corner * divisor);
                    }
                }
            }

            // iv) Copy current frame and features to previous
            imgk.copyTo(imgkm1);
            rQOikm1 = rQOik;
        }
    }

    // TODO: Lab 10

        if (doExport)
        {
            bufferedVideoWriter.stop();
        }
        bufferedVideoReader.stop();
}
