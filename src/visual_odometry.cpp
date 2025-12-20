#include <autodiff/forward/dual.hpp>
#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>
#include "BufferedVideo.h"
#include "Pose.hpp"
#include "rotation.hpp"
#include "Camera.h"
#include "DJIVideoCaption.h"
#include "rotation.hpp"
#include "GaussianInfo.hpp"
#include "funcmin.hpp"
#include "SystemVisualNav.h"
#include "MeasurementOutdoorFlowBundle.h"
#include "visual_odometry.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

// Forward declarations
static Eigen::Vector6d getInitialPose(const DJIVideoCaption & caption0);
static void plotGroundPlane(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor);
static void plotHorizon(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor);
static void plotCompass(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor);
static void plotEpipole(cv::Mat & img, const Eigen::Vector6d & etak, const Eigen::Vector6d & etakm1, const Camera & camera, const int & divisor);


void runVisualOdometryFromVideo(const std::filesystem::path & videoPath, const std::filesystem::path & cameraPath, const std::filesystem::path & outputDirectory)
{
    // TODO: Lab 11
    int imgModulus  = 10;                    // Take frames divisible by this number
    int divisor     = 2;                    // Image scaling factor (used for plotting only)

    assert(!videoPath.empty());

    // Subtitle path
    std::filesystem::path subtitlePath = videoPath.parent_path() / (videoPath.stem().string() + ".SRT");
    assert(std::filesystem::exists(subtitlePath));

    // Load and parse subtitle file
    std::vector<DJIVideoCaption> djiVideoCaption = getVideoCaptions(subtitlePath);

    // Output video path
    std::filesystem::path outputPath;
    bool doExport = !outputDirectory.empty();
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

    // Set camera pose w.r.t. body
    // TODO: Lab 11
    Eigen::Matrix3d Rbc;
        Rbc <<  0, 0, 1,
                1, 0, 0,
                0, 1, 0;
    Eigen::Vector3d rBCn = Eigen::Vector3d::Zero();
    camera.Tbc.rotationMatrix = Rbc;
    camera.Tbc.translationVector = rBCn;

    // Open input video
    cv::VideoCapture cap(videoPath.string());
    assert(cap.isOpened());
    int nFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    assert(nFrames > 0);

    std::cout << "Input video: " << videoPath.string() << std::endl;
    std::cout << "Subtitle file: " << subtitlePath.string() << std::endl;
    std::cout << "Total number of frames: " << nFrames << std::endl;
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout   << "Input video frame rate: " << fps << std::endl;
    std::cout   << "Input video dimensions: [" 
                << cap.get(cv::CAP_PROP_FRAME_WIDTH) << " x "
                << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << "]"
                << std::endl;

    BufferedVideoReader bufferedVideoReader(5);
    bufferedVideoReader.start(cap);

    cv::VideoWriter videoOut;
    BufferedVideoWriter bufferedVideoWriter(3);
    if (doExport)
    {
        cv::Size frameSize;
        frameSize.width     = cap.get(cv::CAP_PROP_FRAME_WIDTH)/divisor;
        frameSize.height    = cap.get(cv::CAP_PROP_FRAME_HEIGHT)/divisor;
        double outputFps    = fps/imgModulus;
        // int codec = cap.get(cv::CAP_PROP_FOURCC); // use same output video codec as input video
        int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // manually specify output video codec
        videoOut.open(outputPath.string(), codec, outputFps, frameSize);
        bufferedVideoWriter.start(videoOut);
    }

    // Visual odometry
    auto p0 = GaussianInfo<double>::fromSqrtInfo(Eigen::VectorXd::Zero(18), Eigen::MatrixXd::Zero(18, 18)); // dummy initial distribution
    SystemVisualNav system(p0); // dummy system
    Eigen::VectorXd etakm1(6);
    Eigen::VectorXd etak(6);
    etak = getInitialPose(djiVideoCaption[0]);

    cv::Mat imgk_raw;
    cv::Mat imgkm1_raw;
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOikm1;

    for (int i = 0, k = 0;; ++i)
    {
        // Capture frame by frame
        imgk_raw = bufferedVideoReader.read();
        if (imgk_raw.empty())
        {
            break;
        }

        if (i % imgModulus == 0)
        {
            if (k > 0)
            {
                // Create measurement data from image pair and previous tracked features
                MeasurementOutdoorFlowBundle measurement(i/fps, camera, imgk_raw, imgkm1_raw, rQOikm1);

                rQOikm1 = measurement.trackedPreviousFeatures();
                const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQOik = measurement.trackedCurrentFeatures();

                // Create cost function with prototype V = costFunc(eta, g, H)
                auto costFunc = [&](const Eigen::VectorXd & etak, Eigen::VectorXd & g, Eigen::MatrixXd & H)
                {
                    return measurement.costOdometry(etak, etakm1, g, H);
                };

                // Minimise cost (maximise log likelihood)
                etak = etakm1; // Start optimisation at initial conditions
                const int verbosity = 3; // 0:none, 1:dots, 2:summary, 3:iter
                int ret = funcmin::NewtonTrust(costFunc, etak, verbosity);
                assert(ret == 0);

                Eigen::Vector3d rBNn = etak.head<3>();
                Eigen::Matrix3d Rnb = rpy2rot(etak.tail<3>());

                std::cout << "rBNn: \n" << rBNn << std::endl;
                std::cout << "Rnb: \n" << Rnb << std::endl;

                Eigen::VectorXd x(18);
                x.setZero();
                x.segment<6>(6) = etak;
                x.segment<6>(12) = etakm1;

                // Prepare output image
                cv::Mat imgout;
                cv::resize(imgk_raw, imgout, cv::Size(), 1.0/divisor, 1.0/divisor);

                // Predicted flow field (used for plotting onto original image)
                Eigen::Matrix<double, 2, Eigen::Dynamic> rQOik_hat = measurement.predictedFeatures(x, system);

                // Plotting
                std::vector<cv::Point2d> rQOikm1_scaled, rQOik_scaled, rQOik_hat_scaled;
                int np = rQOik.cols();
                rQOikm1_scaled.resize(np);
                rQOik_scaled.resize(np);
                rQOik_hat_scaled.resize(np);
                for (int j = 0; j < np; ++j)
                {
                    rQOikm1_scaled[j].x     = rQOikm1(0, j)/divisor;
                    rQOikm1_scaled[j].y     = rQOikm1(1, j)/divisor;

                    rQOik_scaled[j].x       = rQOik(0, j)/divisor;
                    rQOik_scaled[j].y       = rQOik(1, j)/divisor;

                    rQOik_hat_scaled[j].x   = rQOik_hat(0, j)/divisor;
                    rQOik_hat_scaled[j].y   = rQOik_hat(1, j)/divisor;
                }

                // Plot flow vectors
                for (int j = 0; j < rQOik.cols(); ++j)
                {
                    // Predicted flow
                    cv::arrowedLine(imgout, rQOikm1_scaled[j], rQOik_hat_scaled[j], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);

                    if (measurement.inlierMask()[j])
                    {
                        // Measured inliers
                        cv::arrowedLine(imgout, rQOikm1_scaled[j], rQOik_scaled[j], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                    }
                    else
                    {
                        // Measured outliers
                        cv::arrowedLine(imgout, rQOikm1_scaled[j], rQOik_scaled[j], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                    }
                }

                plotGroundPlane(imgout, etak, camera, divisor);
                plotHorizon(imgout, etak, camera, divisor);
                plotCompass(imgout, etak, camera, divisor);
                plotEpipole(imgout, etak, etakm1, camera, divisor);

                cv::imshow("Visual odometry demo", imgout);
                char key = cv::waitKey(1);
                if (key == 'q')
                {
                    std::cout << "Key '" << key << "' pressed. Terminating program." << std::endl;
                    break;
                }
                if (doExport)
                {
                    bufferedVideoWriter.write(imgout);
                }
                rQOikm1.resize(2, rQOik.cols());
                rQOikm1 = rQOik;
            }

            imgk_raw.copyTo(imgkm1_raw);
            etakm1 = etak;
            k++;
        }
    } 

    if (doExport)
    {
         bufferedVideoWriter.stop();
    }
    bufferedVideoReader.stop();
}

Eigen::Vector6d getInitialPose(const DJIVideoCaption & caption0)
{
    double h        = caption0.altitude;    // Altitude (GPS) [m]
    double ga       = h - 7;                // Altitude (AGL) [m]

    Eigen::Vector6d eta0;
    // TODO: Lab 11
    eta0 << 0, 0, -ga,          // Initial position 
            -0.015, 0.11, 0;     // Initial orientation 
    return eta0;
}

// Assignement 2
// Eigen::Vector6d getInitialPose(const DJIVideoCaption & caption0)
// {
//     double h        = caption0.altitude;    // Altitude (GPS) [m]
//     double ga       = h - 7;                // Altitude (AGL) [m]

//     Eigen::Vector6d eta0;
//     eta0 << 0, 0, -ga,         // Initial position 
//     -0.009, 0.09, 0;            // Initial orientation 
//     return eta0;
// }

void plotGroundPlane(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor)
{
    Eigen::Matrix3d Rnb = rpy2rot(etak.tail<3>());
    Eigen::Vector3d rBNn = etak.head<3>();
    Pose<double> Tnb(Rnb, rBNn);

    // Body to camera transformation
    Eigen::Matrix3d Rbc = camera.Tbc.rotationMatrix;
    Eigen::Vector3d rCBb = camera.Tbc.translationVector;

    // Navigation to camera transformation
    Eigen::Matrix3d Rnc = Rnb * Rbc;
    Eigen::Vector3d rCNn = rBNn + Rnb * rCBb;
    Pose<double> Tnc(Rnc, rCNn);

    double size = 1000;  // Size of the grid (adjust as needed)
    int step = 100;      // Step size for grid lines (adjust as needed)

    for (int x = -size; x <= size; x += step) {
        for (int y = -size; y <= size; y += step) {
            Eigen::Vector3d p1(x, y, 0);
            Eigen::Vector3d p2(x + step, y, 0);
            Eigen::Vector3d p3(x, y + step, 0);

            // Transform points to camera frame
            Eigen::Vector3d p1_cam = Rnc.transpose() * (p1 - rCNn);
            Eigen::Vector3d p2_cam = Rnc.transpose() * (p2 - rCNn);
            Eigen::Vector3d p3_cam = Rnc.transpose() * (p3 - rCNn);

            // Check if points are in front of the camera (z > 0 in camera frame)
            if (p1_cam.z() > 0 && p2_cam.z() > 0) {
                Eigen::Vector2d img_p1 = camera.vectorToPixel(p1_cam);
                Eigen::Vector2d img_p2 = camera.vectorToPixel(p2_cam);

                cv::line(img, cv::Point(img_p1[0]/divisor, img_p1[1]/divisor), 
                              cv::Point(img_p2[0]/divisor, img_p2[1]/divisor), 
                              cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            }
            
            if (p1_cam.z() > 0 && p3_cam.z() > 0) {
                Eigen::Vector2d img_p1 = camera.vectorToPixel(p1_cam);
                Eigen::Vector2d img_p3 = camera.vectorToPixel(p3_cam);

                cv::line(img, cv::Point(img_p1[0]/divisor, img_p1[1]/divisor), 
                              cv::Point(img_p3[0]/divisor, img_p3[1]/divisor), 
                              cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            }
        }
    }
}

void plotHorizon(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor)
{
    Eigen::Matrix3d Rnb = rpy2rot(etak.tail<3>());

    // Body to camera transformation
    Eigen::Matrix3d Rbc = camera.Tbc.rotationMatrix;

    // Navigation to camera transformation
    Eigen::Matrix3d Rnc = Rnb * Rbc;

    // Number of points to use for the horizon line
    const int num_points = 100;

    std::vector<cv::Point2f> horizon_points;

    for (int i = 0; i < num_points; ++i)
    {
        // Create a direction vector in the horizontal plane
        double angle = 2 * M_PI * i / num_points;
        Eigen::Vector3d dir_world(cos(angle), sin(angle), 0);
        
        // Transform the direction vector to camera coordinates
        Eigen::Vector3d dir_cam = Rnc.transpose() * dir_world;
        
        // Project the direction vector onto the image plane
        if (camera.isVectorWithinFOV(cv::Vec3d(dir_cam[0], dir_cam[1], dir_cam[2])))
        {
            Eigen::Vector2d pixel = camera.vectorToPixel(dir_cam);
            horizon_points.push_back(cv::Point2f(pixel[0]/divisor, pixel[1]/divisor));
        }
    }

    // Draw the horizon line
    if (horizon_points.size() > 1)
    {
        for (size_t i = 0; i < horizon_points.size() - 1; ++i)
        {
            cv::line(img, horizon_points[i], horizon_points[i+1], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }
        // Connect the last point to the first to close the loop
        cv::line(img, horizon_points.back(), horizon_points.front(), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }
}

void plotCompass(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor)
{
    Eigen::Matrix3d Rnb = rpy2rot(etak.tail<3>());
    Eigen::Vector3d rBNn = etak.head<3>();

    // Body to camera transformation
    Eigen::Matrix3d Rbc = camera.Tbc.rotationMatrix;
    Eigen::Vector3d rCBb = camera.Tbc.translationVector;

    // Navigation to camera transformation
    Eigen::Matrix3d Rnc = Rnb * Rbc;
    Eigen::Vector3d rCNn = rBNn + Rnb * rCBb;
    Pose<double> Tnc(Rnc, rCNn);

    std::vector<std::string> directions = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"};
    for (int i = 0; i < 8; ++i) {
        double angle = i * M_PI / 4;
        Eigen::Vector3d dir(cos(angle), sin(angle), 0);
        Eigen::Vector3d dir_cam = Rnc.transpose() * dir;
        
        if (camera.isVectorWithinFOV(cv::Vec3d(dir_cam[0], dir_cam[1], dir_cam[2]))) {
            Eigen::Vector2d img_point = camera.vectorToPixel(dir_cam);
            cv::putText(img, directions[i], cv::Point(img_point[0]/divisor, img_point[1]/divisor),
                        cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        }
    }
}

void plotEpipole(cv::Mat & img, const Eigen::Vector6d & etak, const Eigen::Vector6d & etakm1, const Camera & camera, const int & divisor)
{
    Eigen::Matrix3d Rnb = rpy2rot(etak.tail<3>());
    Eigen::Vector3d rBNn = etak.head<3>();
    Eigen::Vector3d rBNnm1 = etakm1.head<3>();
    Eigen::Vector3d t = rBNn - rBNnm1;
    
    // Body to camera transformation
    Eigen::Matrix3d Rbc = camera.Tbc.rotationMatrix;

    // Navigation to camera transformation
    Eigen::Matrix3d Rnc = Rnb * Rbc;
    
    Eigen::Vector3d e = Rnc.transpose() * t;
    
    if (camera.isVectorWithinFOV(cv::Vec3d(e[0], e[1], e[2]))) {
        Eigen::Vector2d img_e = camera.vectorToPixel(e);
        cv::circle(img, cv::Point(img_e[0]/divisor, img_e[1]/divisor), 5, cv::Scalar(255, 165, 0), -1, cv::LINE_AA);
    }
}

