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
#include <iomanip>
#include <sstream>
#include "BufferedVideo.h"
#include "Pose.hpp"
#include "rotation.hpp"
#include "Camera.h"
#include "DJIVideoCaption.h"
#include "rotation.hpp"
#include "GaussianInfo.hpp"
#include "funcmin.hpp"
#include "SystemVisualNav.h"
#include "MeasurementAltimeter.h"
#include "MeasurementOutdoorFlowBundle.h"
#include "visual_odometry.h"
#include "Camera.h"
#include "Plot.h"
#include "Event.h"
#include "SystemEstimator.h" 

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

// Forward declarations
static Eigen::Vector6d getInitialPose(const DJIVideoCaption & caption0);
static void plotGroundPlane(cv::Mat & img, const SystemVisualNav& system, const Camera & camera, const int & divisor);
static void plotHorizon(cv::Mat & img, const SystemVisualNav& system, const Camera & camera, const int & divisor);
static void plotCompass(cv::Mat & img, const SystemVisualNav& system, const Camera & camera, const int & divisor);
static void printZeta(const SystemVisualNav& system);
static void plotAltitude(cv::Mat & img, const double measured_altitude, const SystemVisualNav& system);
static void plotTranslationalVelocity(cv::Mat & img, const SystemVisualNav& system, const Camera & camera, const int & divisor); 
void printStateVector(const SystemVisualNav& system);

void runVisualOdometryFromVideo(const std::filesystem::path & videoPath, const std::filesystem::path & cameraPath, const std::filesystem::path & outputDirectory)
{
    int imgModulus  = 14;                    // Take frames divisible by this number
    int divisor     = 2;                     // Image scaling factor (used for plotting only)
    
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
        std::string outputFilename = videoPath.stem().string() + "_" + std::to_string(divisor) + "_" + std::to_string(imgModulus) + videoPath.extension().string();
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
    std::cout << "Subtitle file: " << subtitlePath.string() << std::endl;
    std::cout << "Total number of frames: " << nFrames << std::endl;
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Input video frame rate: " << fps << std::endl;
    std::cout << "Input video dimensions: [" << cap.get(cv::CAP_PROP_FRAME_WIDTH) << " x " << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << "]" << std::endl;

    BufferedVideoReader bufferedVideoReader(5);
    bufferedVideoReader.start(cap);

    cv::VideoWriter videoOut;
    BufferedVideoWriter bufferedVideoWriter(3);
    cv::Size frameSize;
    if (doExport)
    {
        
        frameSize.width     = cap.get(cv::CAP_PROP_FRAME_WIDTH)/divisor;
        frameSize.height    = cap.get(cv::CAP_PROP_FRAME_HEIGHT)/divisor;
        double outputFps    = fps/imgModulus;
        // int codec = cap.get(cv::CAP_PROP_FOURCC); // use same output video codec as input video
        int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // manually specify output video codec
        videoOut.open(outputPath.string(), codec, outputFps, frameSize);
        bufferedVideoWriter.start(videoOut);
    }
   
    // Visual odometry
    Eigen::VectorXd etak(6);
    Eigen::VectorXd etakm1 = etak;
    etak = getInitialPose(djiVideoCaption[0]);
    Eigen::VectorXd mu = Eigen::VectorXd::Zero(18);                                     // Initial state mean
    mu.segment<3>(0) = Eigen::Vector3d(0, 0, 0);                                        // Initial translational velocity (m/s)
    mu.segment<3>(3) = Eigen::Vector3d(0, 0, 0);                                        // Initial angular velocity (rad/s)
    mu.segment<6>(6) = Eigen::VectorXd(etak);                                           // Set initial pose eta 
    mu.segment<6>(12) = Eigen::VectorXd(etak);                                          // Set initial pose zeta (etakm1)

    // Initialize square root of covariance matrix (upper triangular)
    Eigen::MatrixXd S = Eigen::MatrixXd::Identity(18, 18);

    // Initial velocity uncertainty (first 6 states)
    S.block<3, 3>(0, 0) = Eigen::MatrixXd::Identity(3, 3) * 50;         // m/s
    S.block<3, 3>(3, 3) = Eigen::MatrixXd::Identity(3, 3) * 1;          // rad/s

    // Current pose eta uncertainty (middle 6 states)
    S.block<3, 3>(6, 6) = Eigen::MatrixXd::Identity(3, 3) * 0.01;       // m  
    S.block<3, 3>(9, 9) = Eigen::MatrixXd::Identity(3, 3) * 0.0005;     // rad

    // Previous pose zeta uncertainty (final 6 states)
    S.block<3, 3>(12, 12) = Eigen::MatrixXd::Identity(3, 3) * 50;       // m
    S.block<3, 3>(15, 15) = Eigen::MatrixXd::Identity(3, 3) * 0.1;      // rad
    auto p0 = GaussianInfo<double>::fromSqrtMoment(mu, S);              // Initialise system state density p(x0)     
    SystemVisualNav system(p0); 
    
    // Set camera pose w.r.t. body
    Eigen::Matrix3d Rbc { {0, 0, 1}, {1, 0, 0}, {0, 1, 0} };
    Eigen::Vector3d rBCn = Eigen::Vector3d::Zero();
    camera.Tbc.rotationMatrix = Rbc;
    camera.Tbc.translationVector = rBCn;

    cv::Mat imgk_raw;
    cv::Mat imgkm1_raw;
    cv::Mat imgout;
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOikm1;
    double currentTime = 0.0;
    int currentFrame = 0;
    double altitudekm1 = 0;
    
    double altitude = etak[2];
    int totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);                              // Number of frames in the selected video
    
    for (int i = 0, k = 0; currentFrame < totalFrames; ++i)
    {
        currentFrame++;
        currentTime = i / fps;

        imgk_raw = bufferedVideoReader.read();       // Capture frame by frame

        // cv::resize(imgk_raw, imgout, cv::Size(), 1.0/divisor, 1.0/divisor); // REMOVE THIS WHEN DONE W/ DEBUG 
        if (imgk_raw.empty())
        {
            break;
        }
        
        // Read altimeter value from subtitle file
        for (const auto& caption : djiVideoCaption)
        {
            if (caption.frameNum == i)
            {
                altitude = (caption.altitude - 7);
                break;
            }
        }

        if (altitude != altitudekm1)
        {
                MeasurementAltimeter measurementAltimeter(currentTime, camera, altitude);
                system.setZetaUpdateEnabled(false);  // Don't update zeta for altimeter
                std::cout << "Before altimeter update:" << std::endl;
                printZeta(system);
                measurementAltimeter.process(system);
                std::cout << "After altimeter update:" << std::endl;
                printZeta(system);
                altitudekm1 = altitude;

                MeasurementAltimeter measurementAltimeter(currentTime, camera, altitude);
                system.setZetaUpdateEnabled(false);                                         // Don't update zeta for altimeter
                std::cout << "altimeter update" << std::endl;
                measurementAltimeter.process(system);                                       // Process measurement event (do time update and measurement update)   
                altitudekm1 = altitude;
        }

        if (i % imgModulus == 0)
        {     
            if (k > 0)
            {     
                
                cv::resize(imgk_raw, imgout, cv::Size(), 1.0/divisor, 1.0/divisor);
                MeasurementOutdoorFlowBundle measurementFlowBundle(currentTime, camera, imgk_raw, imgkm1_raw, rQOikm1);

                // std::cout << "Before flow update:" << std::endl;
                // printZeta(system);
                system.setZetaUpdateEnabled(true);  // Do update zeta for flow
                measurementFlowBundle.process(system);
                // std::cout << "After flow update:" << std::endl;
                // printZeta(system);
                // system.setZetaUpdateEnabled(true);  // Do update
                
                // measurementFlowBundle.process(system);      // Process measurement event (do time update and measurement update)
                
                rQOikm1 = measurementFlowBundle.trackedPreviousFeatures();
                const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQOik = measurementFlowBundle.trackedCurrentFeatures();

                // Predicted flow field (used for plotting onto original image)
                Eigen::Matrix<double, 2, Eigen::Dynamic> rQOik_hat = measurementFlowBundle.predictedFeatures(system.density.mean(), system);
                
                // Plotting
                // std::vector<cv::Point2d> rQOikm1_scaled, rQOik_scaled, rQOik_hat_scaled;
                // int np = rQOik.cols();
                // rQOikm1_scaled.resize(np);
                // rQOik_scaled.resize(np);
                // rQOik_hat_scaled.resize(np);
                // for (int j = 0; j < np; ++j)
                // {
                //     rQOikm1_scaled[j].x     = rQOikm1(0, j)/divisor;
                //     rQOikm1_scaled[j].y     = rQOikm1(1, j)/divisor;

                //     rQOik_scaled[j].x       = rQOik(0, j)/divisor;
                //     rQOik_scaled[j].y       = rQOik(1, j)/divisor;

                //     rQOik_hat_scaled[j].x   = rQOik_hat(0, j)/divisor;
                //     rQOik_hat_scaled[j].y   = rQOik_hat(1, j)/divisor;
                // }
                

                // for (int j = 0; j < rQOik.cols(); ++j)
                // {
                //     // Only draw prediction if this was a valid measurement
                //     if (j < measurementFlowBundle.inlierMask().size())  // Check we have a mask entry
                //     {
                //         // Predicted flow - only draw for inliers
                //         if (measurementFlowBundle.inlierMask()[j]) {
                //             cv::arrowedLine(imgout, rQOikm1_scaled[j], rQOik_hat_scaled[j], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
                //             // Measured inliers
                //             cv::arrowedLine(imgout, rQOikm1_scaled[j], rQOik_scaled[j], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                //         }
                //         else {
                //             // Measured outliers
                //             cv::arrowedLine(imgout, rQOikm1_scaled[j], rQOik_scaled[j], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                //         }
                //     }
                // }

                // plotTranslationalVelocity(imgout, system, camera, divisor);  
                // plotGroundPlane(imgout, system, camera, divisor);
                plotHorizon(imgout, system, camera, divisor);
                plotCompass(imgout, system, camera, divisor);
                plotAltitude(imgout, altitude, system);  

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
            k++;
           
        }
    } 

    if (doExport)
    {
        // cv::Mat imgout = plot.getFrame();
        cv::resize(imgout, imgout, frameSize);
        assert(!imgout.empty());                // Ensure the frame is not empty before writing
        bufferedVideoWriter.write(imgout);    
    }
    printStateVector(system);
    bufferedVideoReader.stop();
    bufferedVideoWriter.stop();
}

Eigen::Vector6d getInitialPose(const DJIVideoCaption & caption0)
{
    double h        = caption0.altitude;    // Altitude (GPS) [m]
    double ga       = h - 7;                // Altitude (AGL) [m]

    Eigen::Vector6d eta0;
    eta0 << 0, 0, -ga,         // Initial position 
    -0.009, 0.09, -1;          // Initial orientation 
    return eta0;
}

 void plotGroundPlane(cv::Mat & img, const SystemVisualNav& system, const Camera & camera, const int & divisor)
{
    Eigen::VectorXd state = system.density.mean();
    Eigen::Vector6d etak = state.segment<6>(6);  // Extract elements 6-11
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
}// use cv clipline

void plotHorizon(cv::Mat & img, const SystemVisualNav& system, const Camera & camera, const int & divisor)
{
    // Extract etak from the state vector (elements 6-11)
    Eigen::VectorXd state = system.density.mean();
    Eigen::Vector6d etak = state.segment<6>(6);  // Extract elements 6-11
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
            cv::line(img, horizon_points[i], horizon_points[i+1], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        }
        // Connect the last point to the first to close the loop
        cv::line(img, horizon_points.back(), horizon_points.front(), cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    }
}

void plotCompass(cv::Mat & img, const SystemVisualNav& system, const Camera & camera, const int & divisor)
{
    // Extract etak from the state vector (elements 6-11)
    Eigen::VectorXd state = system.density.mean();
    Eigen::Vector6d etak = state.segment<6>(6);  // Extract elements 6-11
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

// void plotEpipole(cv::Mat & img, const SystemVisualNav& system, const Camera & camera, const int & divisor)
// {
//     // Extract etak from the state vector (elements 6-11)
//     Eigen::VectorXd state = system.density.mean();
//     Eigen::Vector6d etakm1 = state.segment<6>(12);  // Extract elements 12-17
//     Eigen::Vector6d etak = state.segment<6>(6);  // Extract elements 6-11
//     Eigen::Matrix3d Rnb = rpy2rot(etak.tail<3>());
//     Eigen::Vector3d rBNn = etak.head<3>();
//     Eigen::Vector3d rBNnm1 = etakm1.head<3>();
//     Eigen::Vector3d t = rBNn - rBNnm1;
    
//     // Body to camera transformation
//     Eigen::Matrix3d Rbc = camera.Tbc.rotationMatrix;

//     // Navigation to camera transformation
//     Eigen::Matrix3d Rnc = Rnb * Rbc;
    
//     Eigen::Vector3d e = Rnc.transpose() * t;
    
//     if (camera.isVectorWithinFOV(cv::Vec3d(e[0], e[1], e[2]))) {
//         Eigen::Vector2d img_e = camera.vectorToPixel(e);
//         cv::circle(img, cv::Point(img_e[0]/divisor, img_e[1]/divisor), 5, cv::Scalar(255, 165, 0), -1, cv::LINE_AA);
//     }
// }

void printZeta(const SystemVisualNav& system) {
    Eigen::VectorXd state = system.density.mean();
    Eigen::Vector6d zeta = state.segment<6>(12);  // Elements 12-17 are zeta
    
    std::cout << "\nZeta (Previous Pose):" << std::endl;
    std::cout << "Position (m):" << std::endl;
    std::cout << "  N (North): " << zeta(0) << std::endl;
    std::cout << "  E (East): " << zeta(1) << std::endl;
    std::cout << "  D (Down): " << zeta(2) << std::endl;

    std::cout << "Orientation (rad):" << std::endl;
    std::cout << "  Roll: " << zeta(3) << std::endl;
    std::cout << "  Pitch: " << zeta(4) << std::endl;
    std::cout << "  Yaw: " << zeta(5) << std::endl;
    std::cout << std::endl;

    // Also print eta for comparison
    Eigen::Vector6d eta = state.segment<6>(6);  // Elements 6-11 are eta
    std::cout << "Eta (Current Pose) for comparison:" << std::endl;
    std::cout << "Position (m):" << std::endl;
    std::cout << "  N (North): " << eta(0) << std::endl;
    std::cout << "  E (East): " << eta(1) << std::endl;
    std::cout << "  D (Down): " << eta(2) << std::endl;

    std::cout << "Orientation (rad):" << std::endl;
    std::cout << "  Roll: " << eta(3) << std::endl;
    std::cout << "  Pitch: " << eta(4) << std::endl;
    std::cout << "  Yaw: " << eta(5) << std::endl;
    std::cout << "\nDifference (eta - zeta):" << std::endl;
    std::cout << (eta - zeta).transpose() << std::endl;
    std::cout << std::endl;
}

void printStateVector(const SystemVisualNav& system) {
    Eigen::VectorXd state = system.density.mean();
    std::cout << "\nState vector breakdown:" << std::endl;

    // Linear velocity (m/s)
    std::cout << "Linear velocity (m/s):" << std::endl;
    std::cout << "  vN (North): " << state(0) << std::endl;   // N velocity
    std::cout << "  vE (East): " << state(1) << std::endl;    // E velocity
    std::cout << "  vD (Down): " << state(2) << std::endl;    // D velocity

    // Angular velocity (rad/s) -> now as roll rate, pitch rate, yaw rate
    std::cout << "Angular velocity (rad/s):" << std::endl;
    std::cout << "  Roll rate: " << state(3) << std::endl;    // Roll rate
    std::cout << "  Pitch rate: " << state(4) << std::endl;   // Pitch rate
    std::cout << "  Yaw rate: " << state(5) << std::endl;     // Yaw rate

    // Position (m)
    std::cout << "Position (m):" << std::endl;
    std::cout << "  N (North): " << state(6) << std::endl;    // N position
    std::cout << "  E (East): " << state(7) << std::endl;     // E position
    std::cout << "  D (Down): " << state(8) << std::endl;     // D position

    // Orientation (rad)
    std::cout << "Orientation (rad):" << std::endl;
    std::cout << "  Roll: " << state(9) << std::endl;    // Roll
    std::cout << "  Pitch: " << state(10) << std::endl;  // Pitch
    std::cout << "  Yaw: " << state(11) << std::endl;    // Yaw

    std::cout << std::endl;  // Add a blank line for readability
}

void plotTranslationalVelocity(cv::Mat & img, const SystemVisualNav& system, const Camera & camera, const int & divisor)
{
    // Extract velocities and poses from state
    Eigen::VectorXd state = system.density.mean();
    Eigen::Vector3d velocity = state.segment<3>(0);  // First 3 states are velocities
    Eigen::Vector6d etak = state.segment<6>(6);      // Current pose
    Eigen::Vector6d etakm1 = state.segment<6>(12);   // Previous pose
    
    // Get positions
    Eigen::Vector3d rBNn = etak.head<3>();
    Eigen::Vector3d rBNnm1 = etakm1.head<3>();
    
    // Get rotation matrices
    Eigen::Matrix3d Rnb = rpy2rot(etak.tail<3>());
    Eigen::Matrix3d Rbc = camera.Tbc.rotationMatrix;
    Eigen::Matrix3d Rnc = Rnb * Rbc;
    
    // Calculate translation vector between frames
    Eigen::Vector3d t = rBNn - rBNnm1;
    
    // Transform translation to camera frame
    Eigen::Vector3d t_cam = Rnc.transpose() * t;
    
    // Project epipole into image
    if (t_cam.z() > 0)  // Only plot if in front of camera
    {
        // Project epipole to get velocity direction in image
        Eigen::Vector2d epipole = camera.vectorToPixel(t_cam);
        
        // Get velocity magnitude
        double velocity_magnitude = velocity.norm();
        double scale_factor = 50.0;  // Adjust this to make marker more visible
        double marker_size = std::min(30.0, velocity_magnitude * scale_factor);
        
        // Draw circular marker at epipole
        cv::circle(img, 
                  cv::Point(epipole[0]/divisor, epipole[1]/divisor),
                  marker_size/divisor,
                  cv::Scalar(0, 165, 255),  // Orange color (BGR)
                  2);  // Filled circle
        
        // Add velocity magnitude text
        std::stringstream ss;
        ss << std::fixed << std::setprecision(1) << velocity_magnitude << " m/s";
        cv::putText(img, 
                   ss.str(),
                   cv::Point(epipole[0]/divisor + marker_size/divisor + 5, 
                           epipole[1]/divisor),
                   cv::FONT_HERSHEY_SIMPLEX,
                   0.5,
                   cv::Scalar(0, 165, 255),
                   1,
                   cv::LINE_AA);
    }
}

static void plotAltitude(cv::Mat & img, const double measured_altitude, const SystemVisualNav& system) {
    // Get predicted altitude from system state (index 8 is Down position)
    double predicted_altitude = system.density.mean()(8);
    
    // Format altitude strings with 2 decimal places
    std::stringstream ss_meas, ss_pred;
    ss_meas << std::fixed << std::setprecision(2) << "Measured Altitude: " << measured_altitude << "m";
    ss_pred << std::fixed << std::setprecision(2) << "Estimated Altitude: " << -predicted_altitude << "m";
    
    // Position text in top-left corner
    cv::Point measPos(20, 30);  // Position for measured altitude
    cv::Point predPos(20, 60);  // Position for predicted altitude, slightly below measured
    
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.7;
    int thickness = 2;

    cv::putText(img, 
                ss_meas.str(),
                measPos,
                fontFace,
                fontScale,
                cv::Scalar(0, 255, 0),  // Green for measured
                thickness,
                cv::LINE_AA);
                
    cv::putText(img, 
                ss_pred.str(),
                predPos,
                fontFace,
                fontScale,
                cv::Scalar(255, 0, 0),  // Blue for estimated
                thickness,
                cv::LINE_AA);
}