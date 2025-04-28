#include <filesystem>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/core/types.hpp>
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/imgcodecs.hpp>
#include "BufferedVideo.h"
#include "visualNavigation.h"
#include "Camera.h"
#include "Plot.h"
#include "plot_util.h"
#include "imagefeatures.h"
#include "image_features.h"
#include "rotation.hpp"
#include "SystemSLAMPoseLandmarks.h"
#include "SystemSLAMPointLandmarks.h"
#include "MeasurementSLAMPointBundle.h"
#include "MeasurementSLAMIdenticalTagBundle.h"
#include "MeasurementSLAMUniqueTagBundle.h"


// Structure to hold feature information
struct Feature {
    cv::Point2f point; // The location of the feature in the image
    float score;       // The Harris corner score of the feature
};

void runVisualNavigationFromVideo(const std::filesystem::path & videoPath, const std::filesystem::path & cameraPath, int scenario, int interactive, const std::filesystem::path & outputDirectory)
{
    assert(!videoPath.empty());

    // Output video path
    std::filesystem::path outputPath;
    bool doExport = !outputDirectory.empty();
    if (doExport)
    {
        std::string outputFilename = videoPath.stem().string()
                                   + "_out"
                                   + videoPath.extension().string();
        outputPath = outputDirectory / outputFilename;
    }

    // Load camera calibration
    Camera cam;
    
    if (!std::filesystem::exists(cameraPath))
    {
        std::cout << "File: " << cameraPath << " does not exist" << std::endl;
        return;
    }
    cv::FileStorage fs(cameraPath.string(), cv::FileStorage::READ);
    assert(fs.isOpened());
    fs["camera"] >> cam;

    // Display loaded calibration data
    cam.printCalibration();

    // Open input video
    cv::VideoCapture cap(videoPath.string());
    assert(cap.isOpened());
    double fps = cap.get(cv::CAP_PROP_FPS);

    BufferedVideoReader bufferedVideoReader(5);
    bufferedVideoReader.start(cap);

    cv::VideoWriter videoOut;
    BufferedVideoWriter bufferedVideoWriter(3);
    //cv::Size frameSize(2 * static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    cv::Size frameSize(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    
    if (doExport)
    {

        double outputFps    = fps;
        //int codec = cap.get(cv::CAP_PROP_FOURCC); // use same output video codec as input video
        int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // manually specify output video codec
        videoOut.open(outputPath.string(), codec, outputFps, frameSize, true);
        assert(videoOut.isOpened());
        std::cout << "framesize width" << frameSize.width << std::endl;
        std::cout << "framesize height" << frameSize.height << std::endl;
        bufferedVideoWriter.start(videoOut);
    }

    // Visual navigation
    double currentTime = 0.0;
    Eigen::Matrix3d Rbc;
        Rbc <<  0, 0, 1,
                1, 0, 0,
                0, 1, 0;
    Eigen::Vector3d rBCn = Eigen::Vector3d::Zero();
    cam.Tbc.rotationMatrix = Rbc;
    cam.Tbc.translationVector = rBCn;
    Plot plot(cam);                                                                   // Initialise plot object
    int totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);                              // Number of frames in the selected video
    //int totalFrames = 100;
    int currentFrame = 0;                                                               // Frame counter

    Eigen::VectorXd mu = Eigen::VectorXd::Zero(12);                                     // Initial state mean
    mu.segment<3>(0) = Eigen::Vector3d(0.01, 0.01, 0.01);                               // Initial linear velocity (m/s)
    mu.segment<3>(3) = Eigen::Vector3d(0.01, 0.01, 0.01);                               // Initial angular velocity (rad/s)
    mu.segment<3>(6) = Eigen::Vector3d(0, 0, -1.5);                                     // Set initial position (0 North, 0 East, -1.7 Down)
    mu.segment<3>(9) = Eigen::Vector3d(0, 0, 0);                                        // Set inital orientation (c3 aligned with b1)

    // Initialize square root of covariance matrix (upper triangular)
    Eigen::MatrixXd S = Eigen::MatrixXd::Identity(12, 12) * std::sqrt(0.5);  // Initialize square root of covariance matrix (upper triangular)
    auto p0 = GaussianInfo<double>::fromSqrtMoment(mu, S);                   // Initialise system state density p(x0)                   

    if (scenario == 1) {
        
        std::cout << "Running scenario 1: Unique Tags" << std::endl;
        int maxNumFeatures = 25;
        SystemSLAMPoseLandmarks system(p0);

    while (true) {
        
        // Get next input frame
        cv::Mat imgin = bufferedVideoReader.read();
        
        if (imgin.empty()) {
            break;
        }

        //printStateVector(system);
        currentFrame++;
        currentTime += 1.0 / fps;
        bool isLastFrame = (currentFrame == totalFrames);
        //cv::resize(imgin, resized_img, cv::Size(imgin.cols*resize_scale, imgin.rows*resize_scale), cv::INTER_LINEAR);
        ArUcoDetectionResult result = detectAndDrawArUco(imgin, maxNumFeatures);                    // Identify tags using ArUco detector in imagefeatures.cpp                                               
        Eigen::Matrix<double, 8, Eigen::Dynamic> Y = buildMeasurementVector(result.markerCorners);  // Build measurement vector
        MeasurementUniqueTagBundle measurement(currentTime, Y, cam, result.markerIds);              // Create measurement object
        measurement.process(system, scenario);                                                      // Process measurement event (do time update and measurement update)
        
        visualizeMeasurements(result.image, result, measurement, cam);                              // Visualize detected tags from the current frame
        system.view() = result.image.clone();                                                       // Get a copy of the image for plot to draw on
        plot.setData(system, measurement);                                                          // Update plot
        plot.render();                                                                              // Render 

        // Write output frame 
        if (doExport) {

                cv::Mat imgout = plot.getFrame();
                cv::resize(imgout, imgout, frameSize);
                assert(!imgout.empty());  // Ensure the frame is not empty before writing
                bufferedVideoWriter.write(imgout);    

            }
            
        if (interactive == 2 || (interactive == 1 && isLastFrame)) {
                plot.start();    // Start handling plot GUI events (blocking)
            }
        }
    }

    else if (scenario == 2) {

        std::cout << "Running scenario 2: Identical Tags" << std::endl;
        int maxNumFeatures = 25;
        SystemSLAMPoseLandmarks system(p0);

        while (true) {

            //printStateVector(system);
            currentFrame++;
            currentTime += 1.0 / fps;
            bool isLastFrame = (currentFrame == totalFrames);
            cv::Mat imgin = bufferedVideoReader.read();
            if (imgin.empty()) break;

            ArUcoDetectionResult result = detectAndDrawArUco(imgin, maxNumFeatures);
            system.view() = result.image.clone();
            Eigen::Matrix<double, 8, Eigen::Dynamic> Y = buildMeasurementVector(result.markerCorners);
            MeasurementIdenticalTagBundle measurement(currentTime, Y, cam, result.markerIds);
            measurement.process(system, scenario);

            visualizeMeasurementsIdenticalTags(result.image, result, measurement, cam);
            
            plot.setData(system, measurement);
            plot.render();
            // Write output frame 
            if (doExport) {

                    cv::Mat imgout = plot.getFrame();
                    cv::resize(imgout, imgout, frameSize);
                    assert(!imgout.empty());  // Ensure the frame is not empty before writing
                    bufferedVideoWriter.write(imgout);    

                }
                
            if (interactive == 2 || (interactive == 1 && isLastFrame)) {
                    plot.start();    // Start handling plot GUI events (blocking)
                }    
        }
    }

    else if (scenario == 3) {

        std::cout << "Running scenario 3: Points" << std::endl;
        SystemSLAMPointLandmarks system(p0);
        int maxNumFeatures = 500;

        while (true) {

            //SystemSLAMPointLandmarks system = exampleSystemFromChessboardImage(camera, chessboardImage);

            // Get next input frame
            cv::Mat imgin = bufferedVideoReader.read();
            
            if (imgin.empty()) break;
            system.view() = imgin;
            //printStateVector(system);
            currentFrame++;
            currentTime += 1.0 / fps;
            bool isLastFrame = (currentFrame == totalFrames);
            
            std::vector<PointFeature> features = detectFeatures(imgin, maxNumFeatures);
            std::cout << features.size() << " features found in frame" << std::endl;
            assert(features.size() > 0);
            assert(features.size() <= maxNumFeatures);

            if (currentFrame==1) { 

                // Select initial landmarks
                std::vector<PointFeature> initialLandmarks = selectInitialLandmarks(features, 20);  // 20 is the max number of initial landmarks

                // Initialize landmarks in the system
                for (const auto& landmark : initialLandmarks) {
                    Eigen::Vector3d landmarkPosition = estimateLandmarkPosition(landmark, system, cam);
                    system.initializeNewLandmark(landmarkPosition);
                }    
            }
            

            Eigen::Matrix<double, 2, Eigen::Dynamic> Y(2, features.size()); // Build measurement vector
            for (std::size_t i = 0; i < features.size(); ++i)
            {
                Y.col(i) << features[i].x, features[i].y;
            }

            MeasurementPointBundle measurement(currentTime, Y, cam);
            std::cout << "Before measurement.process(system):" << std::endl;
            std::cout << "Number of landmarks in system: " << system.numberLandmarks() << std::endl;

            //measurement.process(system, scenario);

            std::cout << "After measurement.process(system):" << std::endl;
            std::cout << "Number of landmarks in system: " << system.numberLandmarks() << std::endl; 
            // ------------------------------------------------------------------------
            // Select landmarks expected to be within the field of view of the camera
            // ------------------------------------------------------------------------
            std::vector<std::size_t> idxLandmarks;
            idxLandmarks.reserve(system.numberLandmarks());  // Reserve maximum possible size to avoid reallocation
            for (std::size_t j = 0; j < system.numberLandmarks(); ++j)
            {
                Eigen::Vector3d murPNn = system.landmarkPositionDensity(j).mean();
                cv::Vec3d rPNn;
                cv::eigen2cv(murPNn, rPNn);
                // Extract camera pose from system state
                Eigen::Vector3d rCNn = system.cameraPosition(cam, system.density.mean());
                Eigen::Matrix3d Rnc = system.cameraOrientation(cam, system.density.mean());
                Pose<double> Tnc;
                Tnc.rotationMatrix = Rnc;
                Tnc.translationVector = rCNn;

                // Convert camera pose to body pose
                Pose<double> Tnb = cam.cameraToBody(Tnc);
                
                if (cam.isWorldWithinFOV(rPNn, Tnb))
                {
                    std::cout << "Landmark " << j << " is expected to be within camera FOV" << std::endl;
                    idxLandmarks.push_back(j);
                }
                else
                {
                    std::cout << "Landmark " << j << " is NOT expected to be within camera FOV" << std::endl;
                }
            }

            // ------------------------------------------------------------------------
            // Run data association
            // ------------------------------------------------------------------------
            const std::vector<int> & idxFeatures = measurement.associate(system, idxLandmarks);

            //     // ------------------------------------------------------------------------
            //     // Visualisation and console output
            //     // ------------------------------------------------------------------------
            plotAllFeatures(system.view(), Y);
            //const std::vector<int>& idxFeatures = measurement.getIdxFeatures();
            //const std::vector<std::size_t>& idxLandmarks = measurement.getIdxLandmarks();
            std::cout << "idxlandmarks size "<< idxLandmarks.size() << "." << std::endl;
            std::cout << "idxfeatures size "<< idxFeatures.size() << "." << std::endl;

            for (std::size_t jj = 0; jj < idxFeatures.size(); ++jj)
            {
                int j           = idxLandmarks[jj];     // Index of landmark in state vector
                int i           = idxFeatures[jj];      // Index of feature
                bool isMatch    = i >= 0;

                GaussianInfo<double> featureDensity = measurement.predictFeatureDensity(system, j);
                const Eigen::VectorXd & murQOi = featureDensity.mean();

                // Plot confidence ellipse and landmark index text
                Eigen::Vector3d colour = isMatch ? Eigen::Vector3d(0, 0, 255) : Eigen::Vector3d(255, 0, 0);
                plotGaussianConfidenceEllipse(system.view(), featureDensity, colour);
                plotLandmarkIndex(system.view(), murQOi, colour, j);

                if (isMatch)
                    std::cout << "Feature " << i << " located at [" << Y.col(i).transpose() << "] in image matches landmark " << j <<"." << std::endl;
                else
                    std::cout << "No feature associated with landmark "<< j << "." << std::endl;
            }
            plotMatchedFeatures(system.view(), idxFeatures, Y);
            std::cout << std::endl;
                                                    
            // // Eigen::Matrix<double, 2, Eigen::Dynamic> Y = buildMeasurementVector(result.markerCorners);  // Build measurement vector
            //                                    // Create measurement object
            // // measurement.process(system);                                                                // Process measurement event (do time update and measurement update)
            // Visualize initialized landmarks
            cv::Mat visualizationImage = imgin.clone();
            //visualizePointLandmarks(visualizationImage, system);

            // // Display the image
            // cv::imshow("Initialized Landmarks", visualizationImage);
            // cv::waitKey(0);

            //visualizeMeasurementsPoint(result.image, result, measurement, cam);                               // Visualize detected tags from the current frame
            //system.view() = result.image.clone();                                                        // Get a copy of the image for plot to draw on
            plot.setData(system, measurement);                                                             // Update plot
            plot.render();                                                                                 // Render 

            // Write output frame 
            if (doExport) {
                    cv::Mat imgout = plot.getFrame();
                    cv::resize(imgout, imgout, frameSize);
                    assert(!imgout.empty());  // Ensure the frame is not empty before writing
                    bufferedVideoWriter.write(imgout);    
                }
                
            if (interactive == 2 || (interactive == 1 && isLastFrame)) {
                    plot.start();    // Start handling plot GUI events (blocking)
                }
        }
    }

    else {
        // Throw an exception for invalid scenario
        throw std::invalid_argument("Invalid scenario. Please choose 1, 2, or 3.");
    }

    if (doExport)
    {
        bufferedVideoWriter.stop();
        std::cout << "Exported video to: " << outputPath << std::endl;
    }
    bufferedVideoReader.stop();
    videoOut.release();
    cap.release();
}

// Function to draw pose axes on the image
void drawPoseAxes(cv::Mat& img, const Eigen::Matrix4d& pose, const Camera& camera, double length)
{
    std::vector<cv::Point3f> axesPoints = {
        cv::Point3f(0, 0, 0),
        cv::Point3f(length, 0, 0),
        cv::Point3f(0, length, 0),
        cv::Point3f(0, 0, length)
    };

    std::vector<cv::Point2f> imagePoints;
    cv::Mat rotVec, transVec;
    
    // Extract rotation and translation from pose matrix
    Eigen::Matrix3d rotMat = pose.block<3,3>(0,0);
    Eigen::Vector3d trans = pose.block<3,1>(0,3);
    
    cv::eigen2cv(rotMat, rotVec);
    cv::Rodrigues(rotVec, rotVec); // Convert rotation matrix to rotation vector
    cv::eigen2cv(trans, transVec);

    cv::projectPoints(axesPoints, rotVec, transVec, camera.cameraMatrix, camera.distCoeffs, imagePoints);

    // Draw axes
    cv::line(img, imagePoints[0], imagePoints[1], cv::Scalar(0,0,255), 2); // X-axis (Red)
    cv::line(img, imagePoints[0], imagePoints[2], cv::Scalar(0,255,0), 2); // Y-axis (Green)
    cv::line(img, imagePoints[0], imagePoints[3], cv::Scalar(255,0,0), 2); // Z-axis (Blue)
}

void visualizeMeasurements(cv::Mat& outputImage, const ArUcoDetectionResult& result, const MeasurementUniqueTagBundle& measurement, const Camera& cam)
{

    for (size_t i = 0; i < result.markerIds.size(); ++i) {
        //int markerId = result.markerIds[i];
        const std::vector<cv::Point2f>& corners = result.markerCorners[i];

        // Use the getTagPose function to estimate the pose of the tag
        Eigen::Matrix4d tagPose = measurement.getTagPoseCamera(corners, cam, 0.166);  // Assuming 166mm tag size

        // Draw the pose axes
        drawPoseAxes(outputImage, tagPose, cam, 0.083);  // Draw axes half the size of the tag
    }
    
}

void visualizeMeasurementsIdenticalTags(cv::Mat& outputImage, const ArUcoDetectionResult& result, const MeasurementIdenticalTagBundle& measurement, const Camera& cam)
{

    for (size_t i = 0; i < result.markerIds.size(); ++i) {
        //int markerId = result.markerIds[i];
        const std::vector<cv::Point2f>& corners = result.markerCorners[i];

        // Use the getTagPose function to estimate the pose of the tag
        Eigen::Matrix4d tagPose = measurement.getTagPoseCamera(corners, cam, 0.166);  // Assuming 166mm tag size

        // Draw the pose axes
        drawPoseAxes(outputImage, tagPose, cam, 0.083);  // Draw axes half the size of the tag
    }
    
}

Eigen::Matrix<double, 8, Eigen::Dynamic> buildMeasurementVector(const std::vector<std::vector<cv::Point2f>>& features)
{
    Eigen::Matrix<double, 8, Eigen::Dynamic> Y(8, features.size());
    
    for (std::size_t i = 0; i < features.size(); ++i)
    {
            Y.col(i) << features[i][0].x, features[i][0].y,
                        features[i][1].x, features[i][1].y,
                        features[i][2].x, features[i][2].y,
                        features[i][3].x, features[i][3].y;
    }
    
    return Y;
}

std::vector<PointFeature> selectInitialLandmarks(const std::vector<PointFeature>& features, int maxLandmarks) {
    std::vector<PointFeature> selectedLandmarks;
    const float minDistance = 30.0f;  // Minimum pixel distance between landmarks

    for (const auto& feature : features) {
        bool isFarEnough = true;
        for (const auto& landmark : selectedLandmarks) {
            float distance = std::sqrt(std::pow(feature.x - landmark.x, 2) + std::pow(feature.y - landmark.y, 2));
            if (distance < minDistance) {
                isFarEnough = false;
                break;
            }
        }

        if (isFarEnough) {
            selectedLandmarks.push_back(feature);
            if (selectedLandmarks.size() >= maxLandmarks) {
                break;
            }
        }
    }

    return selectedLandmarks;
}

Eigen::Vector3d estimateLandmarkPosition(const PointFeature& feature, const SystemSLAMPointLandmarks& system, const Camera& camera) {
    Eigen::Vector3d rCNn = system.cameraPosition(camera, system.density.mean());
    Eigen::Matrix3d Rnc = system.cameraOrientation(camera, system.density.mean());

    cv::Vec2d cvFeaturePosition(feature.x, feature.y);
    cv::Vec3d cvUnitVector = camera.pixelToVector(cvFeaturePosition);
    Eigen::Vector3d unitVector(cvUnitVector[0], cvUnitVector[1], cvUnitVector[2]);

    double estimatedDepth = 1.0; // Adjust based on your environment

    return Rnc * (unitVector * estimatedDepth) + rCNn;
}

void visualisePointLandmarks(cv::Mat& image, const std::vector<PointFeature>& landmarks) {
    for (const auto& landmark : landmarks) {
        cv::circle(image, cv::Point(landmark.x, landmark.y), 5, cv::Scalar(0, 255, 0), -1);
    }
}

// Eigen::Matrix<double, 2, Eigen::Dynamic> buildMeasurementVectorPoint(const std::vector<std::vector<cv::Point2f>>& features)
// {
//     Eigen::Matrix<double, 2, Eigen::Dynamic> Y(2, features.size()); // Build measurement vector
//             for (std::size_t i = 0; i < features.size(); ++i)
//             {
//                 Y.col(i) << features[i].x, features[i].y;
//             }
    
//     return Y;
// }

void printStateVector(const SystemSLAMPoseLandmarks& system) {
    Eigen::VectorXd state = system.density.mean();
    std::cout << "\nState vector breakdown:" << std::endl;

    // Linear velocity (m/s)
    std::cout << "Linear velocity (m/s):" << std::endl;
    std::cout << "  vx: " << state(0) << std::endl;
    std::cout << "  vy: " << state(1) << std::endl;
    std::cout << "  vz: " << state(2) << std::endl;

    // Angular velocity (rad/s)
    std::cout << "Angular velocity (rad/s):" << std::endl;
    std::cout << "  ωx: " << state(3) << std::endl;
    std::cout << "  ωy: " << state(4) << std::endl;
    std::cout << "  ωz: " << state(5) << std::endl;

    // Position (m)
    std::cout << "Position (m):" << std::endl;
    std::cout << "  x: " << state(6) << std::endl;
    std::cout << "  y: " << state(7) << std::endl;
    std::cout << "  z: " << state(8) << std::endl;

    // Orientation (rad)
    std::cout << "Orientation (rad):" << std::endl;
    std::cout << "  roll: " << state(9) << std::endl;
    std::cout << "  pitch: " << state(10) << std::endl;
    std::cout << "  yaw: " << state(11) << std::endl;

    std::cout << std::endl;  // Add a blank line for readability
}