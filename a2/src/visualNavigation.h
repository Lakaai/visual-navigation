// #ifndef VISUALNAVIGATION_H
// #define VISUALNAVIGATION_H

// #include <filesystem>
// #include <opencv2/core/mat.hpp>
// #include <Eigen/Core>
// #include "Camera.h"
// #include "imagefeatures.h"
// #include "image_features.h"
// // #include "MeasurementSLAMUniqueTagBundle.h"
// // #include "MeasurementSLAMIdenticalTagBundle.h"
// #include "SystemVisualNavPointLandmarks.h"  


// void runVisualNavigationFromVideo(const std::filesystem::path & videoPath, const std::filesystem::path & cameraPath, int scenario = 3, int interactive = 0, const std::filesystem::path & outputDirectory = "");
// void drawPoseAxes(cv::Mat& img, const Eigen::Matrix4d& pose, const Camera& camera, double length);
// // void visualizeMeasurements(cv::Mat& outputImage, const ArUcoDetectionResult& result, const MeasurementUniqueTagBundle& measurement, const Camera& cam);
// // void visualizeMeasurementsIdenticalTags(cv::Mat& outputImage, const ArUcoDetectionResult& result, const MeasurementIdenticalTagBundle& measurement, const Camera& cam);
// Eigen::Matrix<double, 8, Eigen::Dynamic> buildMeasurementVector(const std::vector<std::vector<cv::Point2f>>& features);
// // void printStateVector(const SystemSLAMPoseLandmarks& system);
// void visualizePointLandmarks(cv::Mat& image, const std::vector<PointFeature>& landmarks);
// // Eigen::Vector3d estimateLandmarkPosition(const PointFeature& feature, const SystemSLAMPointLandmarks& system, const Camera& camera);
// std::vector<PointFeature> selectInitialLandmarks(const std::vector<PointFeature>& features, int maxLandmarks);

// #endif