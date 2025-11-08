#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/imgcodecs.hpp>
#include "imagefeatures.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

// Structure to hold feature information
struct Feature {
    cv::Point2f point; // The location of the feature in the image
    float score;       // The Harris corner score of the feature
};

cv::Mat detectAndDrawHarris(const cv::Mat & img, int maxNumFeatures)
{
    // Copy the input image to draw features on
    cv::Mat imgout = img.clone();

    // Harris corner detection parameters
    float harrisThreshold = 100.0; // Threshold for detecting corners
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;

    // Convert to grayscale if necessary
    cv::Mat imgGrayScale;
    if (img.channels() == 1) {
        imgGrayScale = img;
    } else {
        cv::cvtColor(img, imgGrayScale, cv::COLOR_BGR2GRAY);
    }

    // Detect Harris corners
    cv::Mat harrisDst;
    cv::cornerHarris(imgGrayScale, harrisDst, blockSize, apertureSize, k);

    // Normalize and convert to absolute values
    cv::Mat harrisNorm, harrisScaled;
    cv::normalize(harrisDst, harrisNorm, 0, 255, cv::NORM_MINMAX, CV_32FC1);
    cv::convertScaleAbs(harrisNorm, harrisScaled);

    // Collect features above the threshold
    std::vector<Feature> features;
    for (int i = 0; i < harrisNorm.rows; i++) {
        for (int j = 0; j < harrisNorm.cols; j++) {
            float score = harrisNorm.at<float>(i, j);
            if (score > harrisThreshold) {
                features.push_back({cv::Point2f(j, i), score});
            }
        }
    }

    // Draw orange circles around all features above the threshhold
    for (size_t i = 0; i < features.size(); i++) {
        cv::circle(imgout, features[i].point, 2, cv::Scalar(0,165,255), 2, cv::LINE_AA);
    }

    // Sort features by score in descending order
    std::sort(features.begin(), features.end(), [](const Feature& a, const Feature& b) {
        return a.score > b.score;
    });

    // Draw circles and text around the top N features
    for (size_t i = 0; i < std::min<size_t>(features.size(), maxNumFeatures); i++) {
        cv::circle(imgout, features[i].point, 5, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::putText(imgout, std::to_string(i + 1), features[i].point, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    // Print the sorted list of features
    std::cout << "Top " << maxNumFeatures << " Harris features:\n";
    for (size_t i = 0; i < std::min<size_t>(features.size(), maxNumFeatures); i++) {
        std::cout << "Index: " << i + 1 << ", Position: (" << features[i].point.x << ", " << features[i].point.y << "), Score: " << features[i].score << "\n";
    }

    return imgout;
}

cv::Mat detectAndDrawShiAndTomasi(const cv::Mat & img, int maxNumFeatures)
{
    // TODO
    // Copy the input image to draw features on
    cv::Mat imgout = img.clone();

    // Shi-Tomasi corner detection parameters
    float shiTomasiThreshold = 60.0; // Threshold for detecting corners
    int blockSize = 2;                // Neighborhood size
    int apertureSize = 3;             // Aperture parameter for the Sobel operator
    double k = 0.04;                  // Harris detector free parameter (not used here but usually included in Harris)

    // Convert to grayscale if necessary
    cv::Mat imgGrayScale;
    if (img.channels() == 1) {
        imgGrayScale = img;
    } else {
        cv::cvtColor(img, imgGrayScale, cv::COLOR_BGR2GRAY);
    }

    // Compute corner response using the minimum eigenvalue
    cv::Mat eigenVal;
    cv::cornerMinEigenVal(imgGrayScale, eigenVal, blockSize);

    // Normalize and convert to absolute values
    cv::Mat eigenValNorm, eigenValScaled;
    cv::normalize(eigenVal, eigenValNorm, 0, 255, cv::NORM_MINMAX, CV_32FC1);
    cv::convertScaleAbs(eigenValNorm, eigenValScaled);

    // Collect features above the threshold
    std::vector<Feature> features;
    for (int i = 0; i < eigenValNorm.rows; i++) {
        for (int j = 0; j < eigenValNorm.cols; j++) {
            float score = eigenValNorm.at<float>(i, j);
            if (score > shiTomasiThreshold) {
                features.push_back({cv::Point2f(j, i), score});
            }
        }
    }

    // Draw orange circles around all features above the threshhold
    for (size_t i = 0; i < features.size(); i++) {
        cv::circle(imgout, features[i].point, 2, cv::Scalar(0,165,255), 2, cv::LINE_AA);
    }

    // Sort features by score in descending order
    std::sort(features.begin(), features.end(), [](const Feature& a, const Feature& b) {
        return a.score > b.score;
    });

    // Draw circles and text around the top N features
    for (size_t i = 0; i < std::min<size_t>(features.size(), maxNumFeatures); i++) {
        cv::circle(imgout, features[i].point, 5, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::putText(imgout, std::to_string(i + 1), features[i].point, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    // Print the sorted list of features
    std::cout << "Top " << maxNumFeatures << " Shi-Tomasi features:\n";
    for (size_t i = 0; i < std::min<size_t>(features.size(), maxNumFeatures); i++) {
        std::cout << "Index: " << i + 1 << ", Position: (" << features[i].point.x << ", " << features[i].point.y << "), Score: " << features[i].score << "\n";
    }
    return imgout;
}

cv::Mat detectAndDrawFAST(const cv::Mat & img, int maxNumFeatures)
{
    // Copy the input image to draw features on
    cv::Mat imgout = img.clone();

    // FAST corner detection parameters
    int threshold = 110;  // Threshold for corner detection
    bool nonmaxSuppression = true;  // Apply non-max suppression

    // Convert to grayscale if necessary
    cv::Mat imgGrayScale;
    if (img.channels() == 1) {
        imgGrayScale = img;
    } else {
        cv::cvtColor(img, imgGrayScale, cv::COLOR_BGR2GRAY);
    }

    // Create a FAST feature detector
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(threshold, nonmaxSuppression);

    // Detect keypoints
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(imgGrayScale, keypoints);

    // Collect features and their scores using keypoint.response
    std::vector<Feature> features;
    for (const auto& keypoint : keypoints) {
        features.push_back({keypoint.pt, keypoint.response});
    }

    // Draw orange circles around all features above the threshold
    for (size_t i = 0; i < features.size(); i++) {
        cv::circle(imgout, features[i].point, 2, cv::Scalar(0,165,255), 2, cv::LINE_AA);
    }

    // Sort features by score (keypoint.response) in descending order
    std::sort(features.begin(), features.end(), [](const Feature& a, const Feature& b) {
        return a.score > b.score;
    });

    // Draw circles and text around the top N features
    for (size_t i = 0; i < std::min<size_t>(features.size(), maxNumFeatures); i++) {
        cv::circle(imgout, features[i].point, 5, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::putText(imgout, std::to_string(i + 1), features[i].point, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    // Print the sorted list of features
    std::cout << "Top " << maxNumFeatures << " FAST features:\n";
    for (size_t i = 0; i < std::min<size_t>(features.size(), maxNumFeatures); i++) {
        std::cout << "Index: " << i + 1 << ", Position: (" << features[i].point.x << ", " << features[i].point.y << "), Score: " << features[i].score << "\n";
    }

    return imgout;
}


cv::Mat detectAndDrawArUco(const cv::Mat & img, int maxNumFeatures)
{
    // TODO
    // Copy the input image to draw features on
    cv::Mat imgout = img.clone();

    // Define the ArUco dictionary
    cv::aruco::DetectorParameters parameters = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    // Vectors to hold the detected markers and their corners
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;

    // Detect markers
    cv::aruco::ArucoDetector detector(dictionary, parameters);
    detector.detectMarkers(img, markerCorners, markerIds, rejectedCandidates);

    // Draw detected markers on the output image
    cv::aruco::drawDetectedMarkers(imgout, markerCorners, markerIds);

    // Create a list of markers with their corner positions and sort them by their IDs
    std::vector<std::pair<int, std::vector<cv::Point2f>>> markers;
    for (size_t i = 0; i < markerIds.size(); i++) {
        markers.push_back({markerIds[i], markerCorners[i]});
    }
    
    // Sort markers by their IDs
    std::sort(markers.begin(), markers.end(), [](const std::pair<int, std::vector<cv::Point2f>>& a, const std::pair<int, std::vector<cv::Point2f>>& b) {
        return a.first < b.first;
    });

    // Print sorted marker corner locations
    std::cout << "Detected ArUco markers:\n";
    for (const auto& marker : markers) {
        std::cout << "Marker ID: " << marker.first << "\n";
        std::cout << "Corners:\n";
        for (const auto& corner : marker.second) {
            std::cout << "  (" << corner.x << ", " << corner.y << ")\n";
        }
    }
    return imgout;
}
