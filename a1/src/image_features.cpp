#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include "image_features.h"

PointFeature::PointFeature()
    : score(0)
    , x(0)
    , y(0)
{}

PointFeature::PointFeature(const double & score_, const double & x_, const double & y_)
    : score(score_)
    , x(x_)
    , y(y_)
{}

bool PointFeature::operator<(const PointFeature & other) const
{
    return (score > other.score);
}

std::vector<PointFeature> detectFeatures(const cv::Mat & img, const int & maxNumFeatures)
{
    std::vector<PointFeature> features;
    
    // Convert image to grayscale if it's not already
    cv::Mat gray;
    if (img.channels() == 3)
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img.clone();

    // Detect features using Shi-Tomasi corner detector
    std::vector<cv::Point2f> corners;
    std::vector<float> quality_scores;
    cv::goodFeaturesToTrack(gray, corners, maxNumFeatures, 0.02, 10, cv::Mat(), 3, true, 0.04);

    // Get quality scores for sorting
    cv::Mat quality_map;
    cv::cornerMinEigenVal(gray, quality_map, 3, 3);

    // Create PointFeature objects
    features.reserve(corners.size());
    for (size_t i = 0; i < corners.size(); ++i)
    {
        float score = quality_map.at<float>(corners[i].y, corners[i].x);
        features.emplace_back(score, corners[i].x, corners[i].y);
    }

    // Sort features by score (descending order)
    std::sort(features.begin(), features.end());

    // Cap number of features to maxNumFeatures
    if (features.size() > static_cast<size_t>(maxNumFeatures))
    {
        features.resize(maxNumFeatures);
    }

    return features;
}