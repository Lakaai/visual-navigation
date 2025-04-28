#include <cstddef>
#include <iostream>
#include <numeric>
#include <vector>
#include <algorithm>
#include <Eigen/Core>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include "GaussianInfo.hpp"
#include "rotation.hpp"
#include "SystemEstimator.h"
#include "MeasurementOutdoorFlowBundle.h"

MeasurementOutdoorFlowBundle::MeasurementOutdoorFlowBundle(double time, const Camera & camera, const cv::Mat & imgk_raw, const cv::Mat & imgkm1_raw, const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQOikm1)
    : Measurement(time)
    , camera_(camera)
    , rQOikm1_(rQOikm1)
    , rQOik_()
    , rQbarOikm1_()
    , rQbarOik_()
    , mask_()
    , pkm1_()
    , pk_()
    , sigma_(5) 
{
    // Parameters maxNumFeatures = 
    const int divisor = 2;
    const int maxNumFeatures = 2000; // 1200 
    const int minNumFeatures = 1600; // 900 minDist 50 1e-4 qualityLevel

    cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS, 30, 0.01);
    cv::Size subPixWinSize(11, 11);
    cv::Size winSize(21, 21);

    // Convert images to grayscale
    cv::Mat imgk_gray, imgkm1_gray;
    cv::cvtColor(imgk_raw, imgk_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgkm1_raw, imgkm1_gray, cv::COLOR_BGR2GRAY);

    // Scale images
    cv::Mat imgk_scaled, imgkm1_scaled;
    cv::resize(imgk_gray, imgk_scaled, cv::Size(), 1.0/divisor, 1.0/divisor);
    cv::resize(imgkm1_gray, imgkm1_scaled, cv::Size(), 1.0/divisor, 1.0/divisor);

    std::cout << "rQOikm1_ size: " << rQOikm1_.cols() << std::endl;

    if (rQOikm1_.cols() == 0)
    {
        // First frame: Initialize features
        std::cout << "First frame: Initializing features" << std::endl;
        std::vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(imgk_scaled, corners, maxNumFeatures, 1e-4, 50, cv::Mat(), 5, false, 0.04);
        cv::cornerSubPix(imgk_scaled, corners, subPixWinSize, cv::Size(-1,-1), termcrit);

        rQOik_.resize(2, corners.size());
        for (size_t i = 0; i < corners.size(); ++i)
        {
            rQOik_(0, i) = corners[i].x * divisor;
            rQOik_(1, i) = corners[i].y * divisor;
        }

        // For the first frame, set rQOikm1_ equal to rQOik_
        rQOikm1_ = rQOik_;

        std::cout << "Initialized " << rQOik_.cols() << " features" << std::endl;
    }
    
    else
    {
        // Subsequent frames: Track features
        std::vector<cv::Point2f> rQOikm1_scaled(rQOikm1_.cols());
        for (int j = 0; j < rQOikm1_.cols(); ++j)
        {
            rQOikm1_scaled[j].x = rQOikm1_(0, j) / divisor;
            rQOikm1_scaled[j].y = rQOikm1_(1, j) / divisor;
        }

        std::vector<cv::Point2f> rQOik_scaled;
        std::vector<uchar> status;
        std::vector<float> err;

        cv::calcOpticalFlowPyrLK(imgkm1_scaled, imgk_scaled, rQOikm1_scaled, rQOik_scaled, status, err, winSize, 3, termcrit, 0, 0.001);

        // Keep points that have been matched between both frames
        int np = 0;
        rQOik_.resize(2, status.size());
        for (size_t i = 0; i < status.size(); ++i)
        {
            if (status[i])
            {
                rQOik_(0, np) = rQOik_scaled[i].x * divisor;
                rQOik_(1, np) = rQOik_scaled[i].y * divisor;
                rQOikm1_.col(np) = rQOikm1_.col(i);
                ++np;
            }
        }
        rQOik_.conservativeResize(2, np);
        rQOikm1_.conservativeResize(2, np);

        std::cout << "Tracked " << np << " features" << std::endl;

        // If we've lost too many features, detect new ones
        if (np < minNumFeatures)
        {
            std::cout << "Detecting new features" << std::endl;
            std::vector<cv::Point2f> new_corners;
            cv::goodFeaturesToTrack(imgk_scaled, new_corners, maxNumFeatures - np, 1e-4, 50, cv::Mat(), 5, false, 0.04);
            cv::cornerSubPix(imgk_scaled, new_corners, subPixWinSize, cv::Size(-1,-1), termcrit);       
            int new_features = new_corners.size();
            rQOik_.conservativeResize(2, np + new_features);
            rQOikm1_.conservativeResize(2, np + new_features);
            for (int i = 0; i < new_features; ++i)
            {
                rQOik_(0, np + i) = new_corners[i].x * divisor;
                rQOik_(1, np + i) = new_corners[i].y * divisor;
                rQOikm1_.col(np + i) = rQOik_.col(np + i);  // New features are assumed to be stationary for their first frame
            }

            std::cout << "Added " << new_features << " new features" << std::endl;
        }
    }

    // Calculate undistorted feature locations
    rQbarOik_ = camera_.undistort(rQOik_);
    rQbarOikm1_ = camera_.undistort(rQOikm1_);

    // Use RANSAC to find fundamental matrix and determine inliers
    std::vector<cv::Point2f> points1, points2;
    for (int i = 0; i < rQbarOikm1_.cols(); ++i)
    {
        points1.push_back(cv::Point2f(rQbarOikm1_(0, i), rQbarOikm1_(1, i)));
        points2.push_back(cv::Point2f(rQbarOik_(0, i), rQbarOik_(1, i)));
    }

    cv::Mat F = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 1.0, 0.99, mask_);

    // Keep only the inliers
    int nInliers = std::count(mask_.begin(), mask_.end(), 1);
    pkm1_ = Eigen::MatrixXd::Ones(3, nInliers);
    pk_ = Eigen::MatrixXd::Ones(3, nInliers);
    int inlierIdx = 0;
    for (size_t i = 0; i < mask_.size(); ++i)
    {
        if (mask_[i])
        {
            pkm1_.block<2,1>(0, inlierIdx) = rQbarOikm1_.col(i);
            pk_.block<2,1>(0, inlierIdx) = rQbarOik_.col(i);
            ++inlierIdx;
        }
    }

    std::cout << "Inliers: " << nInliers << ", Outliers: " << mask_.size() - nInliers << std::endl;
}

Eigen::VectorXd MeasurementOutdoorFlowBundle::simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    Eigen::VectorXd y;
    throw std::runtime_error("Not implemented");
    return y;
}

const Eigen::Matrix<double, 2, Eigen::Dynamic> & MeasurementOutdoorFlowBundle::trackedPreviousFeatures() const
{
    return rQOikm1_;
}

const Eigen::Matrix<double, 2, Eigen::Dynamic> & MeasurementOutdoorFlowBundle::trackedCurrentFeatures() const
{
    return rQOik_;
}

const std::vector<unsigned char> & MeasurementOutdoorFlowBundle::inlierMask() const
{
    return mask_;
}

double MeasurementOutdoorFlowBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    return logLikelihood<double>(x, system);
}

double MeasurementOutdoorFlowBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const
{
    // Evaluate gradient for Newton and quasi-Newton methods

    using autodiff::dual;
    using autodiff::gradient;

    // Convert input to autodiff type
    Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();

    // Compute gradient and function value using autodiff
    autodiff::dual fdual;
    g = gradient(&MeasurementOutdoorFlowBundle::logLikelihood<autodiff::dual>, autodiff::wrt(x_dual), autodiff::at(this, x_dual, system), fdual);
    
    return logLikelihood(x, system);
}


double MeasurementOutdoorFlowBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    // Evaluate Hessian for Newton method
    H.resize(x.size(), x.size());
    H.setZero();
    // TODO: Assignment 2
    return logLikelihood(x, system, g);
}

Eigen::Matrix<double, 2, Eigen::Dynamic> MeasurementOutdoorFlowBundle::predictedFeatures(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    std::size_t np = rQOik_.cols();
    
    // Initialize homogeneous points matrices with undistorted points
    Eigen::Matrix<double, 3, Eigen::Dynamic> pkm1(3, np);
    Eigen::Matrix<double, 3, Eigen::Dynamic> pk(3, np);
    
    // Use undistorted points
    pkm1.topRows<2>() = rQbarOikm1_;
    pkm1.row(2).setOnes();
    pk.topRows<2>() = rQbarOik_;
    pk.row(2).setOnes();
    
    // Initialize matrix for predictions
    Eigen::Matrix<double, 3, Eigen::Dynamic> pk_hat(3, np);
    
    // Get predictions
    for(size_t i = 0; i < np; ++i) {
        pk_hat.col(i) = predict(x, pkm1.col(i), pk.col(i));
    }

    // Convert to Euclidean coordinates
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQbarOik_hat = pk_hat.topRows<2>().array().rowwise()/pk_hat.row(2).array();
    
    
    // Apply distortion to get back to distorted image coordinates
    return camera_.distort(rQbarOik_hat);
}
