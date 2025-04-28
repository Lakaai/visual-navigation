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
    , sigma_(2.0)
{
    const int divisor = 1;
    const int maxNumFeatures = 1000;
    const int minNumFeatures = 700;

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
        cv::goodFeaturesToTrack(imgk_scaled, corners, maxNumFeatures, 1e-4, 50, cv::Mat(), 3, false, 0.04);
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
            cv::goodFeaturesToTrack(imgk_scaled, new_corners, maxNumFeatures - np, 1e-4, 50, cv::Mat(), 3, false, 0.04);
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

// MeasurementOutdoorFlowBundle::MeasurementOutdoorFlowBundle(double time, const Camera & camera, const cv::Mat & imgk_raw, const cv::Mat & imgkm1_raw, const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQOikm1)
//     : Measurement(time)
//     , camera_(camera)
//     , rQOikm1_(rQOikm1)
//     , rQOik_()
//     , rQbarOikm1_()
//     , rQbarOik_()
//     , mask_()
//     , pkm1_()
//     , pk_()
//     , sigma_(2.0) // TODO: Assignment(s)
// {
//     // TODO: Assignment(s)
//     // updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;
//     // updateMethod_ = UpdateMethod::BFGSTRUSTSQRT;

//     // TODO: Lab 11
//     const int divisor               = 1;                // Image scaling factor
//     const int maxNumFeatures        = 1000;                // Maximum number of features per frame
//     const int minNumFeatures        = 700;                // Minimum number of feature per frame

//     cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,30,0.01);
//     cv::Size subPixWinSize(11, 11);     // Window size for subpixel refinement
//     cv::Size winSize(21, 21);           // Window size for optical flow

//     // Convert images to grayscale
//     cv::Mat imgk_gray;
//     cv::Mat imgkm1_gray;
//     cv::cvtColor(imgk_raw, imgk_gray, cv::COLOR_BGR2GRAY);
//     cv::cvtColor(imgkm1_raw, imgkm1_gray, cv::COLOR_BGR2GRAY);

//     // Scale images
//     cv::Mat imgk_scaled;
//     cv::Mat imgkm1_scaled;
//     cv::resize(imgk_gray, imgk_scaled, cv::Size(), 1.0/divisor, 1.0/divisor);
//     cv::resize(imgkm1_gray, imgkm1_scaled, cv::Size(), 1.0/divisor, 1.0/divisor);

//     // Variables for feature tracking
//     std::vector<uchar> status;
//     std::vector<float> err;

//     // Parameters for goodFeaturesToTrack
//     double qualityLevel = 0.01;
//     double minDistance = 10;
//     int blockSize = 3;
//     bool useHarrisDetector = false;
//     double k = 0.04;

//     std::vector<cv::Point2f> rQOikm1_scaled;
//     std::cout << "rQOikm1_ size: " << rQOikm1_.cols() << std::endl;

//     if (rQOikm1_.cols() < minNumFeatures)
//     {
//         std::cout << "Initializing new features" << std::endl;
//         std::vector<cv::Point2f> corners;
//         cv::goodFeaturesToTrack(imgk_scaled, corners, maxNumFeatures, 0.01, 10, cv::Mat(), 3, false, 0.04);
//         std::cout << "Number of corners detected: " << corners.size() << std::endl;

//         cv::cornerSubPix(imgk_scaled, corners, subPixWinSize, cv::Size(-1,-1), termcrit);
//         std::cout << "Number of corners detected after subpix: " << corners.size() << std::endl;


//         rQOik_.resize(2, corners.size());
//         for (size_t i = 0; i < corners.size(); ++i)
//         {
//             rQOik_(0, i) = corners[i].x * divisor;
//             rQOik_(1, i) = corners[i].y * divisor;
//         }
//     }
//     else
//     {
//         rQOikm1_scaled.resize(rQOikm1_.cols());
//         for (int j = 0; j < rQOikm1_.cols(); ++j)
//         {
//             rQOikm1_scaled[j].x = rQOikm1_(0, j) / divisor;
//             rQOikm1_scaled[j].y = rQOikm1_(1, j) / divisor;
//             std::cout << "rQOikm1_scaled size after initialization: " << rQOikm1_scaled.size() << std::endl;
//         }
//     }

//     std::cout << "rQOikm1_scaled size: " << rQOikm1_scaled.size() << std::endl;

//     // Track features from frame k-1 to frame k using the scaled images and previous points
//     std::vector<cv::Point2f> rQOik_scaled;
//     // TODO: Lab 11
//     // cv::calcOpticalFlowPyrLK(imgkm1_scaled, imgk_scaled, rQOikm1_scaled, rQOik_scaled, status, err);
//      cv::calcOpticalFlowPyrLK(imgkm1_scaled, imgk_scaled, rQOikm1_scaled, rQOik_scaled, status, err, winSize, 3, termcrit, 0, 0.001);

//     // Keep points that have been matched between both frames
//     int np = 0;
//     rQOik_.resize(2, status.size());
//     rQOikm1_.resize(2, status.size());
//     for (size_t i = 0; i < status.size(); ++i)
//     {
//         if (status[i])
//         {
//             rQOik_(0, np) = rQOik_scaled[i].x * divisor;
//             rQOik_(1, np) = rQOik_scaled[i].y * divisor;
//             rQOikm1_(0, np) = rQOikm1_scaled[i].x * divisor;
//             rQOikm1_(1, np) = rQOikm1_scaled[i].y * divisor;
//             ++np;
//         }
//     }
//     rQOik_.conservativeResize(2, np);
//     rQOikm1_.conservativeResize(2, np);
//     // TODO: Lab 11
//     std::cout << "After filtering by status, there are " << np << " associations." << std::endl;

//     // Calculate where flow points would be in the unscaled image
//     rQOik_.resize(2, np);
//     rQOikm1_.resize(2, np);
//     for (int j = 0; j < np; ++j)
//     {
//         rQOik_(0, j) = rQOik_scaled[j].x*divisor;
//         rQOik_(1, j) = rQOik_scaled[j].y*divisor;

//         rQOikm1_(0, j) = rQOikm1_scaled[j].x*divisor;
//         rQOikm1_(1, j) = rQOikm1_scaled[j].y*divisor;
//     }

//     // Calculate the undistorted location of features
//     rQbarOik_.resize(2, np);
//     rQbarOikm1_.resize(2, np);

//     // TODO: Lab 11
//     // Use RANSAC to find fundamental matrix and determine inliers (mask_)
//     std::vector<cv::Point2f> points1, points2;
//     for (int i = 0; i < np; ++i)
//     {
//         points1.push_back(cv::Point2f(rQbarOikm1_(0, i), rQbarOikm1_(1, i)));
//         points2.push_back(cv::Point2f(rQbarOik_(0, i), rQbarOik_(1, i)));
//     }

//     // Note: We don't actually use the fundamental matrix computed here, it is just used to
//     //       determine which undistorted flow vectors are consistent with the epipolar constraint.

//     // TODO: Lab 11
//     cv::Mat F = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 1.0, 0.99, mask_);
//     int nInliers = std::count(mask_.begin(), mask_.end(), true);
//     std::cout<< "No. inliers = " << nInliers <<  ", No. outliers  = " << mask_.size() - nInliers <<  std::endl;

//     // Inlier undistorted homogeneous points (pkm1_ and pk_)
//     pk_     = Eigen::MatrixXd::Ones(3, nInliers);
//     pkm1_   = Eigen::MatrixXd::Ones(3, nInliers);
//     // TODO: Lab 11
//     int inlierIndex = 0;
//     for (size_t i = 0; i < mask_.size(); ++i)
//     {
//         if (mask_[i])
//         {
//             pk_.block<2, 1>(0, inlierIndex) = rQbarOik_.col(i);
//             pkm1_.block<2, 1>(0, inlierIndex) = rQbarOikm1_.col(i);
//             ++inlierIndex;
//         }
//     }
// }

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
    return logLikelihoodImpl(x);
}

double MeasurementOutdoorFlowBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const
{
    // Evaluate gradient for Newton and quasi-Newton methods
    g.resize(x.size());
    g.setZero();
    // TODO: Assignment 2
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

    // Predict undistorted homogeneous image points in current frame
    Eigen::Matrix<double, 3, Eigen::Dynamic> pk(3, np);
    Eigen::Matrix<double, 3, Eigen::Dynamic> pkm1(3, np);
    pkm1.topRows<2>() = rQbarOikm1_;
    pkm1.row(2).setOnes();
    pk.topRows<2>() = rQbarOik_;
    pk.row(2).setOnes();

    Eigen::Matrix<double, 3, Eigen::Dynamic> pk_hat = predictFlowImpl(x, pkm1, pk);
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQbarOik_hat = pk_hat.topRows<2>().array().rowwise()/pk_hat.row(2).array();
    assert(rQbarOik_hat.cols() == np);
    
    // Compute image coordinates (with lens distortion)
    // double fx = camera_.cameraMatrix.at<double>( 0,  0);
    // double fy = camera_.cameraMatrix.at<double>( 1,  1);
    // double cx = camera_.cameraMatrix.at<double>( 0,  2);
    // double cy = camera_.cameraMatrix.at<double>( 1,  2);
    
    // TODO: Lab 11
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOik_hat = camera_.distort(rQbarOik_hat);
    return rQOik_hat;
}

// Note: costOdometry is used only in Lab 11, not Assignment 2.
double MeasurementOutdoorFlowBundle::costOdometry(const Eigen::VectorXd & etak, const Eigen::VectorXd & etakm1) const
{
    return costOdometryImpl(etak, etakm1);
}

double MeasurementOutdoorFlowBundle::costOdometry(const Eigen::VectorXd & etak, const Eigen::VectorXd & etakm1, Eigen::VectorXd & g) const
{
    // Forward-mode autodifferentiation
    Eigen::Matrix<autodiff::dual, Eigen::Dynamic, 1> etakdual = etak.cast<autodiff::dual>();
    autodiff::dual fdual;
    g = gradient(&MeasurementOutdoorFlowBundle::costOdometryImpl<autodiff::dual>, wrt(etakdual), at(this, etakdual, etakm1), fdual);
    return val(fdual);
}

double MeasurementOutdoorFlowBundle::costOdometry(const Eigen::VectorXd & etak, const Eigen::VectorXd & etakm1, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    // Forward-mode autodifferentiation
    Eigen::Matrix<autodiff::dual2nd, Eigen::Dynamic, 1> etakdual = etak.cast<autodiff::dual2nd>();
    autodiff::dual2nd fdual;
    H = hessian(&MeasurementOutdoorFlowBundle::costOdometryImpl<autodiff::dual2nd>, wrt(etakdual), at(this, etakdual, etakm1), fdual, g);
    return val(fdual);
}

