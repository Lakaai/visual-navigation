#ifndef MEASUREMENTOUTDOORFLOWBUNDLE_H
#define MEASUREMENTOUTDOORFLOWBUNDLE_H

#include <vector>
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include "SystemEstimator.h"
#include "Pose.hpp"
#include "Camera.h"
#include "Measurement.h"

class MeasurementOutdoorFlowBundle : public Measurement
{
public:
    MeasurementOutdoorFlowBundle(double time, const Camera & camera, const cv::Mat & imgk_raw, const cv::Mat & imgkm1_raw, const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQOikm1);
    virtual Eigen::VectorXd simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const override;

    // Helper functions for log likelihood and visualisation
    template <typename Scalar> Eigen::Matrix<Scalar, 3, Eigen::Dynamic> predictFlowImpl(const Eigen::VectorX<Scalar> & x, const Eigen::Matrix<double, 3, Eigen::Dynamic> & pkm1, const Eigen::Matrix<double, 3, Eigen::Dynamic> & pk) const;
    template <typename Scalar> Scalar logLikelihoodImpl(const Eigen::VectorX<Scalar> & x) const;
    Eigen::Matrix<double, 2, Eigen::Dynamic> predictedFeatures(const Eigen::VectorXd & x, const SystemEstimator & system) const;

    // Note: costOdometry is used only in Lab 11.
    //       Assignment 2 uses costJointDensity instead.
    template <typename Scalar> Scalar costOdometryImpl(const Eigen::VectorX<Scalar> & etak, const Eigen::VectorXd & etakm1) const;
    double costOdometry(const Eigen::VectorXd & etak, const Eigen::VectorXd & etakm1) const;
    double costOdometry(const Eigen::VectorXd & etak, const Eigen::VectorXd & etakm1, Eigen::VectorXd & g) const;
    double costOdometry(const Eigen::VectorXd & etak, const Eigen::VectorXd & etakm1, Eigen::VectorXd & g, Eigen::MatrixXd & H) const;

    const Eigen::Matrix<double, 2, Eigen::Dynamic> & trackedPreviousFeatures() const;
    const Eigen::Matrix<double, 2, Eigen::Dynamic> & trackedCurrentFeatures() const;
    const std::vector<unsigned char> & inlierMask() const;
protected:
    const Camera & camera_;

    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOikm1_;      // Measured features for previous frame
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOik_;        // Measured features for current frame

    Eigen::Matrix<double, 2, Eigen::Dynamic> rQbarOikm1_;   // Undistorted features for previous frame
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQbarOik_;     // Undistorted features for current frame

    std::vector<unsigned char> mask_;                       // Inlier mask

    Eigen::Matrix<double, 3, Eigen::Dynamic> pkm1_;         // Inlier undistorted homogeneous points in previous frame
    Eigen::Matrix<double, 3, Eigen::Dynamic> pk_;           // Inlier undistorted homogeneous points in current frame

    double sigma_;                                          // Feature error standard deviation (in pixels)
};

template <typename Scalar>
Eigen::Matrix<Scalar, 3, Eigen::Dynamic> MeasurementOutdoorFlowBundle::predictFlowImpl(const Eigen::VectorX<Scalar> & x, const Eigen::Matrix<double, 3, Eigen::Dynamic> & pkm1, const Eigen::Matrix<double, 3, Eigen::Dynamic> & pk) const
{
    assert(x.rows() >= 18);
    assert(x.cols() == 1);
    assert(pkm1.cols() == pk.cols());
    
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> pk_hat(3, pkm1.cols());

    // Extract η_k and η_{k-1} from the state vector x
    // η represents the body pose (position and orientation) in the navigation frame
    Eigen::Matrix<Scalar, 6, 1> eta_k = x.template segment<6>(6);
    Eigen::Matrix<Scalar, 6, 1> eta_km1 = x.template segment<6>(12);

    // Extract position and orientation of the body from η_k and η_{k-1}
    Eigen::Matrix<Scalar, 3, 1> rBNnk = eta_k.template head<3>();
    Eigen::Matrix<Scalar, 3, 1> rBNnkm1 = eta_km1.template head<3>();
    Eigen::Matrix<Scalar, 3, 3> Rnbk = rpy2rot(eta_k.template tail<3>());
    Eigen::Matrix<Scalar, 3, 3> Rnbkm1 = rpy2rot(eta_km1.template tail<3>());

    // Manually cast Tbc to Scalar
    Eigen::Matrix<Scalar, 3, 3> Rbc = camera_.Tbc.rotationMatrix.template cast<Scalar>();
    Eigen::Matrix<Scalar, 3, 1> rCBb = camera_.Tbc.translationVector.template cast<Scalar>();

    // Compute camera poses
    Eigen::Matrix<Scalar, 3, 1> rCNnk = rBNnk + Rnbk * rCBb;
    Eigen::Matrix<Scalar, 3, 1> rCNnkm1 = rBNnkm1 + Rnbkm1 * rCBb;
    Eigen::Matrix<Scalar, 3, 3> Rnck = Rnbk * Rbc;
    Eigen::Matrix<Scalar, 3, 3> Rnckm1 = Rnbkm1 * Rbc;

    // Convert cv::Mat to Eigen::Matrix and cast to Scalar
    Eigen::Matrix<Scalar, 3, 3> K = Eigen::Matrix<Scalar, 3, 3>::Zero();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            K(i, j) = static_cast<Scalar>(camera_.cameraMatrix.at<double>(i, j));
        }
    }

    // Compute homographies
    Eigen::Matrix<Scalar, 3, 3> H_z = K * Rnck.transpose() * 
        (Eigen::Matrix<Scalar, 3, 3>::Identity() - (rCNnkm1 - rCNnk) * Eigen::Matrix<Scalar, 3, 1>::UnitZ().transpose() / rCNnkm1.z()) * 
        Rnckm1 * K.inverse();
    Eigen::Matrix<Scalar, 3, 3> H_inf = K * Rnck.transpose() * Rnckm1 * K.inverse();

    // Predict feature locations
    for (Eigen::Index i = 0; i < pkm1.cols(); ++i)
    {
        Eigen::Matrix<Scalar, 3, 1> p_km1 = pkm1.col(i).template cast<Scalar>();
        Eigen::Matrix<Scalar, 3, 1> p_k = pk.col(i).template cast<Scalar>();

        // Determine if the point is above or below the horizon
        Scalar z_proj = (Rnck * K.inverse() * p_k).z();

        if (z_proj > 0)
        {
            // Point is below horizon, use ground plane homography
            pk_hat.col(i) = H_z * p_km1;
        }
        else
        {
            // Point is above horizon, use infinity homography
            pk_hat.col(i) = H_inf * p_km1;
        }
    }

    return pk_hat;
}

template <typename Scalar>
Scalar MeasurementOutdoorFlowBundle::logLikelihoodImpl(const Eigen::VectorX<Scalar> & x) const
{
    assert(pkm1_.cols() == pk_.cols());

    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> pk_hat = predictFlowImpl(x, pkm1_, pk_);
    Eigen::Matrix<Scalar, 2, Eigen::Dynamic> rQbarOik_hat = pk_hat.template topRows<2>().array().rowwise()/pk_hat.row(2).array();
    
    // TODO: Lab 11
    // Calculate the error between predicted and actual flow
    Eigen::Matrix<Scalar, 2, Eigen::Dynamic> error = rQbarOik_hat - pk_.template topRows<2>().cast<Scalar>();

    // Print error statistics (only for double type to avoid autodiff prints)
    
        double mean_error = 0;
        double max_error = 0;
        int large_errors = 0;
        
        for (int i = 0; i < error.cols(); ++i) {
            // Explicitly cast to double
            double err_x = static_cast<double>(error.col(i)[0]);
            double err_y = static_cast<double>(error.col(i)[1]);
            double err_magnitude = std::sqrt(err_x*err_x + err_y*err_y);
            
            mean_error += err_magnitude;
            max_error = std::max(max_error, err_magnitude);
            if (err_magnitude > 5.0) large_errors++;
        }
        mean_error /= error.cols();
    
    // Calculate the log-likelihood
    Scalar logLik = 0;
    const int n = error.cols(); // Number of flow vectors
    const Scalar logSqrt2Pi = std::log(std::sqrt(2 * M_PI));

    for (int i = 0; i < n; ++i)
    {
        // Mahalanobis distance for each flow vector
        Scalar mahalanobis_dist = error.col(i).squaredNorm() / (sigma_ * sigma_);
        
        // Log-likelihood contribution of this flow vector
        logLik += -logSqrt2Pi - std::log(sigma_) - 0.5 * mahalanobis_dist;
    }

    return logLik;
}

// Note: costOdometry is used only in Lab 11, not Assignment 2.
template <typename Scalar>
Scalar MeasurementOutdoorFlowBundle::costOdometryImpl(const Eigen::VectorX<Scalar> & etak, const Eigen::VectorXd & etakm1) const
{
    Eigen::VectorX<Scalar> x(18);
    x.template segment<6>(6) = etak;
    x.template segment<6>(12) = etakm1;
    return -logLikelihoodImpl(x);
}

#endif
