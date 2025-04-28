#ifndef MEASUREMENT_OUTDOOR_FLOW_BUNDLE_H
#define MEASUREMENT_OUTDOOR_FLOW_BUNDLE_H

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
    // template <typename Scalar> Eigen::Matrix<Scalar, 3, Eigen::Dynamic> predictFlow(const Eigen::VectorX<Scalar> & x, const Eigen::Matrix<double, 3, Eigen::Dynamic> & pkm1, const Eigen::Matrix<double, 3, Eigen::Dynamic> & pk) const;
    // template <typename Scalar> Scalar logLikelihood(const Eigen::VectorX<Scalar> & x, const SystemEstimator & system) const;
    Eigen::Matrix<double, 2, Eigen::Dynamic> predictedFeatures(const Eigen::VectorXd & x, const SystemEstimator & system) const;
    //GaussianInfo<double> predictDensity(const Eigen::VectorXd & x, const SystemEstimator & system) const;
    // Helper functions for log likelihood and visualisation
    // template <typename Scalar> Scalar predict(const Eigen::VectorX<Scalar> & x, const SystemEstimator & system) const;
    //template <typename Scalar> Eigen::Matrix<Scalar, 3, Eigen::Dynamic> predict(const Eigen::VectorX<Scalar> & x, const Eigen::Matrix<double, 3, Eigen::Dynamic> & pkm1, const Eigen::Matrix<double, 3, Eigen::Dynamic> & pk) const;
    template <typename Scalar> Scalar logLikelihood(const Eigen::VectorX<Scalar> & x, const SystemEstimator & system) const;
    template <typename Scalar> Eigen::Matrix<Scalar, 3, 1> predict(const Eigen::VectorX<Scalar> & x, const Eigen::Vector3d & pkm1_i, const Eigen::Vector3d & pk_i) const;
    template <typename Scalar> GaussianInfo<Scalar> predictDensity(const Eigen::VectorX<Scalar> & x, const SystemEstimator & system, int i) const;
    template <typename Scalar> GaussianInfo<Scalar> noiseDensity(const SystemEstimator & system) const;
    const Eigen::Matrix<double, 2, Eigen::Dynamic> & trackedPreviousFeatures() const;
    const Eigen::Matrix<double, 2, Eigen::Dynamic> & trackedCurrentFeatures() const;
    Eigen::Matrix<double, 2, Eigen::Dynamic> getPredictedFeatures() const { return predictedFeatures_; }
    const std::vector<unsigned char> & inlierMask() const;
protected:

    const Camera & camera_;
    Eigen::Matrix<double, 2, Eigen::Dynamic> predictedFeatures_;
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOikm1_;      // Measured features for previous frame
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOik_;        // Measured features for current frame

    Eigen::Matrix<double, 2, Eigen::Dynamic> rQbarOikm1_;   // Undistorted features for previous frame
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQbarOik_;     // Undistorted features for current frame

    std::vector<unsigned char> mask_;                       // Inlier mask

    Eigen::Matrix<double, 3, Eigen::Dynamic> pkm1_;         // Inlier undistorted homogeneous points in previous frame
    Eigen::Matrix<double, 3, Eigen::Dynamic> pk_;           // Inlier undistorted homogeneous points in current frame

    double sigma_;                                          // Feature error standard deviation (in pixels)
};

// template <typename Scalar>
// Eigen::Matrix<Scalar, 3, Eigen::Dynamic> MeasurementOutdoorFlowBundle::predict(const Eigen::VectorX<Scalar> & x, const Eigen::Matrix<double, 3, Eigen::Dynamic> & pkm1, const Eigen::Matrix<double, 3, Eigen::Dynamic> & pk) const
// {
//     assert(x.rows() >= 18);
//     assert(x.cols() == 1);
//     assert(pkm1.cols() == pk.cols());
    
//     Eigen::Matrix<Scalar, 3, Eigen::Dynamic> pk_hat(3, pkm1.cols());

//     // Extract η_k and η_{k-1} from the state vector x
//     // η represents the body pose (position and orientation) in the navigation frame
//     Eigen::Matrix<Scalar, 6, 1> eta_k = x.template segment<6>(6);
//     Eigen::Matrix<Scalar, 6, 1> eta_km1 = x.template segment<6>(12);

//     // Extract position and orientation of the body from η_k and η_{k-1}
//     Eigen::Matrix<Scalar, 3, 1> rBNnk = eta_k.template head<3>();
//     Eigen::Matrix<Scalar, 3, 1> rBNnkm1 = eta_km1.template head<3>();
//     Eigen::Matrix<Scalar, 3, 3> Rnbk = rpy2rot(eta_k.template tail<3>());
//     Eigen::Matrix<Scalar, 3, 3> Rnbkm1 = rpy2rot(eta_km1.template tail<3>());

//     // Manually cast Tbc to Scalar
//     Eigen::Matrix<Scalar, 3, 3> Rbc = camera_.Tbc.rotationMatrix.template cast<Scalar>();
//     Eigen::Matrix<Scalar, 3, 1> rCBb = camera_.Tbc.translationVector.template cast<Scalar>();

//     // Compute camera poses
//     Eigen::Matrix<Scalar, 3, 1> rCNnk = rBNnk + Rnbk * rCBb;
//     Eigen::Matrix<Scalar, 3, 1> rCNnkm1 = rBNnkm1 + Rnbkm1 * rCBb;
//     Eigen::Matrix<Scalar, 3, 3> Rnck = Rnbk * Rbc;
//     Eigen::Matrix<Scalar, 3, 3> Rnckm1 = Rnbkm1 * Rbc;

//     // Convert cv::Mat to Eigen::Matrix and cast to Scalar
//     Eigen::Matrix<Scalar, 3, 3> K = Eigen::Matrix<Scalar, 3, 3>::Zero();
//     for (int i = 0; i < 3; ++i) {
//         for (int j = 0; j < 3; ++j) {
//             K(i, j) = static_cast<Scalar>(camera_.cameraMatrix.at<double>(i, j));
//         }
//     }

//     // Compute homographies
//     Eigen::Matrix<Scalar, 3, 3> H_z = K * Rnck.transpose() * 
//         (Eigen::Matrix<Scalar, 3, 3>::Identity() - (rCNnkm1 - rCNnk) * Eigen::Matrix<Scalar, 3, 1>::UnitZ().transpose() / rCNnkm1.z()) * 
//         Rnckm1 * K.inverse();
//     Eigen::Matrix<Scalar, 3, 3> H_inf = K * Rnck.transpose() * Rnckm1 * K.inverse();

//     // Predict feature locations
//     for (Eigen::Index i = 0; i < pkm1.cols(); ++i)
//     {
//         Eigen::Matrix<Scalar, 3, 1> p_km1 = pkm1.col(i).template cast<Scalar>();
//         Eigen::Matrix<Scalar, 3, 1> p_k = pk.col(i).template cast<Scalar>();

//         // Determine if the point is above or below the horizon
//         Scalar z_proj = (Rnck * K.inverse() * p_k).z();

//         if (z_proj > 0)
//         {
//             // Point is below horizon, use ground plane homography
//             pk_hat.col(i) = H_z * p_km1;
//         }
//         else
//         {
//             // Point is above horizon, use infinity homography
//             pk_hat.col(i) = H_inf * p_km1;
//         }
//     }

//     return pk_hat;
// }
template <typename Scalar>
Eigen::Matrix<Scalar, 3, 1> MeasurementOutdoorFlowBundle::predict(const Eigen::VectorX<Scalar> & x, 
                                                                 const Eigen::Vector3d & pkm1_i,
                                                                 const Eigen::Vector3d & pk_i) const
{
    assert(x.rows() >= 18);
    assert(x.cols() == 1);
    
    // Extract η_k and η_{k-1} from the state vector x
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

    // Convert input points to Scalar type
    Eigen::Matrix<Scalar, 3, 1> p_km1 = pkm1_i.template cast<Scalar>();
    Eigen::Matrix<Scalar, 3, 1> p_k = pk_i.template cast<Scalar>();

    // Determine if the point is above or below the horizon
    Scalar z_proj = (Rnck * K.inverse() * p_k).z();

    // Apply appropriate homography
    Eigen::Matrix<Scalar, 3, 1> p_k_hat;
    if (z_proj > 0) {
        // Point is below horizon, use ground plane homography
        p_k_hat = H_z * p_km1;
    } else {
        // Point is above horizon, use infinity homography
        p_k_hat = H_inf * p_km1;
    }

    // Convert to Euclidean coordinates
    // Eigen::Matrix<Scalar, 2, 1> prediction;
    // prediction[0] = p_k_hat[0] / p_k_hat[2];
    // prediction[1] = p_k_hat[1] / p_k_hat[2];

    return p_k_hat;
}

//template <typename Scalar>
// Scalar MeasurementOutdoorFlowBundle::logLikelihood(const Eigen::VectorX<Scalar> & x, const SystemEstimator & system) const
// {
//     // Get predicted flow
//     Eigen::Matrix<Scalar, 3, Eigen::Dynamic> pk_hat = predict(x, pkm1_, pk_);
//     Eigen::Matrix<Scalar, 2, Eigen::Dynamic> rQbarOik_hat = pk_hat.template topRows<2>().array().rowwise()/pk_hat.row(2).array();
    
//     const int n = pk_.cols();  // Number of flow points
//     Scalar totalLogLik = 0;

//     // Process each point independently
//     for(int i = 0; i < n; ++i) {
//         // Extract single point measurements
//         // Eigen::Matrix<Scalar, 2, 1> measurement = pk_.template topRows<2>().array().rowwise()/pk_.row(2).array(); // covert to euclidean vector
//         // Extract single point measurements - convert from homogeneous to Euclidean
//         Eigen::Matrix<Scalar, 2, 1> measurement;
//         measurement[0] = pk_(0,i) / pk_(2,i);  // x/w
//         measurement[1] = pk_(1,i) / pk_(2,i);  // y/w
//         measurement = measurement.template cast<Scalar>();

//         Eigen::Matrix<Scalar, 2, 1> prediction = rQbarOik_hat.col(i);
        
//         // Create single point measurement density
//         auto likelihood = predictDensity<Scalar>(x, system);
        
//         // Add log likelihood for this point
//         totalLogLik += likelihood.log(measurement);
//     }

//     return totalLogLik;
// }


// WORKING LOGLIKELIHOOD BELOW 

// template <typename Scalar>
// Scalar MeasurementOutdoorFlowBundle::logLikelihood(const Eigen::VectorX<Scalar> & x, const SystemEstimator & system) const
// {
//     Eigen::Matrix<Scalar, 3, Eigen::Dynamic> pk_hat = predict(x, pkm1_, pk_);
//     Eigen::Matrix<Scalar, 2, Eigen::Dynamic> rQbarOik_hat = pk_hat.template topRows<2>().array().rowwise()/pk_hat.row(2).array();

//     // Calculate log likelihood directly without building large matrices
//     const int n = pk_.cols();
//     Scalar logLik = 0;
//     const Scalar logTwoPi = std::log(2 * M_PI);
    
//     // Process each point independently
//     for (int i = 0; i < n; ++i) {
//         Eigen::Matrix<Scalar, 2, 1> error = 
//             pk_.template topRows<2>().col(i).template cast<Scalar>() - rQbarOik_hat.col(i);
            
//         // Using sigma directly for each 2D measurement
//         logLik += -logTwoPi - 2*std::log(sigma_) - 0.5 * error.squaredNorm()/(sigma_*sigma_);
//     }

//     return logLik;
// }


template <typename Scalar>
Scalar MeasurementOutdoorFlowBundle::logLikelihood(const Eigen::VectorX<Scalar> & x, const SystemEstimator & system) const
{
    const int n = pkm1_.cols();
    Scalar totalLogLik = 0;

    // Process each point independently     
    for(int i = 0; i < n; ++i) {

        // Convert measurement to Euclidean coordinates
        Eigen::Matrix<Scalar, 2, 1> rQbarOik;
        rQbarOik[0] = pk_(0,i) / pk_(2,i);  
        rQbarOik[1] = pk_(1,i) / pk_(2,i);
        
        // Single point measurement density         
        auto likelihood = predictDensity<Scalar>(x, system, i);
        
        // Add log likelihood for this point         
        totalLogLik += likelihood.log(rQbarOik);     
    }

    return totalLogLik;
}

template <typename Scalar>
GaussianInfo<Scalar> MeasurementOutdoorFlowBundle::predictDensity(const Eigen::VectorX<Scalar> & x, const SystemEstimator & system, int i) const 
{
    // Get prediction in homogeneous coordinates
    Eigen::Matrix<Scalar, 3, 1> pk_hat = predict(x, pkm1_.col(i), pk_.col(i));   

    // Convert homogeneous to Euclidean
    Eigen::Matrix<Scalar, 2, 1> rQbarOik_hat = pk_hat.template head<2>() / pk_hat[2];

    // Get noise density
    const auto & Xi = noiseDensity<Scalar>(system).sqrtInfoMat();

    // Return GaussianInfo with correct dimensions
    return GaussianInfo<Scalar>::fromSqrtInfo(Xi*rQbarOik_hat, Xi);
}
// template <typename Scalar>
// GaussianInfo<Scalar> MeasurementOutdoorFlowBundle::predictDensity(const Eigen::VectorX<Scalar> & x, const SystemEstimator & system) const {
//     auto h = predict(x, pkm1_, pk_);
//     const auto & Xi = noiseDensity<Scalar>(system).sqrtInfoMat();
//     return GaussianInfo<Scalar>::fromSqrtInfo(Xi*h, Xi);
// }

template <typename Scalar>
GaussianInfo<Scalar> MeasurementOutdoorFlowBundle::noiseDensity(const SystemEstimator & system) const 
{  
    // // Create block diagonal covariance matrix for all flow measurements
    // Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> SR = 
    //     Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(2, 2) * (sigma_);
        // More efficient version using fromSqrtInfo:
    Eigen::Matrix<Scalar, 2, 2> Xi = Eigen::Matrix<Scalar, 2, 2>::Identity() * (1.0/sigma_);
    return GaussianInfo<Scalar>::fromSqrtInfo(Xi);
        
    //return GaussianInfo<Scalar>::fromSqrtMoment(SR);
}

#endif
