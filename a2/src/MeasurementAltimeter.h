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
#include "GaussianInfo.hpp"

class MeasurementAltimeter : public Measurement
{
public:
    MeasurementAltimeter(double time, const Camera & camera, double altitude);
    virtual Eigen::VectorXd simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const override;

    // Helper functions for log likelihood and visualisation
    template <typename Scalar> Scalar predict(const Eigen::VectorX<Scalar> & x, const SystemEstimator & system) const;
    template <typename Scalar> Scalar logLikelihood(const Eigen::VectorX<Scalar> & x, const SystemEstimator & system) const;
    template <typename Scalar> GaussianInfo<Scalar> predictDensity(const Eigen::VectorX<Scalar> & x, const SystemEstimator & system) const;
    template <typename Scalar> GaussianInfo<Scalar> noiseDensity(const SystemEstimator & system) const;

protected:
    Camera camera_;
    double measuredAltitude_;
    double sigma_;                                          // Altimeter error standard deviation (in meters maybe)
};

template <typename Scalar>
// Evaluate h(x) from the measurement model y = h(x) + v
Scalar MeasurementAltimeter::predict(const Eigen::VectorX<Scalar> & x, const SystemEstimator & system) const
{
    assert(x.rows() >= 18);
    assert(x.cols() == 1);

    // Extract the relevant state variables from the input vector x
    Eigen::Vector3<Scalar> rBNn = x.template segment<3>(6);  // Position
    Eigen::Vector3<Scalar> ThetaNb = x.template segment<3>(9);  // Orientation (RPY)

    // Compute the camera pose in the world frame
    Eigen::Matrix3<Scalar> Rnb = rpy2rot(ThetaNb);
    Pose<Scalar> Tnb(Rnb, rBNn);
    Pose<Scalar> Tnc = camera_.bodyToCamera(Tnb);

    // Compute the camera position in the world frame
    Eigen::Vector3<Scalar> rCNn = Tnc.translationVector;

    Scalar h = -static_cast<Scalar>(rCNn[2]);
    return h;
}

template <typename Scalar>
GaussianInfo<Scalar> MeasurementAltimeter::noiseDensity(const SystemEstimator & system) const {
    // SR is an upper triangular matrix such that SR.'*SR = R is the measurement noise covariance
    Eigen::Matrix<Scalar, 1, 1> SR = Eigen::Matrix<Scalar, 1, 1>::Identity() * sigma_;
    return GaussianInfo<Scalar>::fromSqrtMoment(SR);
}

template <typename Scalar>
Scalar MeasurementAltimeter::logLikelihood(const Eigen::VectorX<Scalar> & x, const SystemEstimator & system) const {
    auto likelihood = predictDensity<Scalar>(x, system);
    
    // Create a 1D vector for the measurement
    Eigen::Matrix<Scalar, 1, 1> measurement;
    measurement(0) = static_cast<Scalar>(measuredAltitude_);

     // Print altimeter debug info
    // std::cout << "Altimeter Update:" << std::endl;
    // std::cout << "Measured altitude: " << measuredAltitude_ << std::endl;
    // std::cout << "Predicted altitude: " << predict(x, system) << std::endl;
    // std::cout << "Sigma: " << sigma_ << std::endl;
    // std::cout << "Log likelihood: " << likelihood.log(measurement) << std::endl;
    
    return likelihood.log(measurement);
}

template <typename Scalar>
GaussianInfo<Scalar> MeasurementAltimeter::predictDensity(const Eigen::VectorX<Scalar> & x, const SystemEstimator & system) const {
    auto h = predict(x, system);
    const auto & Xi = noiseDensity<Scalar>(system).sqrtInfoMat();
    return GaussianInfo<Scalar>::fromSqrtInfo(Xi*h, Xi);
}

// template <typename Scalar>
// Scalar MeasurementAltimeter::logLikelihood(const Eigen::VectorX<Scalar> & x) const
// {
    
//     // Eigen::Matrix<Scalar, 3, Eigen::Dynamic> predictedAltitude = predictAltitude();

//     // // Calculate the error between predicted and measured altitude
//     // Eigen::Matrix<Scalar, 2, Eigen::Dynamic> error = predictedAltitude - measuredAltitude_;

//     // Calculate the log-likelihood
//     Scalar logLik = 0;
//     // const int n = error.cols(); // Number of flow vectors
//     // const Scalar logSqrt2Pi = std::log(std::sqrt(2 * M_PI));

//     // for (int i = 0; i < n; ++i)
//     // {
//     //     // Mahalanobis distance for each flow vector
//     //     Scalar mahalanobisDistance = error.col(i).squaredNorm() / (sigma_ * sigma_);
        
//     //     // Log-likelihood contribution of this flow vector
//     //     logLik += -logSqrt2Pi - std::log(sigma_) - 0.5 * mahalanobisDistance;
//     // }
//     return logLik;
// }

// template <typename Scalar>
// Scalar MeasurementAltimeter::logLikelihood(const Eigen::VectorX<Scalar> & x, const SystemEstimator & system) const
// {
//     Scalar h = predictDensity(x, system); // Predicted Altitude

//     // Calculate the error between predicted and measured altitude
//     Scalar error = static_cast<Scalar>(measuredAltitude_) - static_cast<Scalar>(predictedAltitude);

//     // Calculate the log-likelihood
//     Scalar logLik = -std::log(std::sqrt(2 * M_PI) * sigma_) - (error * error) / (2 * sigma_ * sigma_);

//     return logLik;
// }

// GaussianInfo<double> MeasurementAltimeter::predictDensity(const Eigen::VectorXd & x, const SystemEstimator & system) const
// {
//     Eigen::VectorXd h = predict(x, system);
//     const Eigen::MatrixXd & Xi = noiseDensity(system).sqrtInfoMat();
//     return GaussianInfo<double>::fromSqrtInfo(Xi*h, Xi);
// }

// template <typename Scalar>
// Scalar MeasurementAltimeter::logLikelihood(const Eigen::VectorX<Scalar>& x, const SystemEstimator& system) const
// {
//     assert(x.rows() >= 18);
//     assert(x.cols() == 1);

//     // Extract the relevant state variables from the input vector x
//     Eigen::Vector3<Scalar> rBNn = x.template segment<3>(6);  // Position
//     Eigen::Vector3<Scalar> ThetaNb = x.template segment<3>(9);  // Orientation (RPY)

//     // Compute the camera pose in the world frame
//     Eigen::Matrix3<Scalar> Rnb = rpy2rot(ThetaNb);
//     Pose<Scalar> Tnb(Rnb, rBNn);
//     Pose<Scalar> Tnc = camera_.bodyToCamera(Tnb);

//     // Compute the camera position in the world frame
//     Eigen::Vector3<Scalar> rCNn = Tnc.translationVector;

//     // Calculate the predicted altitude
//     Scalar predictedAltitude = -static_cast<Scalar>(rCNn[2]);

//     // Calculate the error between predicted and measured altitude
//     Scalar error = static_cast<Scalar>(measuredAltitude_) - predictedAltitude;

//     // Calculate the log-likelihood
//     Scalar logLik = -std::log(std::sqrt(2 * M_PI) * sigma_) - (error * error) / (2 * sigma_ * sigma_);

//     return logLik;
// }

#endif
