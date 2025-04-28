#include <cmath>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include "GaussianInfo.hpp"
#include "MeasurementGaussianLikelihood.h"
#include "MeasurementRADAR.h"

const double MeasurementRADAR::r1 = 5000;    // Horizontal position of sensor [m]
const double MeasurementRADAR::r2 = 5000;    // Vertical position of sensor [m]

MeasurementRADAR::MeasurementRADAR(double time, const Eigen::VectorXd & y)
    : MeasurementGaussianLikelihood(time, y)
{
    // updateMethod_ = UpdateMethod::BFGSLMSQRT;
    // updateMethod_ = UpdateMethod::BFGSTRUSTSQRT;
    // updateMethod_ = UpdateMethod::SR1TRUSTEIG;
    updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;

    // updateMethod_ = UpdateMethod::AFFINE;
    // updateMethod_ = UpdateMethod::GAUSSNEWTON;
    // updateMethod_ = UpdateMethod::LEVELBERGMARQUARDT;
}

MeasurementRADAR::MeasurementRADAR(double time, const Eigen::VectorXd & y, int verbosity)
    : MeasurementGaussianLikelihood(time, y, verbosity)
{
    // updateMethod_ = UpdateMethod::BFGSLMSQRT;
    // updateMethod_ = UpdateMethod::BFGSTRUSTSQRT;
    // updateMethod_ = UpdateMethod::SR1TRUSTEIG;
    updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;

    // updateMethod_ = UpdateMethod::AFFINE;
    // updateMethod_ = UpdateMethod::GAUSSNEWTON;
    // updateMethod_ = UpdateMethod::LEVELBERGMARQUARDT;
}

MeasurementRADAR::~MeasurementRADAR() = default;

std::string MeasurementRADAR::getProcessString() const
{
    return "RADAR measurement update:";
}

// Evaluate h(x) from the measurement model y = h(x) + v
Eigen::VectorXd MeasurementRADAR::predict(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    Eigen::VectorXd h(1);
    double h1 = x(0);  // altitude
    h(0) = std::sqrt(r1*r1 + (h1 - r2)*(h1 - r2));
    return h;
}

// Evaluate h(x) and its Jacobian J = dh/fx from the measurement model y = h(x) + v
Eigen::VectorXd MeasurementRADAR::predict(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::MatrixXd & dhdx) const
{
    Eigen::VectorXd h = predict(x, system);

    //              dh_i
    // dhdx(i, j) = ----
    //              dx_j
    dhdx.resize(h.size(), x.size());
    dhdx.setZero();
    
    double h1 = x(0);  // altitude
    double range = std::sqrt(r1*r1 + (h1 - r2)*(h1 - r2));
    dhdx(0, 0) = (h1 - r2) / range;  // dh/dh
    // Other elements remain zero as the measurement doesn't depend on velocity or drag coefficient

    return h;
}

Eigen::VectorXd MeasurementRADAR::predict(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::MatrixXd & dhdx, Eigen::Tensor<double, 3> & d2hdx2) const
{
    Eigen::VectorXd h = predict(x, system, dhdx);

    //                    d^2 h_i     d 
    // d2hdx2(i, j, k) = --------- = ---- dhdx(i, j)
    //                   dx_j dx_k   dx_k
    d2hdx2.resize(h.size(), x.size(), x.size());
    d2hdx2.setZero();
    
    double h1 = x(0);  // altitude
    double range = std::sqrt(r1*r1 + (h1 - r2)*(h1 - r2));
    d2hdx2(0, 0, 0) = 1 / range - std::pow((h1 - r2) / range, 2) / range;
    // Other elements remain zero
    return h;
}

GaussianInfo<double> MeasurementRADAR::noiseDensity(const SystemEstimator & system) const
{
    // SR is an upper triangular matrix such that SR.'*SR = R is the measurement noise covariance
    double sigma_rng = 50.0;  // 50 meters standard deviation
    Eigen::MatrixXd SR = Eigen::MatrixXd::Identity(1, 1) * sigma_rng;
    return GaussianInfo<double>::fromSqrtMoment(SR);
}

