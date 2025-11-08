#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
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
#include "MeasurementAltimeter.h"

MeasurementAltimeter::MeasurementAltimeter(double time, const Camera & camera, double altitude)
    : Measurement(time)
    , camera_(camera)
    , measuredAltitude_(altitude)
    , sigma_(15.0) 
{
    verbosity_ = 3;
}

Eigen::VectorXd MeasurementAltimeter::simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    Eigen::VectorXd y;
    throw std::runtime_error("Not implemented");
    return y;
}

double MeasurementAltimeter::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
     return logLikelihood<double>(x, system);
}

double MeasurementAltimeter::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const
{
    // Evaluate gradient for Newton and quasi-Newton methods
    using autodiff::dual;
    using autodiff::gradient;

    // Convert input to autodiff type
    Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();

    // Compute gradient and function value using autodiff
    autodiff::dual fdual;
    g = gradient(&MeasurementAltimeter::logLikelihood<autodiff::dual>, autodiff::wrt(x_dual), autodiff::at(this, x_dual, system), fdual);

    return logLikelihood(x, system);
}

double MeasurementAltimeter::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    // Evaluate Hessian for Newton method
    H.resize(x.size(), x.size());
    H.setZero();
    // TODO: Assignment 2
    return logLikelihood(x, system, g);
}