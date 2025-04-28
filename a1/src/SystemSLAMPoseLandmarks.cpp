#include <cmath>
#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemSLAM.h"
#include "SystemSLAMPoseLandmarks.h"

SystemSLAMPoseLandmarks::SystemSLAMPoseLandmarks(const GaussianInfo<double> & density)
    : SystemSLAM(density)
{
    
}

SystemSLAM * SystemSLAMPoseLandmarks::clone() const
{
    return new SystemSLAMPoseLandmarks(*this);
}

std::size_t SystemSLAMPoseLandmarks::numberLandmarks() const
{
    return (density.dim() - 12)/6;
}

std::size_t SystemSLAMPoseLandmarks::landmarkPositionIndex(std::size_t idxLandmark) const
{
    assert(idxLandmark < numberLandmarks());
    return 12 + 6*idxLandmark;    
}

// void SystemSLAMPoseLandmarks::addLandmark(const Eigen::VectorXd& landmarkState)
// {
//     // Expand the state vector
//     Eigen::VectorXd newState(density.mean().size() + landmarkState.size());
//     newState << density.mean(), landmarkState;

//     // Expand the covariance matrix
//     Eigen::MatrixXd newCovariance(newState.size(), newState.size());
//     newCovariance.setZero();
//     newCovariance.topLeftCorner(density.cov().rows(), density.cov().cols()) = density.cov();

//     // Set initial covariance for the new landmark
//     Eigen::MatrixXd initialLandmarkCov = Eigen::MatrixXd::Identity(landmarkState.size(), landmarkState.size()) * 0.1; // Adjust as needed
//     newCovariance.bottomRightCorner(landmarkState.size(), landmarkState.size()) = initialLandmarkCov;

//     // Update system state
//     density = GaussianInfo<double>::fromMoment(newState, newCovariance);

//     // Add the new landmark ID
//     landmarkIds_.push_back(nextLandmarkId_++);
// }