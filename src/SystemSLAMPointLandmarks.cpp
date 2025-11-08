#include <cmath>
#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemSLAM.h"
#include "SystemSLAMPointLandmarks.h"

SystemSLAMPointLandmarks::SystemSLAMPointLandmarks(const GaussianInfo<double> & density)
    : SystemSLAM(density)
{

}

SystemSLAM * SystemSLAMPointLandmarks::clone() const
{
    return new SystemSLAMPointLandmarks(*this);
}

std::size_t SystemSLAMPointLandmarks::numberLandmarks() const
{
    return (density.dim() - 12)/3;
}

std::size_t SystemSLAMPointLandmarks::landmarkPositionIndex(std::size_t idxLandmark) const
{
    assert(idxLandmark < numberLandmarks());
    return 12 + 3*idxLandmark;    
}

void SystemSLAMPointLandmarks::incrementFailedObservations(std::size_t idxLandmark) {
    failedObservations_[idxLandmark]++;
}

int SystemSLAMPointLandmarks::getFailedObservations(std::size_t idxLandmark) const {
    auto it = failedObservations_.find(idxLandmark);
    return (it != failedObservations_.end()) ? it->second : 0;
}

void SystemSLAMPointLandmarks::resetFailedObservations(std::size_t idxLandmark) {
    failedObservations_[idxLandmark] = 0;
}

void SystemSLAMPointLandmarks::removeLandmark(std::size_t idxLandmark) {
    std::size_t startIdx = landmarkPositionIndex(idxLandmark);
    std::size_t endIdx = startIdx + 3;

    // Remove the landmark state from the mean vector
    Eigen::VectorXd newMean = density.mean();
    newMean.segment(startIdx, density.dim() - endIdx) = newMean.segment(endIdx, density.dim() - endIdx);
    newMean.conservativeResize(density.dim() - 3);

    // Remove the landmark state from the covariance matrix
    Eigen::MatrixXd newCov = density.cov();
    newCov.block(startIdx, 0, newCov.rows() - endIdx, startIdx) = newCov.block(endIdx, 0, newCov.rows() - endIdx, startIdx);
    newCov.block(0, startIdx, startIdx, newCov.cols() - endIdx) = newCov.block(0, endIdx, startIdx, newCov.cols() - endIdx);
    newCov.block(startIdx, startIdx, newCov.rows() - endIdx, newCov.cols() - endIdx) = newCov.block(endIdx, endIdx, newCov.rows() - endIdx, newCov.cols() - endIdx);
    newCov.conservativeResize(density.dim() - 3, density.dim() - 3);

    // Update the density with the new mean and covariance
    density = GaussianInfo<double>::fromMoment(newMean, newCov);

    // Remove the landmark from the failedObservations map and update indices
    for (auto it = failedObservations_.begin(); it != failedObservations_.end();) {
        if (it->first == idxLandmark) {
            it = failedObservations_.erase(it);
        } else if (it->first > idxLandmark) {
            failedObservations_[it->first - 1] = it->second;
            it = failedObservations_.erase(it);
        } else {
            ++it;
        }
    }
}

void SystemSLAMPointLandmarks::initializeNewLandmark(const Eigen::Vector3d& position) {
    std::size_t oldDim = density.dim();
    std::size_t newDim = oldDim + 3;

    // Expand the mean vector
    Eigen::VectorXd newMean(newDim);
    newMean << density.mean(), position;

    // Expand the covariance matrix
    Eigen::MatrixXd newCov(newDim, newDim);
    newCov.setZero();
    newCov.topLeftCorner(oldDim, oldDim) = density.cov();
    
    // Set initial covariance for the new landmark
    double initialUncertainty = 1e-6; // Adjust this value based on your system's characteristics
    newCov.bottomRightCorner(3, 3) = Eigen::Matrix3d::Identity() * initialUncertainty;

    // Update the density with the new mean and covariance
    density = GaussianInfo<double>::fromMoment(newMean, newCov);

    // Initialize failed observations count for the new landmark
    failedObservations_[numberLandmarks() - 1] = 0;
}
