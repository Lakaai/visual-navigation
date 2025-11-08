#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <cstddef>
#include <numeric>
#include <vector>
#include <stdexcept>
#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemBase.h"
#include "SystemEstimator.h"
#include "SystemSLAM.h"
#include "Camera.h"
#include "Measurement.h"
#include "MeasurementSLAM.h"
#include "MeasurementSLAMPointBundle.h"
//using namespace autodiff;


MeasurementPointBundle::MeasurementPointBundle(double time, const Eigen::Matrix<double, 2, Eigen::Dynamic> & Y, const Camera & camera)
    : MeasurementSLAM(time, camera)
    , Y_(Y)
    , sigma_(1.0) // TODO: Assignment(s)
    , camera_(camera)

{
    // updateMethod_ = UpdateMethod::BFGSLMSQRT;
    updateMethod_ = UpdateMethod::BFGSTRUSTSQRT;
    // updateMethod_ = UpdateMethod::SR1TRUSTEIG;
    // updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;
}

MeasurementSLAM * MeasurementPointBundle::clone() const
{
    return new MeasurementPointBundle(*this);
}

Eigen::VectorXd MeasurementPointBundle::simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    Eigen::VectorXd y(Y_.size());
    throw std::runtime_error("Not implemented");
    return y;
}

double MeasurementPointBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    return logLikelihood<double>(x, system);
}

double MeasurementPointBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const
{
    // Evaluate gradient for Newton and quasi-Newton methods
    using autodiff::dual;
    using autodiff::gradient;

    // Convert input to autodiff type
    Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();

    // Compute gradient and function value using autodiff
    autodiff::dual fdual;
    g = gradient(&MeasurementPointBundle::logLikelihood<autodiff::dual>, autodiff::wrt(x_dual), autodiff::at(this, x_dual, system), fdual);

    // return val(fdual);
    return logLikelihood(x, system);
}

double MeasurementPointBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    // Evaluate Hessian for Newton method
    H.resize(x.size(), x.size());
    H.setZero();
    // TODO: Assignment(s)
    return logLikelihood(x, system, g);
}

void MeasurementPointBundle::update(SystemBase & system)
{
    SystemSLAMPointLandmarks & systemSLAM = dynamic_cast<SystemSLAMPointLandmarks &>(system);

    // TODO: Assignment(s)

    // Identify landmarks with matching features (data association)

    // 1. Identify landmarks with matching features (data association)
    std::vector<std::size_t> idxLandmarks_(systemSLAM.numberLandmarks());
    std::cout << "idxlandmarks size in update" << idxLandmarks_.size() << std::endl;
    std::iota(idxLandmarks_.begin(), idxLandmarks_.end(), 0);
    const std::vector<int> & idxFeatures_ = associate(systemSLAM, idxLandmarks_);
    std::cout << "idxfeatures size in update" << idxFeatures_.size() << std::endl;
    // Clear previous associations and update based on associate function results
    associatedLandmarks_.clear();
    for (size_t i = 0; i < idxFeatures_.size(); ++i) {
        if (idxFeatures_[i] != -1) {
            associatedLandmarks_.push_back(idxFeatures_[i]);
            std::cout << "assocaitedlandmarks size in update" << associatedLandmarks_.size() << std::endl;
        }
    }

    // 2. Remove failed landmarks from map (consecutive failures to match)
    for (std::size_t j = 0; j < idxLandmarks_.size(); ++j) {
        if (idxFeatures_[j] < 0) {
            systemSLAM.incrementFailedObservations(idxLandmarks_[j]);
            if (systemSLAM.getFailedObservations(idxLandmarks_[j]) > 10) { // Adjust threshold as needed
                systemSLAM.removeLandmark(idxLandmarks_[j]);
            }
        } else {
            systemSLAM.resetFailedObservations(idxLandmarks_[j]);
        }
    }   

    // 3. Identify surplus features that do not correspond to landmarks in the map
    std::vector<int> unassociatedFeatures;
    for (int i = 0; i < Y_.cols(); ++i) {
        if (std::find(idxFeatures_.begin(), idxFeatures_.end(), i) == idxFeatures_.end()) {
            unassociatedFeatures.push_back(i);
        }
    }

    // Print the size of unassociatedFeatures
    std::cout << "unassociatedFeatures size: " << unassociatedFeatures.size() << std::endl;
    
    // 4. Initialize new landmarks from best surplus features
    const int maxNewLandmarks = 5; // Adjust as needed
    int newLandmarksAdded = 0;
    
    for (int i : unassociatedFeatures) {
        if (newLandmarksAdded >= maxNewLandmarks) break;

        Eigen::Vector2d featurePosition(Y_(0, i), Y_(1, i));
        
        // Get current camera pose
        Eigen::Vector3d rCNn = systemSLAM.cameraPosition(camera_, systemSLAM.density.mean());
        Eigen::Matrix3d Rnc = systemSLAM.cameraOrientation(camera_, systemSLAM.density.mean());
        Pose<double> Tnc(Rnc, rCNn);

        // Assuming featurePosition is Eigen::Vector2d
        cv::Vec2d cvFeaturePosition(featurePosition.x(), featurePosition.y());

        // Use pixelToVector function
        cv::Vec3d cvUnitVector = camera_.pixelToVector(cvFeaturePosition);

        // Convert cv::Vec3d to Eigen::Vector3d
        Eigen::Vector3d unitVector(cvUnitVector[0], cvUnitVector[1], cvUnitVector[2]);

        // Estimate depth (you might want to use a more sophisticated method)
        double estimatedDepth = 2.0; // Example depth, adjust based on your scenario

        // Calculate the 3D position of the landmark in the camera frame
        Eigen::Vector3d landmarkPositionCamera = unitVector * estimatedDepth;

        // Transform the landmark position to the world frame
        Eigen::Vector3d landmarkPosition = Rnc.transpose() * landmarkPositionCamera + rCNn;

        // Now you can use this landmarkPosition to initialize the new landmark
        systemSLAM.initializeNewLandmark(landmarkPosition);
        newLandmarksAdded++;
    }

    // // Remove failed landmarks from map (consecutive failures to match)
    // for (std::size_t j = 0; j < idxLandmarks.size(); ++j) {
    //     if (idxFeatures[j] < 0) {
    //         systemSLAM.incrementFailedObservations(idxLandmarks[j]);
    //         if (systemSLAM.getFailedObservations(idxLandmarks[j]) > 10) {
    //             systemSLAM.removeLandmark(idxLandmarks[j]);
    //         }
    //     } else {
    //         systemSLAM.resetFailedObservations(idxLandmarks[j]);
    //     }
    // }

    // Identify surplus features that do not correspond to landmarks in the map
    // Initialise up to Nmax – N new landmarks from best surplus features
    
    Measurement::update(system);    // Do the actual measurement update
}

// Image feature location for a given landmark and Jacobian
Eigen::Vector2d MeasurementPointBundle::predictFeature(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemSLAM & system, std::size_t idxLandmark) const
{
    // Set elements of J
    // TODO: Lab 7 (optional)
    using autodiff::dual;
    using autodiff::jacobian;

    // Create a mutable copy of x
    Eigen::VectorX<dual> x_dual = x.cast<dual>();

    // Define lambda function for autodiff
    auto func = [&](const auto& x) -> Eigen::Vector2<dual> {
        return this->predictFeature(x, system, idxLandmark);
    };

    // Compute Jacobian and function value using autodiff
    Eigen::Vector2<dual> y;
    J = jacobian(func, autodiff::wrt(x_dual), autodiff::at(x_dual), y);
    
    // Return the evaluated function value (cast to double)
    return y.cast<double>();
}

// Density of image feature location for a given landmark
GaussianInfo<double> MeasurementPointBundle::predictFeatureDensity(const SystemSLAM & system, std::size_t idxLandmark) const
{
    const std::size_t & nx = system.density.dim();
    const std::size_t ny = 2;

    //   y   =   h(x) + v  
    // \___/   \__________/
    //   ya  =   ha(x, v)
    //
    // Helper function to evaluate ha(x, v) and its Jacobian Ja = [dha/dx, dha/dv]
    const auto func = [&](const Eigen::VectorXd & xv, Eigen::MatrixXd & Ja)
    {
        assert(xv.size() == nx + ny);
        Eigen::VectorXd x = xv.head(nx);
        Eigen::VectorXd v = xv.tail(ny);
        Eigen::MatrixXd J;
        Eigen::VectorXd ya = predictFeature(x, J, system, idxLandmark) + v;
        Ja.resize(ny, nx + ny);
        Ja << J, Eigen::MatrixXd::Identity(ny, ny);
        return ya;
    };
    
    auto pv = GaussianInfo<double>::fromSqrtMoment(sigma_*Eigen::MatrixXd::Identity(ny, ny));
    auto pxv = system.density*pv;   // p(x, v) = p(x)*p(v)
    // // Print pv
    // std::cout << "pv mean:\n" << pv.mean() << std::endl;
    // std::cout << "pv cov:\n" << pv.cov() << std::endl;

    // // Print pxv
    // std::cout << "pxv:\n" << pxv.mean() << std::endl;
    // std::cout << "pxv cov:\n" << pxv.cov().diagonal() << std::endl;
    return pxv.affineTransform(func);
}

// Image feature locations for a bundle of landmarks
Eigen::VectorXd MeasurementPointBundle::predictFeatureBundle(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const
{
    // Get the number of landmarks
    const std::size_t & nL = idxLandmarks.size();
    
    // Get the dimension of the state vector
    const std::size_t & nx = system.density.dim();
    
    // Ensure the state vector size matches the system's dimension
    assert(x.size() == nx);

    // Initialize the output vector h to store all predicted features
    // The size is 2*nL because each feature has 2 coordinates (x and y in the image)
    Eigen::VectorXd h(2*nL);
    
    // Initialize the Jacobian matrix J
    // Rows: 2*nL (2 for each landmark)
    // Columns: nx (dimension of the state vector)
    J.resize(2*nL, nx);
    
    // Iterate through each landmark
    for (std::size_t i = 0; i < nL; ++i)
    {
        // Jacobian for the current feature prediction
        Eigen::MatrixXd Jfeature;
        
        // Predict the feature location and its Jacobian for the current landmark
        Eigen::Vector2d rQOi = predictFeature(x, Jfeature, system, idxLandmarks[i]);
        
        // Set the pair of elements in h corresponding to this landmark
        // We use 2*i as the starting index because each feature occupies 2 elements
        h.segment<2>(2*i) = rQOi;
        
        // Set the corresponding pair of rows in the Jacobian matrix J
        // This forms the block structure described in equation (8)
        // Set pair of rows of J
        J.block(2*i, 0, 2, nx) = Jfeature;
    }
    
    // Return the complete feature bundle prediction vector
    return h;
}

// Density of image features for a set of landmarks
GaussianInfo<double> MeasurementPointBundle::predictFeatureBundleDensity(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const
{
    const std::size_t & nx = system.density.dim();
    const std::size_t ny = 2*idxLandmarks.size();

    //   y   =   h(x) + v  
    // \___/   \__________/
    //   ya  =   ha(x, v)
    //
    // Helper function to evaluate ha(x, v) and its Jacobian Ja = [dha/dx, dha/dv]
    const auto func = [&](const Eigen::VectorXd & xv, Eigen::MatrixXd & Ja)
    {
        assert(xv.size() == nx + ny);
        Eigen::VectorXd x = xv.head(nx);
        Eigen::VectorXd v = xv.tail(ny);
        Eigen::MatrixXd J;
        Eigen::VectorXd ya = predictFeatureBundle(x, J, system, idxLandmarks) + v;
        Ja.resize(ny, nx + ny);
        Ja << J, Eigen::MatrixXd::Identity(ny, ny);
        return ya;
    };

    auto pv = GaussianInfo<double>::fromSqrtMoment(sigma_*Eigen::MatrixXd::Identity(ny, ny));
    auto pxv = system.density*pv;   // p(x, v) = p(x)*p(v)
    return pxv.affineTransform(func);
    
}

#include "association_util.h"
const std::vector<int> & MeasurementPointBundle::associate(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks)
{ 

    
    
    for (size_t i = 0; i < idxLandmarks.size(); ++i) {
        size_t idx = system.landmarkPositionIndex(idxLandmarks[i]);
        std::cout << "Landmark " << i << " position covariance: \n" 
                  << system.density.cov().block<3,3>(idx,idx) << std::endl;
    }
    std::cout << "Entering associate function" << std::endl;
    std::cout << "Number of landmarks passed to associate: " << idxLandmarks.size() << std::endl;

    if (idxLandmarks.empty()) {
        std::cout << "Warning: No landmarks passed to associate function" << std::endl;
        return idxFeatures_;
    }

    for (size_t i = 0; i < idxLandmarks.size(); ++i) {
    Eigen::Vector3d landmarkPos = system.landmarkPositionDensity(idxLandmarks[i]).mean();
    std::cout << "Landmark " << i << " position: " << landmarkPos.transpose() << std::endl;
    }

    for (int i = 0; i < std::min<int>(5, static_cast<int>(Y_.cols())); ++i) {
    std::cout << "Feature " << i << " position: " << Y_.col(i).transpose() << std::endl;
    }

    GaussianInfo<double> featureBundleDensity = predictFeatureBundleDensity(system, idxLandmarks);
    snn(system, featureBundleDensity, idxLandmarks, Y_, camera_, idxFeatures_);

    std::cout << "After association:" << std::endl;
    std::cout << "idxFeatures_ size: " << idxFeatures_.size() << std::endl;
    std::cout << "First few idxFeatures_: ";
    for (size_t i = 0; i < std::min(size_t(5), idxFeatures_.size()); ++i) {
        std::cout << idxFeatures_[i] << " ";
    }
    std::cout << std::endl;

    return idxFeatures_;

}