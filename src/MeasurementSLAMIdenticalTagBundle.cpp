#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <cstddef>
#include <numeric>
#include <vector>
#include <stdexcept>
#include <set>
#include "GaussianInfo.hpp"
#include "SystemBase.h"
#include "SystemEstimator.h"
#include "SystemSLAM.h"
#include "Camera.h"
#include "Measurement.h"
#include "MeasurementSLAM.h"
#include "MeasurementSLAMIdenticalTagBundle.h"
#include "association_util.h"
#include "rotation.hpp"
#include <opencv2/core/types.hpp>
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/imgcodecs.hpp>
#include "Plot.h"
#include "plot_util.h"

using namespace autodiff;
MeasurementIdenticalTagBundle::MeasurementIdenticalTagBundle(double time, const Eigen::Matrix<double, 8, Eigen::Dynamic> & Y, const Camera & camera, const std::vector<int> & markerIds)
    : MeasurementSLAM(time, camera)
    , Y_(Y)
    , sigma_(1)
    , markerIds_(markerIds)
    , camera_(camera)

{
    // updateMethod_ = UpdateMethod::BFGSLMSQRT;
     updateMethod_ = UpdateMethod::BFGSTRUSTSQRT;
    // updateMethod_ = UpdateMethod::SR1TRUSTEIG;
    // updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;
}


MeasurementSLAM * MeasurementIdenticalTagBundle::clone() const
{
    return new MeasurementIdenticalTagBundle(*this);
}

Eigen::VectorXd MeasurementIdenticalTagBundle::simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    Eigen::VectorXd y(Y_.size());
    throw std::runtime_error("Not implemented");
    return y;
}

double MeasurementIdenticalTagBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    return logLikelihood<double>(x, system);
}

double MeasurementIdenticalTagBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const
{
    // Evaluate gradient for Newton and quasi-Newton methods
    using autodiff::dual;
    using autodiff::gradient;

    // Convert input to autodiff type
    Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();

    // Compute gradient and function value using autodiff
    autodiff::dual fdual;
    g = gradient(&MeasurementIdenticalTagBundle::logLikelihood<autodiff::dual>, autodiff::wrt(x_dual), autodiff::at(this, x_dual, system), fdual);

    // return val(fdual);
    return logLikelihood(x, system);
}


double MeasurementIdenticalTagBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    // Evaluate Hessian for Newton method
    H.resize(x.size(), x.size());
    H.setZero();
    // TODO: Assignment(s)
    return logLikelihood(x, system, g);
}

// void MeasurementUniqueTagBundle::update(SystemBase & system)
// {
//     SystemSLAMPoseLandmarks & systemSLAM = dynamic_cast<SystemSLAMPoseLandmarks &>(system);

//     // Get existing landmark IDs
//     std::vector<int> existingLandmarkIds;
//     for (std::size_t i = 0; i < systemSLAM.numberLandmarks(); ++i) {
//         existingLandmarkIds.push_back(systemSLAM.getLandmarkId(i));
//     }

//     // Get pose for each detected tag
//     std::vector<Eigen::Matrix4d> tagPoses;
//     for (int i = 0; i < Y_.cols(); ++i) {
//         std::vector<cv::Point2f> corners;
//         for (int j = 0; j < 4; ++j) {
//             corners.push_back(cv::Point2f(Y_(2*j, i), Y_(2*j+1, i)));
//         }
        
//         Eigen::Vector3d rBNn = systemSLAM.density.mean().segment<3>(6);  // body position in world frame
//         Eigen::Vector3d ThetaNb = systemSLAM.density.mean().segment<3>(9);  // body orientation in world frame
//         Eigen::Matrix3d Rnb = rpy2rot(ThetaNb);
//         Pose<double> bodyPoseWorld(Rnb, rBNn);

//         Eigen::Matrix4d tagPoseWorld = getTagPoseWorld(corners, camera_, 0.166, bodyPoseWorld);
        
//         tagPoses.push_back(tagPoseWorld);
    
//     }
    
//     // Initialize new landmarks only for new markers
//     for (size_t i = 0; i < markerIds_.size(); ++i) {
//         int markerId = markerIds_[i];
//         if (std::find(existingLandmarkIds.begin(), existingLandmarkIds.end(), markerId) == existingLandmarkIds.end()) {
//             // This is a new marker, initialize it as a landmark
//             initializeNewLandmark(systemSLAM, markerId, tagPoses[i]);
//             std::cout << "Initialized new landmark with ID: " << markerId << std::endl;
//         }
//     }

//     // // Perform data association and update
//     // // TODO_ 
//     std::vector<std::size_t> idxLandmarks(systemSLAM.numberLandmarks());
//     std::iota(idxLandmarks.begin(), idxLandmarks.end(), 0);
//     const std::vector<int> & idxFeatures = associate(systemSLAM, idxLandmarks);

//     // Perform the actual measurement update
//     Measurement::update(system);
// }

void MeasurementIdenticalTagBundle::update(SystemBase & system)
{
    SystemSLAMPoseLandmarks & systemSLAM = dynamic_cast<SystemSLAMPoseLandmarks &>(system);

    // Clear previous visibility
    visibleLandmarks_.clear();
    
    // Get existing landmark IDs
    std::vector<int> existingLandmarkIds;
    for (std::size_t i = 0; i < systemSLAM.numberLandmarks(); ++i) {
        existingLandmarkIds.push_back(systemSLAM.getLandmarkId(i));
    }

    // Extract current camera pose
    Eigen::Vector3d rBNn = systemSLAM.density.mean().segment<3>(6);
    Eigen::Vector3d ThetaNb = systemSLAM.density.mean().segment<3>(9);
    Eigen::Matrix3d Rnb = rpy2rot(ThetaNb);
    Pose<double> Tnb(Rnb, rBNn);

    // Determine visible landmarks
    for (size_t i = 0; i < systemSLAM.numberLandmarks(); ++i) {
        std::size_t idx = systemSLAM.landmarkPositionIndex(i);
        Eigen::Vector3d rjNn = systemSLAM.density.mean().segment<3>(idx);
        cv::Vec3d rjNn_cv(rjNn.x(), rjNn.y(), rjNn.z());
        
        if (camera_.isWorldWithinFOV(rjNn_cv, Tnb)) {
            visibleLandmarks_.insert(i);
        }
    }

    
    // Get pose for each detected tag
    std::vector<Eigen::Matrix4d> tagPoses;
    for (int i = 0; i < Y_.cols(); ++i) {
        std::vector<cv::Point2f> corners;
        for (int j = 0; j < 4; ++j) {
            corners.push_back(cv::Point2f(Y_(2*j, i), Y_(2*j+1, i)));
        }
        
        Eigen::Matrix4d tagPoseWorld = getTagPoseWorld(corners, camera_, 0.166, Pose<double>(Rnb, rBNn));
        tagPoses.push_back(tagPoseWorld);
    }
    
    // Initialize new landmarks only for new markers
    for (size_t i = 0; i < markerIds_.size(); ++i) {
        int markerId = markerIds_[i];
        if (std::find(existingLandmarkIds.begin(), existingLandmarkIds.end(), markerId) == existingLandmarkIds.end()) {
            initializeNewLandmark(systemSLAM, markerId, tagPoses[i]);
            std::cout << "Initialized new landmark with ID: " << markerId << std::endl;
        }
    }

    // Prepare idxLandmarks for associate function
    std::vector<std::size_t> idxLandmarks(systemSLAM.numberLandmarks());
    std::iota(idxLandmarks.begin(), idxLandmarks.end(), 0);
    std::cout << "idxLandmarks size inside update: " << idxLandmarks.size() << std::endl;

    // Perform data association using the associate function
    const std::vector<int> & idxFeatures = associate(systemSLAM, idxLandmarks);

    // Clear previous associations and update based on associate function results
    associatedLandmarks_.clear();
    for (size_t i = 0; i < idxFeatures.size(); ++i) {
        if (idxFeatures[i] != -1) {
            associatedLandmarks_.push_back(idxFeatures[i]);
        }
    }


    // Perform the actual measurement update
    Measurement::update(system);
}

// Image feature location for a given landmark and Jacobian
Eigen::Vector2d MeasurementIdenticalTagBundle::predictFeature(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemSLAM & system, std::size_t idxLandmark) const
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
GaussianInfo<double> MeasurementIdenticalTagBundle::predictFeatureDensity(const SystemSLAM & system, std::size_t idxLandmark) const
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

    return pxv.affineTransform(func);
}

// Image feature locations for a bundle of landmarks
Eigen::VectorXd MeasurementIdenticalTagBundle::predictFeatureBundle(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const
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
GaussianInfo<double> MeasurementIdenticalTagBundle::predictFeatureBundleDensity(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const
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

const std::vector<int> & MeasurementIdenticalTagBundle::associate(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks)
{


    // // Iterate through all detected markers
    // for (size_t i = 0; i < markerIds_.size(); ++i) {
    //     int markerId = markerIds_[i];
        
    //     // Get the landmark index of the state for this marker ID
    //     std::size_t landmarkIndex = system.getLandmarkIndex(markerId);
        
    //     // If a landmark with this marker ID exists in the system
    //     if (landmarkIndex != static_cast<std::size_t>(-1)) {
    //         // Find the position of this landmark index in the idxLandmarks vector
    //         auto it = std::find(idxLandmarks.begin(), idxLandmarks.end(), landmarkIndex);
    //         if (it != idxLandmarks.end()) {
    //             // If found, store the index in idxFeatures_ and add to associatedLandmarks_
    //             idxFeatures_[i] = std::distance(idxLandmarks.begin(), it);
    //             associatedLandmarks_.push_back(landmarkIndex);
    //         }
    //     }
    // }
    // // Return the vector of associations
    

    idxFeatures_.clear();
    // Check if there are any landmarks to associate
    if (idxLandmarks.empty()) {
        return idxFeatures_;
    }

    idxFeatures_.resize(idxLandmarks.size(), -1);

    Eigen::Matrix<double, 2, Eigen::Dynamic> tagCentres(2, idxLandmarks.size());
    for (size_t j = 0; j < idxLandmarks.size(); ++j) {
        std::size_t landmarkIdx = idxLandmarks[j];
        Eigen::Vector2d center;
        center << (Y_(0, j) + Y_(2, j) + Y_(4, j) + Y_(6, j)) / 4.0,
                  (Y_(1, j) + Y_(3, j) + Y_(5, j) + Y_(7, j)) / 4.0;
        tagCentres.col(j) = center;
    }

    // Plot the predicted feature coordinates on the image
    cv::Mat image = system.view().clone();
    for (size_t j = 0; j < idxLandmarks.size(); ++j) {
        cv::Point2f predictedFeature(tagCentres(0, j), tagCentres(1, j));
        cv::circle(image, predictedFeature, 5, cv::Scalar(255, 255, 0), -1); // Draw predicted feature as yellow circle
    }
    cv::imshow("Predicted Features", image);
    int key = cv::waitKey(0); // Pause the video until the user presses a key



    double s = snn(system, predictFeatureBundleDensity(system, idxLandmarks), idxLandmarks, tagCentres, camera_, idxFeatures_);

    // // Plot the tag centers on the image
    
    // for (size_t j = 0; j < idxLandmarks.size(); ++j) {
    //     int i = idxFeatures_[j];
    //     if (i >= 0) {
    //         cv::Point2f center(tagCentres(0, j), tagCentres(1, j));
    //         cv::circle(image, center, 5, cv::Scalar(0, 255, 0), -1);
    //         cv::putText(image, std::to_string(j), center + cv::Point2f(10, 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    //     } else {
    //         cv::Point2f center(tagCentres(0, j), tagCentres(1, j));
    //         cv::circle(image, center, 5, cv::Scalar(255, 0, 0), -1);
    //         cv::putText(image, std::to_string(j), center + cv::Point2f(10, 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
    //     }
    // }

    // Display the image
    // cv::imshow("Data Association", image);
    // int key = cv::waitKey(0); // Pause the video until the user presses a key

    // ------------------------------------------------------------------------
    // Visualisation and console output
    // ------------------------------------------------------------------------
    plotAllFeatures(image, tagCentres);
    for (std::size_t jj = 0; jj < idxFeatures_.size(); ++jj)
    {
        int j           = idxLandmarks[jj];     // Index of landmark in state vector
        int i           = idxFeatures_[jj];      // Index of feature
        bool isMatch    = i >= 0;

        GaussianInfo<double> featureDensity = predictFeatureDensity(system, j);
        const Eigen::VectorXd & murQOi = featureDensity.mean();

        // Plot confidence ellipse and landmark index text
        Eigen::Vector3d colour = isMatch ? Eigen::Vector3d(0, 0, 255) : Eigen::Vector3d(255, 0, 0);
        plotGaussianConfidenceEllipse(image, featureDensity, colour);
        plotLandmarkIndex(image, murQOi, colour, j);

        if (isMatch)
            std::cout << "Feature " << i << " located at [" << tagCentres.col(i).transpose() << "] in image matches landmark " << j <<"." << std::endl;
        else
            std::cout << "No feature associated with landmark "<< j << "." << std::endl;
    }
    plotMatchedFeatures(image, idxFeatures_, tagCentres);
    std::cout << std::endl;

    return idxFeatures_;
}

void MeasurementIdenticalTagBundle::initializeNewLandmark(SystemSLAM & system, int markerId, const Eigen::Matrix4d & tagPose)
{
    // Extract position and orientation from the tag pose

    Eigen::Vector3d position = tagPose.block<3,1>(0,3);
    
    Eigen::Matrix3d rotation = tagPose.block<3,3>(0,0);
    
    Eigen::Vector3d orientation = rot2rpy(rotation);                    

    // Create initial state for the new landmark
    Eigen::VectorXd initialLandmarkState(6);
    initialLandmarkState << position, orientation;

    // Expand state vector
    Eigen::VectorXd newState(system.density.mean().size() + 6);
    newState << system.density.mean(), initialLandmarkState;

    // Extract body pose from the state vector
    Eigen::Vector3d rBNn = newState.segment<3>(6);  // Assuming position starts at index 6
    Eigen::Vector3d ThetaNb = newState.segment<3>(9);  // Assuming orientation starts at index 9
    Eigen::Matrix3d Rnb = rpy2rot(ThetaNb);

    // Create body-to-world transformation
    Pose<double> Tnb(Rnb, rBNn);

    // Convert body-to-world to camera-to-world transformation
    Pose<double> Tnc = Tnb * camera_.Tbc;
    

    // Check if the new landmark is within the camera's FOV
    cv::Vec3d landmarkPosition(position[0], position[1], position[2]);

    bool isWithinFOV = camera_.isWorldWithinFOV(landmarkPosition, Tnb);
    if (!isWithinFOV) {
        std::cout << "Warning: Initialized landmark is not within camera's FOV." << std::endl;
    }

    // Check if the landmark projects onto the image frame
    Eigen::Vector2d pixelCoordinates = camera_.worldToPixel(position, Tnb);
    bool isWithinImage = (pixelCoordinates[0] >= 0 && pixelCoordinates[0] < camera_.imageSize.width &&
                          pixelCoordinates[1] >= 0 && pixelCoordinates[1] < camera_.imageSize.height);

    if (!isWithinImage) {
        std::cout << "Warning: Initialized landmark does not project onto the image frame." << std::endl;
        std::cout << "Projected pixel coordinates: " << pixelCoordinates.transpose() << std::endl;
    }

    // Proceed with initialization even if checks fail, but with warnings
    
    // Expand covariance matrix
    Eigen::MatrixXd newCovariance(newState.size(), newState.size());
    newCovariance.setZero();
    newCovariance.topLeftCorner(system.density.cov().rows(), system.density.cov().cols()) = system.density.cov();

    // Set initial covariance for the new landmark
    Eigen::Matrix<double, 6, 6> initialLandmarkCov;
    initialLandmarkCov.setZero();
    initialLandmarkCov.diagonal() << 0.05, 0.05, 0.05, 0.001, 0.001, 0.001; // m^2 and rad
    newCovariance.bottomRightCorner(6, 6) = initialLandmarkCov;

    // Update system state
    system.density = GaussianInfo<double>::fromMoment(newState, newCovariance);

    // Add the new landmark ID to the system
    system.addLandmark(markerId);

    // std::cout << "Initialized new landmark with ID: " << markerId << std::endl;
    // std::cout << "Position: " << position.transpose() << std::endl;
    // std::cout << "Orientation: " << orientation.transpose() << std::endl;
    // std::cout << "Is within FOV: " << (isWithinFOV ? "Yes" : "No") << std::endl;
    // std::cout << "Is within image frame: " << (isWithinImage ? "Yes" : "No") << std::endl;
    // std::cout << "Projected pixel coordinates: " << pixelCoordinates.transpose() << std::endl;
}

Eigen::Matrix4d MeasurementIdenticalTagBundle::getTagPoseCamera(const std::vector<cv::Point2f> &corners, const Camera &camera, double tagLength) const
{
        // Define 3D coordinates of tag corners in tag's local frame
    std::vector<cv::Point3f> objectPoints = {
        cv::Point3f(-tagLength/2,  tagLength/2, 0),
        cv::Point3f( tagLength/2,  tagLength/2, 0),
        cv::Point3f( tagLength/2, -tagLength/2, 0),
        cv::Point3f(-tagLength/2, -tagLength/2, 0)
    };

    // Convert camera intrinsics to OpenCV format
    cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) << 
        camera.cameraMatrix.at<double>(0,0), 0, camera.cameraMatrix.at<double>(0,2),
        0, camera.cameraMatrix.at<double>(1,1), camera.cameraMatrix.at<double>(1,2),
        0, 0, 1);

    cv::Mat distCoeffs = camera.distCoeffs;

    cv::Mat rvec, tvec;
    bool pnpSuccess = cv::solvePnP(objectPoints, corners, cameraMatrix, distCoeffs, rvec, tvec);

    if (!pnpSuccess) {
        std::cerr << "PnP solver failed" << std::endl;
        return Eigen::Matrix4d::Identity();  // Return identity as fallback
    }

    // Convert rotation vector to rotation matrix
    cv::Mat rotMat;
    cv::Rodrigues(rvec, rotMat);

    // // Sanity checks
    // // 1. Check if rotation matrix is valid (determinant should be close to 1)
    // double det = cv::determinant(rotMat);
    // if (std::abs(det - 1.0) > 1e-5) {
    //     std::cerr << "Invalid rotation matrix (det = " << det << ")" << std::endl;
    //     return Eigen::Matrix4d::Identity();
    // }

    // // 2. Check if translation vector is reasonable (e.g., not too far from camera)
    // double translationNorm = cv::norm(tvec);
    // if (translationNorm > 10.0) {  // Adjust this threshold based on your expected tag distances
    //     std::cerr << "Translation seems too large: " << translationNorm << " units" << std::endl;
    // }

    // Create 4x4 transformation matrix
    Eigen::Matrix4d tagTransformMatrix = Eigen::Matrix4d::Identity();
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            tagTransformMatrix(i,j) = rotMat.at<double>(i,j);
        }
        tagTransformMatrix(i,3) = tvec.at<double>(i);
    }

    // // 3. Check if the tag's z-axis (normal) is pointing towards the camera
    // Eigen::Vector3d tagNormal = tagTransformMatrix.block<3,1>(0,2);
    // Eigen::Vector3d tagToCamera = -tagTransformMatrix.block<3,1>(0,3);
    // if (tagNormal.dot(tagToCamera) < 0) {
    //     std::cerr << "Warning: Tag normal is pointing away from the camera" << std::endl;
    // }

    return tagTransformMatrix;
}

Eigen::Matrix4d MeasurementIdenticalTagBundle::getTagPoseWorld(const std::vector<cv::Point2f> &corners, const Camera &camera, double tagLength, const Pose<double> &bodyPoseWorld) const
{
    // Define 3D coordinates of tag corners in tag's local frame
    std::vector<cv::Point3f> objectPoints = {
        cv::Point3f(-tagLength/2,  tagLength/2, 0),
        cv::Point3f( tagLength/2,  tagLength/2, 0),
        cv::Point3f( tagLength/2, -tagLength/2, 0),
        cv::Point3f(-tagLength/2, -tagLength/2, 0)
    };

    // Convert camera intrinsics to OpenCV format
    cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) << 
        camera.cameraMatrix.at<double>(0,0), 0, camera.cameraMatrix.at<double>(0,2),
        0, camera.cameraMatrix.at<double>(1,1), camera.cameraMatrix.at<double>(1,2),
        0, 0, 1);

    cv::Mat distCoeffs = camera.distCoeffs;

    cv::Mat rvec, tvec;
    bool pnpSuccess = cv::solvePnP(objectPoints, corners, cameraMatrix, distCoeffs, rvec, tvec);

    if (!pnpSuccess) {
        std::cerr << "PnP solver failed" << std::endl;
        return Eigen::Matrix4d::Identity();  // Return identity as fallback
    }

    // Convert rotation vector to rotation matrix
    cv::Mat rotMat;
    cv::Rodrigues(rvec, rotMat);

    // Create 4x4 transformation matrix (tag pose in camera frame)
    Eigen::Matrix4d tagPoseCamera = Eigen::Matrix4d::Identity();
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            tagPoseCamera(i,j) = rotMat.at<double>(i,j);
        }
        tagPoseCamera(i,3) = tvec.at<double>(i);
    }


    // Sanity checks
    // // 1. Check if rotation matrix is valid (determinant should be close to 1)
    // double det = cv::determinant(rotMat);
    // if (std::abs(det - 1.0) > 1e-5) {
    //     std::cerr << "Invalid rotation matrix (det = " << det << ")" << std::endl;
    //     return Eigen::Matrix4d::Identity();
    // }

    // // 2. Check if translation vector is reasonable (e.g., not too far from camera)
    // double translationNorm = cv::norm(tvec);
    // if (translationNorm > 10.0) {  // Adjust this threshold based on your expected tag distances
    //     std::cerr << "Translation seems too large: " << translationNorm << " units" << std::endl;
    // }


    // 3. Check if the tag's z-axis (normal) is pointing towards the camera
    // Eigen::Vector3d tagNormal = tagPoseCamera.block<3,1>(0,2);
    // Eigen::Vector3d tagToCamera = -tagPoseCamera.block<3,1>(0,3);
    // if (tagNormal.dot(tagToCamera) < 0) {
    //     std::cerr << "Warning: Tag normal is pointing away from the camera" << std::endl;
    // }

    /// Convert tag pose from camera frame to world frame
    // First, we need to get the camera pose in the world frame
    Eigen::Matrix4d bodyPoseWorldMat = Eigen::Matrix4d::Identity();
    bodyPoseWorldMat.block<3,3>(0,0) = bodyPoseWorld.rotationMatrix;
    bodyPoseWorldMat.block<3,1>(0,3) = bodyPoseWorld.translationVector;

    // Tbc is the transformation from body to camera (should be known and fixed)
    Eigen::Matrix4d Tbc = Eigen::Matrix4d::Identity();
    Tbc.block<3,3>(0,0) = camera.Tbc.rotationMatrix;
    Tbc.block<3,1>(0,3) = camera.Tbc.translationVector;

    // Camera pose in world frame
    Eigen::Matrix4d cameraPoseWorldMat = bodyPoseWorldMat * Tbc;

    // Now transform the tag pose to world frame
    Eigen::Matrix4d tagPoseWorld = cameraPoseWorldMat * tagPoseCamera;

    return tagPoseWorld;
}

