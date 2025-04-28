#ifndef MEASUREMENTSLAMUNIQUETAGBUNDLE_H
#define MEASUREMENTSLAMUNIQUETAGBUNDLE_H

#include <Eigen/Core>
#include "SystemBase.h"
#include "SystemEstimator.h"
#include "Camera.h"
#include "Pose.hpp"
#include "MeasurementSLAM.h"
#include "SystemSLAMPoseLandmarks.h"
#include <set>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

class MeasurementUniqueTagBundle : public MeasurementSLAM
{
public:
    MeasurementUniqueTagBundle(double time, const Eigen::Matrix<double, 8, Eigen::Dynamic> & Y, const Camera & camera, const std::vector<int> & markerIds);
    MeasurementSLAM * clone() const override;
    Eigen::Matrix3d skew(const Eigen::Vector3d& v);
    virtual Eigen::VectorXd simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    template <typename Scalar> Scalar logLikelihood(const Eigen::VectorX<Scalar> & x, const SystemEstimator & system) const;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const override;
    Eigen::Matrix4d getTagPoseWorld(const std::vector<cv::Point2f> &corners, const Camera &camera, double tagLength, const Pose<double> &cameraPoseWorld) const;
    template <typename Scalar> Eigen::Vector2<Scalar> predictFeature(const Eigen::VectorX<Scalar> & x, const SystemSLAM & system, std::size_t idxLandmark) const;
    Eigen::Vector2d predictFeature(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemSLAM & system, std::size_t idxLandmark) const;
    virtual GaussianInfo<double> predictFeatureDensity(const SystemSLAM & system, std::size_t idxLandmark) const override;
    
    template <typename Scalar> Eigen::VectorX<Scalar> predictFeatureBundle(const Eigen::VectorX<Scalar> & x, const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const;
    Eigen::VectorXd predictFeatureBundle(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const;
    virtual GaussianInfo<double> predictFeatureBundleDensity(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const override;
    
    virtual const std::vector<int> & associate(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) override;
    void initializeNewLandmark(SystemSLAM &  system, int markerId, const Eigen::Matrix4d & tagPose);
    Eigen::Matrix4d getTagPoseCamera(const std::vector<cv::Point2f> &corners, const Camera &camera, double tagLength) const;
    const std::vector<size_t>& getAssociatedLandmarks() const { return associatedLandmarks_; }

private:
    std::vector<size_t> associatedLandmarks_;
    std::set<size_t> visibleLandmarks_;  
    
protected:
    virtual void update(SystemBase & system) override;
    Eigen::Matrix<double, 8, Eigen::Dynamic> Y_;    // Feature bundle
    double sigma_;                                  // Feature error standard deviation (in pixels)
    std::vector<int> idxFeatures_;                  // Features associated with visible landmarks
    const std::vector<int> & markerIds_;            // Marker Ids parsed into MeasurentSLAMUniqueTagBundle
    Camera camera_;                                 // Camera object to computer image size |y| used in loglikelihood
};

template <typename Scalar>
double to_double(const Scalar& value) {
    if constexpr (std::is_same_v<Scalar, double>) {
        return value;
    } else {
        return autodiff::val(value);    
    }
}

template <typename Scalar>
Scalar MeasurementUniqueTagBundle::logLikelihood(const Eigen::VectorX<Scalar> & x, const SystemEstimator & system) const
{
    const SystemSLAM & systemSLAM = dynamic_cast<const SystemSLAM &>(system);

    Scalar logLikelihood = Scalar(0);
    const Scalar tagLength = Scalar(0.166);  // 166 mm edge length of each marker
    
    // Extract body pose from the state vector
    Eigen::Vector3<Scalar> rBNn = x.template segment<3>(6);
    Eigen::Vector3<Scalar> ThetaNb = x.template segment<3>(9);
    Eigen::Matrix3<Scalar> Rnb = rpy2rot(ThetaNb);
    Pose<Scalar> Tnb(Rnb, rBNn);

    // Calculate camera pose using body pose and body-to-camera transformation
    Pose<Scalar> Tnc = Tnb * camera_.Tbc;

    for (int i = 0; i < Y_.cols(); ++i) {
        int markerId = markerIds_[i];
        size_t landmarkIndex = systemSLAM.getLandmarkIndex(markerId);
            
            std::size_t idx = systemSLAM.landmarkPositionIndex(landmarkIndex);
            Eigen::Matrix<Scalar, 6, 1> landmark = x.template segment<6>(idx);
            Eigen::Vector3<Scalar> rjNn = landmark.template head<3>();
            Eigen::Vector3<Scalar> Thetanj = landmark.template tail<3>();
            Eigen::Matrix3<Scalar> Rnj = rpy2rot(Thetanj);

            for (int c = 0; c < 4; ++c) {
                Eigen::Vector3<Scalar> rjcjj;
                switch (c) {
                    case 0: rjcjj << -tagLength/2,  tagLength/2, Scalar(0); break;
                    case 1: rjcjj <<  tagLength/2,  tagLength/2, Scalar(0); break;
                    case 2: rjcjj <<  tagLength/2, -tagLength/2, Scalar(0); break;
                    case 3: rjcjj << -tagLength/2, -tagLength/2, Scalar(0); break;
                }
                
                Eigen::Vector3<Scalar> rjcNn = Rnj * rjcjj + rjNn;
                Eigen::Vector2<Scalar> predictedCorner = camera_.worldToPixel(rjcNn, Tnb);
                Eigen::Vector2<Scalar> measuredCorner(Scalar(Y_(2*c, i)), Scalar(Y_(2*c+1, i)));
                Eigen::Vector2<Scalar> error = measuredCorner - predictedCorner;
                logLikelihood += -Scalar(0.5) * error.squaredNorm() / (sigma_ * sigma_) - Scalar(std::log(2 * M_PI * sigma_ * sigma_));
        }
    }
    
    // Add penalty for unassociated visible landmarks
    size_t unassociatedLandmarks = visibleLandmarks_.size() - associatedLandmarks_.size();
    Scalar imageArea = Scalar(camera_.imageSize.width * camera_.imageSize.height);
    GaussianInfo<Scalar> tempGaussian = GaussianInfo<Scalar>::fromMoment(Eigen::Matrix<Scalar, 1, 1>(imageArea), Eigen::Matrix<Scalar, 1, 1>::Constant(1e-6));
    Scalar logImageArea = tempGaussian.log(Eigen::Matrix<Scalar, 1, 1>(imageArea));
    logLikelihood -= Scalar(4 * unassociatedLandmarks) * logImageArea;

    return logLikelihood;
}
    
// Image feature location for a given landmark
template <typename Scalar>
Eigen::Vector2<Scalar> MeasurementUniqueTagBundle::predictFeature(const Eigen::VectorX<Scalar> & x, const SystemSLAM & system, std::size_t idxLandmark) const
{
    // Obtain camera pose from state
    Pose<Scalar> Tnc;
    Tnc.translationVector = system.cameraPosition(camera_, x); // rCNn
    Tnc.rotationMatrix = system.cameraOrientation(camera_, x); // Rnc

    // Obtain landmark position from state
    std::size_t idx = system.landmarkPositionIndex(idxLandmark);
    Eigen::Vector3<Scalar> rPNn = x.template segment<3>(idx);

    // Camera vector
    Eigen::Vector3<Scalar> rPCc;
    rPCc = Tnc.rotationMatrix.transpose() * (rPNn - Tnc.translationVector);

    // Pixel coordinates
    Eigen::Vector2<Scalar> rQOi;
    rQOi = camera_.vectorToPixel(rPCc);

    return rQOi;
}

template <typename Scalar>
Eigen::VectorX<Scalar> MeasurementUniqueTagBundle::predictFeatureBundle(const Eigen::VectorX<Scalar> & x, const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const
{
    // Ensure the state vector size matches the system's dimension
    assert(x.size() == system.density.dim());

    // Get the number of landmarks
    const std::size_t & nL = idxLandmarks.size();
    
    // Initialize the output vector h to store all predicted features
    // The size is 2*nL because each feature has 2 coordinates (x and y in the image)
    Eigen::VectorX<Scalar> h(2*nL);
    
    // Iterate through each landmark
    for (std::size_t i = 0; i < nL; ++i)
    {
        // Predict the feature location for the current landmark
        Eigen::Vector2<Scalar> rQOi = predictFeature(x, system, idxLandmarks[i]);
        
        // Set the pair of elements in h corresponding to this landmark
        // We use 2*i as the starting index because each feature occupies 2 elements
        h.template segment<2>(2*i) = rQOi;
    }
    
    // Return the complete feature bundle prediction vector
    return h;
}

#endif 

