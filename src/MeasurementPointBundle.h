#ifndef MEASUREMENTPOINTBUNDLE_H
#define MEASUREMENTPOINTBUNDLE_H

#include <Eigen/Core>
#include "SystemBase.h"
#include "SystemEstimator.h"
#include "Camera.h"
#include "Pose.hpp"
#include "MeasurementVisualNav.h"
//#include "SystemSLAMPointLandmarks.h"
#include <set>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

class MeasurementPointBundle : public MeasurementVisualNav
{
public:
    MeasurementPointBundle(double time, const Eigen::Matrix<double, 2, Eigen::Dynamic> & Y, const Camera & camera);
    MeasurementVisualNav * clone() const override;
    virtual Eigen::VectorXd simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const override;
    template <typename Scalar> Scalar logLikelihood(const Eigen::VectorX<Scalar> & x, const SystemEstimator & system) const;
    template <typename Scalar> Eigen::Vector2<Scalar> predictFeature(const Eigen::VectorX<Scalar> & x, const SystemVisualNav & system, std::size_t idxLandmark) const;
    Eigen::Vector2d predictFeature(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemVisualNav & system, std::size_t idxLandmark) const;
    virtual GaussianInfo<double> predictFeatureDensity(const SystemVisualNav & system, std::size_t idxLandmark) const override;

    template <typename Scalar> Eigen::VectorX<Scalar> predictFeatureBundle(const Eigen::VectorX<Scalar> & x, const SystemVisualNav & system, const std::vector<std::size_t> & idxLandmarks) const;
    Eigen::VectorXd predictFeatureBundle(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemVisualNav & system, const std::vector<std::size_t> & idxLandmarks) const;
    virtual GaussianInfo<double> predictFeatureBundleDensity(const SystemVisualNav & system, const std::vector<std::size_t> & idxLandmarks) const override;
    const std::vector<int>& getIdxFeatures() const { return idxFeatures_; }
    const std::vector<std::size_t>& getIdxLandmarks() const { return idxLandmarks_; }
    const std::vector<size_t>& getAssociatedLandmarks() const { return associatedLandmarks_; }
    virtual const std::vector<int> & associate(const SystemVisualNav & system, const std::vector<std::size_t> & idxLandmarks) override;
    
    
protected:

    virtual void update(SystemBase & system) override;
    Eigen::Matrix<double, 2, Eigen::Dynamic> Y_;    // Feature bundle
    double sigma_;                                  // Feature error standard deviation (in pixels)
    std::vector<int> idxFeatures_;                  // Features associated with visible landmarks
    std::vector<std::size_t> idxLandmarks_;
    Camera camera_;                                 // Camera object to computer image size |y| used in loglikelihood
    std::vector<size_t> associatedLandmarks_;
    std::set<size_t> visibleLandmarks_;  

};

template <typename Scalar>
Scalar MeasurementPointBundle::logLikelihood(const Eigen::VectorX<Scalar> & x, const SystemEstimator & system) const
{
    //const SystemSLAMPointLandmarks & systemSLAM = dynamic_cast<const SystemSLAM &>(system);

    Scalar logLikelihood = Scalar(0);
    
    // Extract body pose from the state vector
    Eigen::Vector3<Scalar> rBNn = x.template segment<3>(6);  // Assuming position starts at index 6
    Eigen::Vector3<Scalar> ThetaNb = x.template segment<3>(9);  // Assuming orientation starts at index 9
    Eigen::Matrix3<Scalar> Rnb = rpy2rot(ThetaNb);
    Pose<Scalar> Tnb(Rnb, rBNn);

    return logLikelihood = 0.0;
}

// Image feature location for a given landmark
template <typename Scalar>
Eigen::Vector2<Scalar> MeasurementPointBundle::predictFeature(const Eigen::VectorX<Scalar> & x, const SystemVisualNav & system, std::size_t idxLandmark) const
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
    // TODO: Lab 7
    rPCc = Tnc.rotationMatrix.transpose() * (rPNn - Tnc.translationVector);

    // Pixel coordinates
    Eigen::Vector2<Scalar> rQOi;
    
    // TODO: Lab 7
    rQOi = camera_.vectorToPixel(rPCc);

    // std::cout << "Landmark " << idxLandmark << std::endl;
    // std::cout << "Landmark position: " << rPNn.transpose() << std::endl;
    // std::cout << "Projected feature: " << rQOi.transpose() << std::endl;


    return rQOi;
}

template <typename Scalar>
Eigen::VectorX<Scalar> MeasurementPointBundle::predictFeatureBundle(const Eigen::VectorX<Scalar> & x, const SystemVisualNav & system, const std::vector<std::size_t> & idxLandmarks) const
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
