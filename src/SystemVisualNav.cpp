#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <cstddef>
#include <cmath>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include "GaussianInfo.hpp"
#include "SystemEstimator.h"
#include "SystemVisualNav.h"

SystemVisualNav::SystemVisualNav(const GaussianInfo<double> & density)
    : SystemEstimator(density)
{

}

SystemVisualNav * SystemVisualNav::clone() const
{
    return new SystemVisualNav(*this);
}

// Evaluate f(x) from the SDE dx = f(x)*dt + dw
Eigen::VectorXd SystemVisualNav::dynamics(const Eigen::VectorXd & x) const
{
    // const std::size_t & nx = density.dim();
    const std::size_t nx = 12;
    
    Eigen::VectorXd f(nx);
    f.setZero();
    // TODO: Implement in Assignment(s)
     // JK(eta) = [Rnb(thetanb) 0; 0 TK(thetanb)]
    // Extract current state
    Eigen::Vector3d vBNb = x.segment<3>(0);      // Body translational velocity
    Eigen::Vector3d omegaBNb = x.segment<3>(3);  // Body angular velocity
    // Eigen::Vector3d rBNn = x.segment<3>(6);      // Body position
    Eigen::Vector3d Thetanb = x.segment<3>(9);   // Body orientation

    // Compute rotation matrix from body to navigation frame
    Eigen::Matrix3d Rnb = rpy2rot(Thetanb);

    // Update position
    f.segment<3>(6) = Rnb * vBNb;

    // Update orientation
    Eigen::Matrix3d T = Eigen::Matrix3d::Identity();
    double phi = Thetanb(0), theta = Thetanb(1);
    T(0,1) = sin(phi) * tan(theta);
    T(0,2) = cos(phi) * tan(theta);
    T(1,1) = cos(phi);
    T(1,2) = -sin(phi);
    T(2,1) = sin(phi) / cos(theta);
    T(2,2) = cos(phi) / cos(theta);
    f.segment<3>(9) = T * omegaBNb;

    // Velocities and landmark states remain constant in this model
    // f.segment<3>(0) = Eigen::Vector3d::Zero();  // dvBNb/dt = 0
    // f.segment<3>(3) = Eigen::Vector3d::Zero();  // domegaBNb/dt = 0
    // f.segment<3>(12) = Eigen::Vector3d::Zero(); // dm/dt = 0 for all landmarks

    return f;
}

// // Evaluate f(x) and its Jacobian J = df/fx from the SDE dx = f(x)*dt + dw
// Eigen::VectorXd SystemVisualNav::dynamics(const Eigen::VectorXd & x, Eigen::MatrixXd & J) const
// {
//     Eigen::VectorXd f = dynamics(x);

//     // Jacobian J = df/dx
//     J.resize(f.size(), x.size());
//     J.setZero();
//     // TODO: Assignment(s)

//     return f;
// }

// Evaluate f(x) and its Jacobian J = df/fx from the SDE dx = f(x)*dt + dw
Eigen::VectorXd SystemVisualNav::dynamics(const Eigen::VectorXd & x, Eigen::MatrixXd & J) const
{
    Eigen::VectorXd f = dynamics(x);
   
    // Jacobian J = df/dx
    //    
    //     [  0                  0 0 ]
    // J = [ JK d(JK(eta)*nu)/deta 0 ]
    //     [  0                  0 0 ]
    //
    J.resize(f.size(), x.size());
    J.setZero();

    // Extract current state
    Eigen::Vector3d vBNb = x.segment<3>(0);      // Body translational velocity
    Eigen::Vector3d omegaBNb = x.segment<3>(3);  // Body angular velocity
    Eigen::Vector3d Thetanb = x.segment<3>(9);   // Body orientation

    // Compute rotation matrix from body to navigation frame
    Eigen::Matrix3d Rnb = rpy2rot(Thetanb);

    // Jacobian for position update (df_r / dv)
    J.block<3,3>(6, 0) = Rnb;

    // Jacobian for position update (df_r / dTheta)
    Eigen::Matrix3d dRnb_dphi, dRnb_dtheta, dRnb_dpsi;
    double phi = Thetanb(0), theta = Thetanb(1), psi = Thetanb(2);
    double cphi = cos(phi), sphi = sin(phi), ctheta = cos(theta), stheta = sin(theta), cpsi = cos(psi), spsi = sin(psi);

    dRnb_dphi << 
        0, sphi*spsi + cphi*stheta*cpsi, cphi*spsi - sphi*stheta*cpsi,
        0, sphi*cpsi - cphi*stheta*spsi, cphi*cpsi + sphi*stheta*spsi,
        0, cphi*ctheta, -sphi*ctheta;

    dRnb_dtheta << 
        -ctheta*cpsi, -stheta*sphi*cpsi, -stheta*cphi*cpsi,
        -ctheta*spsi, -stheta*sphi*spsi, -stheta*cphi*spsi,
        -stheta, ctheta*sphi, ctheta*cphi;

    dRnb_dpsi << 
        -stheta*spsi, cphi*cpsi - sphi*stheta*spsi, sphi*cpsi + cphi*stheta*spsi,
        stheta*cpsi, -cphi*spsi - sphi*stheta*cpsi, -sphi*spsi + cphi*stheta*cpsi,
        0, 0, 0;

    J.block<3,1>(6, 9) = dRnb_dphi * vBNb;
    J.block<3,1>(6, 10) = dRnb_dtheta * vBNb;
    J.block<3,1>(6, 11) = dRnb_dpsi * vBNb;

    // Jacobian for orientation update (df_Theta / domega)
    Eigen::Matrix3d T;
    T << 1, sin(phi)*tan(theta), cos(phi)*tan(theta),
         0, cos(phi), -sin(phi),
         0, sin(phi)/cos(theta), cos(phi)/cos(theta);
    J.block<3,3>(9, 3) = T;

    // Jacobian for orientation update (df_Theta / dTheta)
    Eigen::Matrix3d dT_dphi, dT_dtheta;
    
    dT_dphi << 
        0, cos(phi)*tan(theta), -sin(phi)*tan(theta),
        0, -sin(phi), -cos(phi),
        0, cos(phi)/cos(theta), -sin(phi)/cos(theta);

    dT_dtheta << 
        0, sin(phi)/cos(theta)/cos(theta), cos(phi)/cos(theta)/cos(theta),
        0, 0, 0,
        0, sin(phi)*sin(theta)/cos(theta)/cos(theta), cos(phi)*sin(theta)/cos(theta)/cos(theta);

    J.block<3,1>(9, 9) = dT_dphi * omegaBNb;
    J.block<3,1>(9, 10) = dT_dtheta * omegaBNb;
    assert(!J.hasNaN() && "Error: NaN found in Jacobian matrix!");

    return f;
}

GaussianInfo<double> SystemVisualNav::processNoiseDensity(double dt) const
{
    // SQ is an upper triangular matrix such that SQ.'*SQ = Q is the power spectral density of the continuous time process noise
    Eigen::MatrixXd SQ;
  
    // Define the number of state variables
    const int nv = 3;       // Number of translational velocity states
    const int nw = 3;       // Number of angular velocity states
    
    // // Define the noise standard deviations
    double sigma_v = 1e-0;     // std dev of translational acceleration noise
    double sigma_w = 1e-2;     // std dev of angular acceleration noise

    //  // Create the square root of the power spectral density matrix
    SQ = Eigen::MatrixXd::Zero(nv + nw, nv + nw);
    
    // // Set the diagonal elements
    SQ.diagonal().segment(0, nv).setConstant(sigma_v);
    SQ.diagonal().segment(nv, nw).setConstant(sigma_w);
    
    // Distribution of noise increment dw ~ N(0, Q*dt) for time increment dt
    return GaussianInfo<double>::fromSqrtMoment(SQ*std::sqrt(dt));
}

std::vector<Eigen::Index> SystemVisualNav::processNoiseIndex() const
{
    // Indices of process model equations where process noise is injected
    std::vector<Eigen::Index> idxQ = {0, 1, 2, 3, 4, 5};
    
    return idxQ;
}

cv::Mat & SystemVisualNav::view()
{
    return view_;
};

const cv::Mat & SystemVisualNav::view() const
{
    return view_;
};

std::size_t SystemVisualNav::numberLandmarks() const
{
    return (density.dim() - 18)/3;
}

std::size_t SystemVisualNav::landmarkPositionIndex(std::size_t idxLandmark) const
{
    assert(idxLandmark < numberLandmarks());
    return 18 + 3*idxLandmark;    
}

GaussianInfo<double> SystemVisualNav::bodyPositionDensity() const
{
    return density.marginal(Eigen::seqN(6, 3));
}

GaussianInfo<double> SystemVisualNav::bodyOrientationDensity() const
{
    return density.marginal(Eigen::seqN(9, 3));
}

GaussianInfo<double> SystemVisualNav::bodyTranslationalVelocityDensity() const
{
    return density.marginal(Eigen::seqN(0, 3));
}

GaussianInfo<double> SystemVisualNav::bodyAngularVelocityDensity() const
{
    return density.marginal(Eigen::seqN(3, 3));
}

Eigen::Vector3d SystemVisualNav::cameraPosition(const Camera & camera, const Eigen::VectorXd & x, Eigen::MatrixXd & J)
{
    Eigen::Vector3<autodiff::dual> rCNn_dual;
    Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();
    J = jacobian(cameraPosition<autodiff::dual>, wrt(x_dual), at(camera, x_dual), rCNn_dual);
    return rCNn_dual.cast<double>();
};

GaussianInfo<double> SystemVisualNav::cameraPositionDensity(const Camera & camera) const
{
    auto f = [&](const Eigen::VectorXd & x, Eigen::MatrixXd & J) { return cameraPosition(camera, x, J); };
    return density.affineTransform(f);
}

Eigen::Vector3d SystemVisualNav::cameraOrientationEuler(const Camera & camera, const Eigen::VectorXd & x, Eigen::MatrixXd & J)
{
    Eigen::Vector3<autodiff::dual> Thetanc_dual;
    Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();
    J = jacobian(cameraOrientationEuler<autodiff::dual>, wrt(x_dual), at(camera, x_dual), Thetanc_dual);
    return Thetanc_dual.cast<double>();
};

GaussianInfo<double> SystemVisualNav::cameraOrientationEulerDensity(const Camera & camera) const
{
    auto f = [&](const Eigen::VectorXd & x, Eigen::MatrixXd & J) { return cameraOrientationEuler(camera, x, J); };
    return density.affineTransform(f);    
}

GaussianInfo<double> SystemVisualNav::landmarkPositionDensity(std::size_t idxLandmark) const
{
    assert(idxLandmark < numberLandmarks());
    std::size_t idx = landmarkPositionIndex(idxLandmark);
    return density.marginal(Eigen::seqN(idx, 3));
}

/// TODO: Add docstringexplaining why zeta is etakm1 
void SystemVisualNav::initialise_state_density(const Eigen::VectorXd nu, const Eigen::VectorXd etak, const Eigen::VectorXd zetak)
{
    Eigen::VectorXd mu = Eigen::VectorXd::Zero(18); 
    mu.segment<6>(0) = Eigen::VectorXd(nu);                                     
    mu.segment<6>(6) = Eigen::VectorXd(etak);                                           
    mu.segment<6>(12) = Eigen::VectorXd(zetak); 

    // Initialise square root of covariance matrix (upper triangular)
    Eigen::MatrixXd S = Eigen::MatrixXd::Identity(18, 18);

    // Initial velocity uncertainty (first 6 states)
    S.block<3, 3>(0, 0) = Eigen::MatrixXd::Identity(3, 3) * 50;         // m/s
    S.block<3, 3>(3, 3) = Eigen::MatrixXd::Identity(3, 3) * 1;          // rad/s

    // Current pose eta uncertainty (middle 6 states)
    S.block<3, 3>(6, 6) = Eigen::MatrixXd::Identity(3, 3) * 0.01;       // m  
    S.block<3, 3>(9, 9) = Eigen::MatrixXd::Identity(3, 3) * 0.0005;     // rad

    // Previous pose zeta uncertainty (final 6 states)
    S.block<3, 3>(12, 12) = Eigen::MatrixXd::Identity(3, 3) * 50;       // m
    S.block<3, 3>(15, 15) = Eigen::MatrixXd::Identity(3, 3) * 0.1;      // rad

    density = GaussianInfo<double>::fromSqrtMoment(mu, S);
}
