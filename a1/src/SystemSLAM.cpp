#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <cstddef>
#include <cmath>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include "GaussianInfo.hpp"
#include "SystemEstimator.h"
#include "SystemSLAM.h"

SystemSLAM::SystemSLAM(const GaussianInfo<double> & density)
    : SystemEstimator(density)
{

}

// Evaluate f(x) from the SDE dx = f(x)*dt + dw
Eigen::VectorXd SystemSLAM::dynamics(const Eigen::VectorXd & x) const
{
    assert(density.dim() == x.size());
    //
    //  dnu/dt =          0 + dwnu/dt
    // deta/dt = JK(eta)*nu +       0
    //   dm/dt =          0 +       0
    // \_____/   \________/   \_____/
    //  dx/dt  =    f(x)    +  dw/dt
    //
    //        [          0 ]
    // f(x) = [ JK(eta)*nu ]
    //        [          0 ] for all map states
    //
    //        [                    0 ]
    //        [                    0 ]
    // f(x) = [    Rnb(thetanb)*vBNb ]
    //        [ TK(thetanb)*omegaBNb ]
    //        [                    0 ] for all map states
    //
    Eigen::VectorXd f(x.size());
    f.setZero();

    // TODO: Implement in Assignment(s)
    // JK(eta) = [Rnb(thetanb) 0; 0 TK(thetanb)]
    // Extract current state
    Eigen::Vector3d vBNb = x.segment<3>(0);      // Body translational velocity
    Eigen::Vector3d omegaBNb = x.segment<3>(3);  // Body angular velocity
    Eigen::Vector3d rBNn = x.segment<3>(6);      // Body position
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

// Evaluate f(x) and its Jacobian J = df/fx from the SDE dx = f(x)*dt + dw
Eigen::VectorXd SystemSLAM::dynamics(const Eigen::VectorXd & x, Eigen::MatrixXd & J) const
{
    Eigen::VectorXd f = dynamics(x);

    // TODO: Implement in Assignment(s)

    // Jacobian J = df/dx
    //    
    //     [  0                  0 0 ]
    // J = [ JK d(JK(eta)*nu)/deta 0 ]
    //     [  0                  0 0 ]
    //
    J.resize(f.size(), x.size());
    J.setZero();

    // TODO: Implement in Assignment(s)
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

    return f;
}

// GaussianInfo<double> SystemSLAM::processNoiseDensity(double dt) const
// {
//     const int n = 6;  // Total number of states with noise (velocities only)

//     // Create the square root of the power spectral density matrix
//     Eigen::MatrixXd SQ = Eigen::MatrixXd::Zero(n, n);
    
//     // Set all diagonal elements to 0.1
//     SQ.diagonal().setConstant(0.1);

//     // Distribution of noise increment dw ~ N(0, Q*dt) for time increment dt
//     return GaussianInfo<double>::fromSqrtMoment(SQ * std::sqrt(dt));

//     // SQ is an upper triangular matrix such that SQ.'*SQ = Q is the power spectral density of the continuous time process noise
//     // Eigen::MatrixXd SQ;
    
//     // TODO: Assignment(s)
//     // Define the number of state variables
//     // const int nv = 3;  // Number of velocity states (3 for translational velocity)
//     // const int nw = 3;  // Number of angular velocity states
//     // const int nr = 3;  // Number of position states
//     // const int nt = 3;  // Number of orientation states
//     // const int nm = 6 * numberLandmarks();  // Number of landmark states

//     // // Define the noise standard deviations
//     // double sigma_v = 0.1;  // m/s^2, std dev of translational acceleration noise
//     // double sigma_w = 0.1;  // rad/s^2, std dev of angular acceleration noise
//     // double sigma_r = 0.01;  // m, std dev of position noise
//     // double sigma_t = 0.01;  // rad, std dev of orientation noise

//     // // Create the square root of the power spectral density matrix
//     // Eigen::MatrixXd SQ = Eigen::MatrixXd::Zero(nv + nw + nr + nt + nm, nv + nw + nr + nt + nm);
    
//     // // Set the diagonal elements for velocities (accelerations in the body frame)
//     // SQ.diagonal().segment(0, nv).setConstant(sigma_v);
//     // SQ.diagonal().segment(nv, nw).setConstant(sigma_w);

//     // // Set the diagonal elements for position and orientation (small direct perturbations)
//     // SQ.diagonal().segment(nv + nw, nr).setConstant(sigma_r);
//     // SQ.diagonal().segment(nv + nw + nr, nt).setConstant(sigma_t);

//     // // Landmarks are assumed static, so no noise is added to their states
//     // // SQ.diagonal().segment(nv + nw + nr + nt, nm).setZero();

//     // // Distribution of noise increment dw ~ N(0, Q*dt) for time increment dt
//     // return GaussianInfo<double>::fromSqrtMoment(SQ*std::sqrt(dt));
// }

GaussianInfo<double> SystemSLAM::processNoiseDensity(double dt, int scenario) const
{
    // Define the number of state variables
    const int nv = 3;  // Number of translational velocity states
    const int nw = 3;  // Number of angular velocity states
    const int nr = 3;  // Number of position states
    const int nt = 3;  // Number of orientation states
    int nm;
    if (scenario == 3) 
    {
        nm = 3 * numberLandmarks();  // For point landmarks
    }
    else if (scenario == 1 || scenario == 2) 
    {
        nm = 6 * numberLandmarks();  // For pose landmarks
    }
        
    // Define the noise standard deviations
    double sigma_v = 0.5;    // m/s^2, std dev of translational acceleration noise
    double sigma_w = 0.1;    // rad/s^2, std dev of angular acceleration noise
    double sigma_r = 0.01;   // m, std dev of position noise
    double sigma_t = 0.005;  // rad, std dev of orientation noise
    double sigma_m = 1e-6;   // Very small noise for landmarks (almost static)

    // Create the square root of the power spectral density matrix
    Eigen::MatrixXd SQ = Eigen::MatrixXd::Zero(nv + nw + nr + nt + nm, nv + nw + nr + nt + nm);
    
    // Set the diagonal elements
    SQ.diagonal().segment(0, nv).setConstant(sigma_v);
    SQ.diagonal().segment(nv, nw).setConstant(sigma_w);
    SQ.diagonal().segment(nv + nw, nr).setConstant(sigma_r);
    SQ.diagonal().segment(nv + nw + nr, nt).setConstant(sigma_t);
    SQ.diagonal().segment(nv + nw + nr + nt, nm).setConstant(sigma_m);

    // Distribution of noise increment dw ~ N(0, Q*dt) for time increment dt
    return GaussianInfo<double>::fromSqrtMoment(SQ * std::sqrt(dt));
}

// std::vector<Eigen::Index> SystemSLAM::processNoiseIndex() const
// {
//     // Indices of process model equations where process noise is injected
//     std::vector<Eigen::Index> idxQ;
//     // TODO: Assignment(s)
//     for (Eigen::Index i = 0; i < 6; ++i) {  // 0-11 for velocities, position, and orientation
//         idxQ.push_back(i);
//     }
//     return idxQ;
// }

std::vector<Eigen::Index> SystemSLAM::processNoiseIndex() const
{
    std::vector<Eigen::Index> idxQ;
    for (Eigen::Index i = 0; i < density.dim(); ++i) {
        idxQ.push_back(i);
    }
    return idxQ;
}

cv::Mat & SystemSLAM::view()
{
    return view_;
};

const cv::Mat & SystemSLAM::view() const
{
    return view_;
};

GaussianInfo<double> SystemSLAM::bodyPositionDensity() const
{
    return density.marginal(Eigen::seqN(6, 3));
}

GaussianInfo<double> SystemSLAM::bodyOrientationDensity() const
{
    return density.marginal(Eigen::seqN(9, 3));
}

GaussianInfo<double> SystemSLAM::bodyTranslationalVelocityDensity() const
{
    return density.marginal(Eigen::seqN(0, 3));
}

GaussianInfo<double> SystemSLAM::bodyAngularVelocityDensity() const
{
    return density.marginal(Eigen::seqN(3, 3));
}

Eigen::Vector3d SystemSLAM::cameraPosition(const Camera & camera, const Eigen::VectorXd & x, Eigen::MatrixXd & J)
{
    Eigen::Vector3<autodiff::dual> rCNn_dual;
    Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();
    J = jacobian(cameraPosition<autodiff::dual>, wrt(x_dual), at(camera, x_dual), rCNn_dual);
    return rCNn_dual.cast<double>();
};

GaussianInfo<double> SystemSLAM::cameraPositionDensity(const Camera & camera) const
{
    auto f = [&](const Eigen::VectorXd & x, Eigen::MatrixXd & J) { return cameraPosition(camera, x, J); };
    return density.affineTransform(f);
}

Eigen::Vector3d SystemSLAM::cameraOrientationEuler(const Camera & camera, const Eigen::VectorXd & x, Eigen::MatrixXd & J)
{
    Eigen::Vector3<autodiff::dual> Thetanc_dual;
    Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();
    J = jacobian(cameraOrientationEuler<autodiff::dual>, wrt(x_dual), at(camera, x_dual), Thetanc_dual);
    return Thetanc_dual.cast<double>();
};

GaussianInfo<double> SystemSLAM::cameraOrientationEulerDensity(const Camera & camera) const
{
    auto f = [&](const Eigen::VectorXd & x, Eigen::MatrixXd & J) { return cameraOrientationEuler(camera, x, J); };
    return density.affineTransform(f);    
}

GaussianInfo<double> SystemSLAM::landmarkPositionDensity(std::size_t idxLandmark) const
{
    assert(idxLandmark < numberLandmarks());
    std::size_t idx = landmarkPositionIndex(idxLandmark);
    return density.marginal(Eigen::seqN(idx, 3));
}

void SystemSLAM::addLandmark(int markerId)
{
    landmarkIds_.push_back(markerId);
}

int SystemSLAM::getLandmarkId(std::size_t idxLandmark) const
{
    if (idxLandmark < landmarkIds_.size()) {
        return landmarkIds_[idxLandmark];
    }
    return -1;  // Return -1 if landmark index is out of range
}

std::size_t SystemSLAM::getLandmarkIndex(int markerId) const
{
    auto it = std::find(landmarkIds_.begin(), landmarkIds_.end(), markerId);
    if (it != landmarkIds_.end()) {
        return std::distance(landmarkIds_.begin(), it);
    }
    return static_cast<std::size_t>(-1);  // Return -1 if marker ID is not found
}

