#include <cmath>
#include <cassert>
#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemEstimator.h"
#include "SystemBallistic.h"

const double SystemBallistic::p0 = 101.325e3;            // Air pressure at sea level [Pa]
const double SystemBallistic::M  = 0.0289644;            // Molar mass of dry air [kg/mol]
const double SystemBallistic::R  = 8.31447;              // Gas constant [J/(mol.K)]
const double SystemBallistic::L  = 0.0065;               // Temperature gradient [K/m]
const double SystemBallistic::T0 = 288.15;               // Temperature at sea level [K]
const double SystemBallistic::g  = 9.81;                 // Acceleration due to gravity [m/s^2]

SystemBallistic::SystemBallistic(const GaussianInfo<double> & density)
    : SystemEstimator(density)
{}

GaussianInfo<double> SystemBallistic::processNoiseDensity(double dt) const
{
    // SQ is an upper triangular matrix such that SQ.'*SQ = Q is the power spectral density of the continuous time process noise
    Eigen::MatrixXd SQ(2, 2);
    // TODO
    SQ.setZero();

    // Set the non-zero elements of SQ based on the given Q matrix
    SQ(0, 0) = std::sqrt(1e-20);  // For velocity
    SQ(1, 1) = std::sqrt(25e-12); // For drag coefficient
    
    // Distribution of noise increment dw ~ N(0, Q*dt) for time increment dt
    return GaussianInfo<double>::fromSqrtMoment(SQ*std::sqrt(dt));
}

std::vector<Eigen::Index> SystemBallistic::processNoiseIndex() const
{
    // Indices of process model equations where process noise is injected
    std::vector<Eigen::Index> idxQ;
    // TODO: Continuous-time process noise in 2nd and 3rd state equations
    idxQ = {1, 2};  // Noise affects velocity (index 1) and drag coefficient (index 2)
    return idxQ;
}

// Evaluate f(x) from the SDE dx = f(x)*dt + dw
Eigen::VectorXd SystemBallistic::dynamics(const Eigen::VectorXd & x) const
{
    Eigen::VectorXd f(x.size());
    // TODO: Set f
    // Extract state variables
    double h = x(0);  // altitude
    double v = x(1);  // velocity
    double c = x(2);  // ballistic drag coefficient

    // // Calculate temperature at altitude h
    double T = T0 - L * h;

    // // Calculate air pressure using the barometric formula
    // double p = p0 * std::pow(1 - L * h / T0, g * M / (R * L));

    // Calculate drag acceleration
    double d = ((0.5 * M * p0) / R) * (1/T) * std::pow(1 - L * h / T0, g * M / (R * L)) * v * v * c;

    // Set f according to the dynamics equations
    f(0) = v;                 // dh/dt = v
    f(1) = d - g;             // dv/dt = d - g
    f(2) = 0;                 // dc/dt = 0 (drag coefficient is constant)
    return f;
}

// Evaluate f(x) and its Jacobian J = df/fx from the SDE dx = f(x)*dt + dw
Eigen::VectorXd SystemBallistic::dynamics(const Eigen::VectorXd & x, Eigen::MatrixXd & J) const
{
    Eigen::VectorXd f = dynamics(x);

    J.resize(f.size(), x.size());
    // TODO: Set J
    // Extract state variables
    double h = x(0);  // altitude
    double v = x(1);  // velocity
    double c = x(2);  // ballistic drag coefficient

    // Calculate temperature at altitude h
    double T = T0 - L * h;

    // Calculate air pressure using the barometric formula
    //double p = p0 * std::pow(1 - L * h / T0, g * M / (R * L));

    // Calculate drag acceleration
    double d = ((0.5 * M * p0) / R) * (1/T) * std::pow(1 - L * h / T0, g * M / (R * L)) * v * v * c;

    // Calculate partial derivatives
    double dd_dh = ((0.5 * M * p0) / R) * v * v * c * (L / std::pow(T0 - L * h, 2) * std::pow(1 - L * h / T0, g * M / (R * L)) - (g * M / (R * T0)) * std::pow(1 - L * h / T0, (g * M / (R * L)) - 1) / (T0 - L * h));
    double dd_dv = 2 * d / v;
    double dd_dc = d / c;

    J.resize(f.size(), x.size());
    J << 0,    1,    0,
         dd_dh, dd_dv, dd_dc,
         0,    0,    0;

    return f;
}
