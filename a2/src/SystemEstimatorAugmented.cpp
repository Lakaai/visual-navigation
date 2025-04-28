#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemEstimatorAugmented.h"


SystemEstimatorAugmented::SystemEstimatorAugmented(const GaussianInfo<double> & density)
    : SystemBase()
    , density(density)
{}

SystemEstimatorAugmented::~SystemEstimatorAugmented() = default;

void SystemEstimatorAugmented::predict(double time)
{
    double dt = time - time_;
    assert(dt >= 0);
    if (dt == 0.0) return;

    // Augment state density with independent noise increment dw ~ N^{-1}(0, LambdaQ/dt)
    // [ x] ~ N^{-1}([ eta ], [ Lambda,          0 ])
    // [dw]         ([   0 ]  [      0, LambdaQ/dt ])

    auto pdw = processNoiseDensity(dt); // p(dw(idxQ)[k])
    auto pxdw = density*pdw;            // p(x[k], dw(idxQ)[k]) = p(x[k])*p(dw(idxQ)[k])

    // Map [ x[k]; dw(idxQ)[k] ] to x[k+1]
    auto func = [&](const Eigen::VectorXd & xdw, Eigen::MatrixXd & J) { return RK4SDEHelper(xdw, dt, J); };
    
    // Map p(x[k], dw(idxQ)[k]) to p(x[k+1])
    density = pxdw.affineTransform(func);

    time_ = time;
}

Eigen::VectorXd SystemEstimatorAugmented::dynamicsEst(const Eigen::VectorXd & x) const
{
    return dynamics(x);
}

Eigen::VectorXd SystemEstimatorAugmented::dynamicsEst(const Eigen::VectorXd & x, Eigen::MatrixXd & J) const
{
    return dynamics(x, J);
}

// Evaluate F(X) from dX = F(X)*dt + dW
Eigen::MatrixXd SystemEstimatorAugmented::augmentedDynamicsEst(const Eigen::MatrixXd & X) const
{
    assert(X.size() > 0);
    int nx = X.rows();
    assert(X.cols() == 2*nx + 1);

    Eigen::VectorXd x = X.col(0);
    Eigen::MatrixXd J;
    Eigen::VectorXd f = dynamicsEst(x, J);
    assert(f.rows() == nx);
    assert(J.rows() == nx);
    assert(J.cols() == nx);

    Eigen::MatrixXd dX(nx, 2*nx + 1);
    dX << f, J*X.block(0, 1, nx, 2*nx);
    return dX;
}

// Map [x[k]; dw(idxQ)[k]] to x[k+1] using RK4
Eigen::VectorXd SystemEstimatorAugmented::RK4SDEHelper(const Eigen::VectorXd & xdw, double dt, Eigen::MatrixXd & J) const
{
    const std::vector<Eigen::Index> & idxQ = processNoiseIndex();

    const std::size_t nx = density.dim();
    const std::size_t nq = idxQ.size();
    // assert(xdw.size() == nx + nq);
    assert(static_cast<std::size_t>(xdw.size()) == nx + nq);


    Eigen::VectorXd x(nx), dw(nx);
    x = xdw.head(nx);
    dw.setZero();
    dw(idxQ) = xdw.tail(nq);

    typedef Eigen::MatrixXd Matrix;

    // X  = [ x,  dx/dx[k],   dx/dw[k] ]
    // dW = [dw, ddw/dx[k], ddw/ddw[k] ]
    Matrix X(nx, 2*nx+1), dW(nx, 2*nx+1);
    X  <<  x, Matrix::Identity(nx, nx), Matrix::Zero(nx, nx);
    dW << dw,     Matrix::Zero(nx, nx), Matrix::Identity(nx, nx);

    Matrix F1, F2, F3, F4, Xnext;
    F1 = augmentedDynamicsEst(                 X);
    F2 = augmentedDynamicsEst(X + (F1*dt + dW)/2);
    F3 = augmentedDynamicsEst(X + (F2*dt + dW)/2);
    F4 = augmentedDynamicsEst(    X + F3*dt + dW);
    Xnext = X + (F1 + 2*F2 + 2*F3 + F4)*dt/6 + dW;
    
    // X[k+1] = [ x[k+1], dx[k+1]/dx[k], dx[k+1]/dw[k] ]
    J.resize(nx, nx + nq);
    J << Xnext.middleCols(1, nx), Xnext.middleCols(nx + 1, nx)(Eigen::all, idxQ);
    assert(!J.hasNaN() && "NaN found in Jacobian matrix inside RK$SDEHelper.");
    return Xnext.col(0);
}
