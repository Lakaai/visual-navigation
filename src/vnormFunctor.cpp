#include <Eigen/Core>
#include "vnorm.hpp"
#include "vnormFunctor.h"

// Functor for vector norm and its derivatives
double VectorNorm::operator()(const Eigen::VectorXd &x)
{
    return vnorm(x);
}

double VectorNorm::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g)
{
    double u = operator()(x);
    Eigen::Index n = x.size();
    g.resize(n,1);
    g = x/u;
    return u;
}

double VectorNorm::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g, Eigen::MatrixXd &H)
{
    double u = operator()(x, g);
    Eigen::Index n = x.size();
    H.resize(n,n);
    H = Eigen::MatrixXd::Identity(n,n)/u - 1.0/(u*u*u)*x*x.transpose();
    return u;
}
