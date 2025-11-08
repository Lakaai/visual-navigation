#ifndef VNORMFUNCTOR_H
#define VNORMFUNCTOR_H

#include <Eigen/Core>

// Functor for vector norm and its derivatives
struct VectorNorm
{
    double operator()(const Eigen::VectorXd &x);
    double operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g);
    double operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g, Eigen::MatrixXd &H);
};

#endif
