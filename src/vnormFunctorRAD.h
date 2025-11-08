#ifndef VNORMFUNCTORRAD_H
#define VNORMFUNCTORRAD_H

#include <Eigen/Core>

// Functor for vector norm and its derivatives using reverse-mode autodifferentiation
struct VectorNormRevAutoDiff
{
    double operator()(const Eigen::VectorXd &x);
    double operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g);
    double operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g, Eigen::MatrixXd &H);
};

#endif
