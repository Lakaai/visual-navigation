#ifndef VNORMFUNCTORFAD_H
#define VNORMFUNCTORFAD_H

#include <Eigen/Core>

// Functor for vector norm and its derivatives using forward-mode autodifferentiation
struct VectorNormFwdAutoDiff
{
    double operator()(const Eigen::VectorXd &x);
    double operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g);
    double operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g, Eigen::MatrixXd &H);
};

#endif
