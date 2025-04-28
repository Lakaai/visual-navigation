#include <Eigen/Core>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include "vnorm.hpp"
#include "vnormFunctorRAD.h"

// Note: The forward-mode and reverse-mode autodiff implementations are in
//       separate compilation units due to the following known issue with autodiff v0.6:
//       https://github.com/autodiff/autodiff/issues/179

// Functor for vector norm and its derivatives using reverse-mode autodifferentiation
double VectorNormRevAutoDiff::operator()(const Eigen::VectorXd &x)
{   
    return vnorm(x);
}

double VectorNormRevAutoDiff::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g)
{
    // Reverse-mode autodifferentiation
    Eigen::VectorX<autodiff::var> xvar = x.cast<autodiff::var>();
    autodiff::var fvar = vnorm(xvar);
    g = gradient(fvar, xvar);
    return val(fvar);
}

double VectorNormRevAutoDiff::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g, Eigen::MatrixXd &H)
{
    // Reverse-mode autodifferentiation
    Eigen::VectorX<autodiff::var> xvar = x.cast<autodiff::var>();
    autodiff::var fvar = vnorm(xvar);
    H = hessian(fvar, xvar, g);
    return val(fvar);
}