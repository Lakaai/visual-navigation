#include <Eigen/Core>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include "rosenbrock.h"

// Templated version of Rosenbrock function
template <typename Scalar = double>
static Scalar rosenbrock(const Eigen::VectorX<Scalar> & x)
{   
    Scalar x2 = x(0)*x(0);
    Scalar ymx2 = x(1) - x2;
    Scalar xm1 = x(0) - 1;
    return (xm1*xm1 + 100*ymx2*ymx2);
}

// Functor for Rosenbrock function and its derivatives
double RosenbrockAnalytical::operator()(const Eigen::VectorXd & x)
{
    return rosenbrock(x);
}

double RosenbrockAnalytical::operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g)
{    
    // Compute analytical gradient g
    g.resize(2, 1);
    // TODO
    double x1 = x(0);
    double x2 = x(1);
    
    g(0) = -2 + 2*x1 - 400*x1*x2 + 400*std::pow(x1, 3);
    g(1) = 200 * (x2 - x1*x1);

    return operator()(x);
}

double RosenbrockAnalytical::operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H)
{
    // Compute analytical Hessian H
    H.resize(2, 2);
    // TODO
    H.resize(2, 2);
    
    double x1 = x(0);
    double x2 = x(1);
    
    H(0, 0) = 2 - 400*x2 + 1200*x1*x1;
    H(0, 1) = H(1, 0) = -400*x1;
    H(1, 1) = 200;
    
    return operator()(x, g);
}

// Functor for Rosenbrock function and its derivatives using forward-mode autodifferentiation
double RosenbrockFwdAutoDiff::operator()(const Eigen::VectorXd & x)
{   
    return rosenbrock(x);
}

double RosenbrockFwdAutoDiff::operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g)
{
    // Forward-mode autodifferentiation
    Eigen::VectorX<autodiff::dual> xdual = x.cast<autodiff::dual>();
    autodiff::dual fdual;
    g = gradient(rosenbrock<autodiff::dual>, wrt(xdual), at(xdual), fdual);
    return val(fdual);
}

double RosenbrockFwdAutoDiff::operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H)
{
    // Forward-mode autodifferentiation
    Eigen::VectorX<autodiff::dual2nd> xdual = x.cast<autodiff::dual2nd>();
    autodiff::dual2nd fdual;
    H = hessian(rosenbrock<autodiff::dual2nd>, wrt(xdual), at(xdual), fdual, g);
    return val(fdual);
}

// Functor for Rosenbrock function and its derivatives using reverse-mode autodifferentiation
double RosenbrockRevAutoDiff::operator()(const Eigen::VectorXd & x)
{   
    return rosenbrock(x);
}

double RosenbrockRevAutoDiff::operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g)
{
    // Reverse-mode autodifferentiation
    Eigen::VectorX<autodiff::var> xvar = x.cast<autodiff::var>();
    autodiff::var fvar = rosenbrock(xvar);
    g = gradient(fvar, xvar); // TODO
    return val(fvar);
}

double RosenbrockRevAutoDiff::operator()(const Eigen::VectorXd & x, Eigen::VectorXd & g, Eigen::MatrixXd & H)
{
    // Reverse-mode autodifferentiation
    Eigen::VectorX<autodiff::var> xvar = x.cast<autodiff::var>();
    autodiff::var fvar = rosenbrock(xvar);
    H = hessian(fvar, xvar, g); // TODO 
    return val(fvar);
}