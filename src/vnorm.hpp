#ifndef VNORM_HPP
#define VNORM_HPP

#include <cmath>
#include <Eigen/Core>

using std::sqrt;

// Templated implementation of vector norm
template<typename Scalar>
static Scalar vnorm(const Eigen::Matrix<Scalar,Eigen::Dynamic,1> &x)
{   
    return sqrt(x.cwiseProduct(x).sum());
}

#endif