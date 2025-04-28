#include <iostream>
#include <Eigen/Core>
#include "Gaussian.h"

int main() {
    // Create two Gaussian distributions
    Eigen::VectorXd mu1(2);
    mu1 << 1.0, 2.0;
    Eigen::MatrixXd S1(2, 2);
    S1 << 1.0, 0.5,
          0.0, 1.0;
    Gaussian g1(mu1, S1);

    Eigen::VectorXd mu2(2);
    mu2 << 3.0, 4.0;
    Eigen::MatrixXd S2(2, 2);
    S2 << 2.0, 0.0,
          0.0, 2.0;
    Gaussian g2(mu2, S2);

    // Add the Gaussians
    Gaussian result = g1.add(g2);

    // Print results
    std::cout << "Result mean:\n" << result.mean() << std::endl;
    std::cout << "Result sqrt covariance:\n" << result.sqrtCov() << std::endl;
    std::cout << "Result covariance:\n" << result.cov() << std::endl;

    return 0;
}