// Tip: Only include headers needed to parse this implementation only
#include <cassert>
#include <Eigen/Core>
#include <Eigen/QR>
#include "Gaussian.h"

Gaussian::Gaussian()
{

}

Gaussian::Gaussian(const Eigen::VectorXd & mu, const Eigen::MatrixXd & S)
    : mu_(mu)
    , S_(S)
{
    assert(mu_.size() == S_.cols());
    assert(S_.isUpperTriangular());
}

Eigen::VectorXd Gaussian::mean() const
{
    return mu_;
}

Eigen::MatrixXd Gaussian::sqrtCov() const
{
    return S_;
}

Eigen::MatrixXd Gaussian::cov() const
{
    return S_.transpose()*S_;
}

// TODO: Add member function implementations
Gaussian Gaussian::add(const Gaussian & other) const
{
    // Sum the means
    Eigen::VectorXd new_mu = mu_ + other.mu_;

    // Prepare the matrix for QR decomposition
    Eigen::MatrixXd A(S_.rows() + other.S_.rows(), S_.cols());
    A << S_, other.S_;

    // Perform QR decomposition
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
    
    // Extract the upper triangular matrix R
    Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();

    // The new S is the top square part of R
    Eigen::MatrixXd new_S = R.topRows(S_.cols());

    return Gaussian(new_mu, new_S);
} 

Gaussian Gaussian::marginalHead(int na) const
{

    // Extract the first na elements of the mean vector
    Eigen::VectorXd mua = mu_.head(na);

    // Extract the top-left na x na block of the square root covariance matrix
    Eigen::MatrixXd Sa = S_.topLeftCorner(na, na);

    // Return a new Gaussian object with the marginal distribution
    return Gaussian(mua, Sa);
}

Gaussian Gaussian::conditionalTailGivenHead(const Eigen::VectorXd & xa) const
{
    int na = xa.size();
    int nb = mu_.size() - na;

    // Sanity check
    assert(na + nb == mu_.size() && "Dimensions mismatch");

    // Extract the relevant blocks from S_
    Eigen::MatrixXd S1 = S_.topLeftCorner(na, na);
    Eigen::MatrixXd S2 = S_.topRightCorner(na, nb);
    Eigen::MatrixXd S3 = S_.bottomRightCorner(nb, nb);

    // Compute the conditional mean
    Eigen::VectorXd mub_a = mu_.tail(nb) + S2.transpose() * S1.triangularView<Eigen::Upper>().solve(xa - mu_.head(na));

    // The conditional square root covariance is simply S3
    Eigen::MatrixXd Sb_a = S3;

    return Gaussian(mub_a, Sb_a);
}

Gaussian Gaussian::permute(const Eigen::ArrayXi & idx) const
{
    // Permute the mean vector
    Eigen::VectorXd new_mu = mu_(idx);

    // Extract the permuted columns of S
    Eigen::MatrixXd S_perm = S_(Eigen::all, idx);

    // Perform QR decomposition
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(S_perm);
    
    // Extract the upper triangular matrix R
    Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();

    // The new S is the top square part of R
    Eigen::MatrixXd new_S = R.topRows(idx.size());

    return Gaussian(new_mu, new_S);
}