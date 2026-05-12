"""
Gaussian utilities for constructing and performing operations on multivariate Gaussian distributions. 
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Callable

class Gaussian:
    """
    Represents a multivariate Gaussian distribution parameterised by a mean vector and covariance matrix.

    The distribution is defined as:
        x ~ N(μ, Σ)

    where:
        μ : mean vector of shape (n,)
        Σ : covariance matrix of shape (n, n)

    :param mean: Mean vector of the Gaussian distribution.
    :type mean: np.ndarray
    :param covariance: Covariance matrix of the Gaussian distribution.
    :type covariance: np.ndarray
    """
    def __init__(self, mean: np.ndarray, covariance: np.ndarray):
        self.mean = mean
        self.covariance = covariance

    @classmethod
    def from_moment(cls, mu: np.ndarray, P: np.ndarray) -> "Gaussian":
        """
        Construct a Gaussian distribution from mean and covariance matrix.

        :param mu: The mean vector of the Gaussian distribution.
        :type mu: np.ndarray
        :param P: The covariance matrix of the Gaussian distribution.
        :type P: np.ndarray
        :return: A Gaussian distribution.
        :rtype: Gaussian

        """
        return cls(mu, P)

    def marginal(self, idx: np.ndarray) -> "Gaussian":
        """
        Extract the marginal distribution over the variables specified by the index set `idx`.

        Reference: https://psesh.github.io/ae8803/useful_codes/gaussians.html

        :param idx: The index set for the variables to marginalise over.
        :type idx: np.ndarray

        """

        mu = self.mean[np.ix_(idx)]
        P = self.covariance[np.ix_(idx, idx)]

        return self.from_moment(mu, P)

    def conditional(self, idx_x, idx_y, y) -> "Gaussian":
        """

        Given the joint Gaussian N(μ, Σ) and index sets for variables A and B, return p(x | y).
        The joint distribution passed to this function must be in the form p([y; x]) and not p([x; y]).

        :param idx_x: The index set for the variables in x.
        :type idx_x: np.ndarray
        :param idx_y: The index set for the variables in y.
        :type idx_y: np.ndarray
        :param y: The observed value for y.

        """

        mu = self.mean
        sigma = self.covariance

        mu_x = mu[idx_x]
        mu_y = mu[idx_y]

        sigma_xx = sigma[np.ix_(idx_x, idx_x)]
        sigma_xy = sigma[np.ix_(idx_x, idx_y)]
        sigma_yx = sigma[np.ix_(idx_y, idx_x)]
        sigma_yy = sigma[np.ix_(idx_y, idx_y)]

        # Expand dimension if necessary, np.linalg.solve requires 2 dimensional array for the first argument
        sigma_yy = np.atleast_2d(sigma_yy)

        # Compute the new mean and covariance of the conditional distribution 𝑝(𝑥 | 𝑦)
        # Dont invert the matrix (Σ𝑦𝑦⁻¹) - https://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix/

        # Instead, solve the linear system Σ𝑦𝑦 * w = v to find w = Σ𝑦𝑦⁻¹ * v
        w = np.linalg.solve(sigma_yy, y - mu_y)

        # Compute the conditional mean μ𝑥 | 𝑦 = μ𝑥 + Σ𝑥𝑦 * Σ𝑦𝑦⁻¹ * (𝑦 - μ𝑦)
        mu_cond = mu_x + sigma_xy @ w

        # Again solve the linear system Σ𝑦𝑦 * w = Σ𝑦𝑥 to find w = Σ𝑦𝑦⁻¹ * Σ𝑦𝑥
        w = np.linalg.solve(sigma_yy, sigma_yx)

        # Compute the conditional covariance Σ𝑥|𝑦 = Σ𝑥𝑥 - Σ𝑥𝑦 * Σ𝑦𝑦⁻¹ * Σ𝑦𝑥
        sigma_cond = sigma_xx - sigma_xy @ w

        # Return the conditional distribution 𝑝(𝑥 | 𝑦)
        return Gaussian.from_moment(mu_cond, sigma_cond)

    def affine_transform(self, func: Callable) -> "Gaussian":
        """
        Apply an affine transformation to propagate the Gaussian distribution through a nonlinear function.

        Computes the gradient of the function using JAX autodiff.

        :param func: A function through which to propagate the Gaussian distribution.
        :type func: Callable
        :return: The transformed Gaussian distribution.
        :rtype: Gaussian
        """

        mu = jnp.array(self.mean)
        P = jnp.array(self.covariance)

        muy = func(mu)

        J = jax.jacfwd(func)(mu)

        Py = J @ P @ J.T

        return Gaussian.from_moment(muy, Py)

    def unscented_transform(self, func: Callable) -> "Gaussian":
        """
        Apply the Unscented Transform to propagate the Gaussian distribution through a nonlinear function.

        :param func: A function through which to propagate the Gaussian distribution.
        :type func: Callable
        :return: The transformed Gaussian distribution.
        :rtype: Gaussian
        """
        kappa = 0
        alpha = 1
        beta = 2
        nx = len(self.mean)

        lambda_ = alpha**2 * (nx + kappa) - nx

        Sx = np.linalg.cholesky((nx + lambda_) * self.covariance)

        # Generate sigma points from the mean and covariance
        chi = np.zeros((nx, 2 * nx + 1))

        # The first sigma point is the mean of the input probability distribution
        chi[:, 0] = self.mean

        for i in range(nx):
            chi[:, i + 1] = self.mean + Sx[:, i]
            chi[:, i + 1 + nx] = self.mean - Sx[:, i]

        # Compute the sigma point weights
        mean_weights = np.zeros(2 * nx + 1)
        covariance_weights = np.zeros(2 * nx + 1)

        mean_weights[0] = lambda_ / (nx + lambda_)
        covariance_weights[0] = lambda_ / (nx + lambda_) + 1 - alpha**2 + beta

        for i in range(1, 2 * nx + 1):
            mean_weights[i] = 1 / (2 * (nx + lambda_))
            covariance_weights[i] = mean_weights[i]

        muy = func(chi[:, 0])
        ny = len(muy)

        # Propagate the sigma points through the non-linear function
        transformed_sigma_points = np.zeros((ny, 2 * nx + 1))
        transformed_sigma_points[:, 0] = muy

        for i in range(1, 2 * nx + 1):
            transformed_sigma_points[:, i] = func(chi[:, i])

        # Compute the mean and covariance of the transformed sigma points
        mu = transformed_sigma_points @ mean_weights
        dy = transformed_sigma_points - mu[:, np.newaxis]
        P = dy @ np.diag(covariance_weights) @ dy.T

        # Symmetrise the covariance matrix
        P = (P + P.T) / 2
        return self.from_moment(mu, P)

    def log_pdf(self, x: np.ndarray) -> float:
        """
        Compute the logarithm of a multivariate Gaussian distribution at the value 'x'.

        If coviance is a single dimension np.atleast_2d will force extra dimension making it compatible with np.linalg.slogdet()
        
        :param x: The input vector at which to evaluate the log-likelihood.
        :type x: np.ndarray
        :param distribution: A multivariate Gaussian distribution with mean `μ` and covariance matrix `Σ`.
        :type distribution: Gaussian
        :return: Logarithm of the distribution evaluated at the value 'x'.
        :rtype: float
        """
        
        P = np.atleast_2d(self.covariance)

        n = len(x)

        delta = x - self.mean

        return -(n / 2) * np.log(2 * np.pi) - (1 / 2) * np.linalg.slogdet(P).logabsdet - (1 / 2) * np.dot(delta, np.linalg.solve(P, delta))

    @staticmethod
    def join(distribution_x: "Gaussian", distribution_y: "Gaussian"):
        """
        Form the joint disbribution of two independent Gaussian distributions.

        :param distribution_x: A Gaussian distribution with mean `μ` and covariance matrix `Σ`.
        :type distribution_x: Gaussian
        :param distribution_y: A Gaussian distribution with mean `μ` and covariance matrix `Σ`.
        :type distribution_y: Gaussian
        :return: The joint distribution of the two independent Gaussian distributions.
        :rtype: Gaussian
        """

        mean = np.concatenate((distribution_x.mean, distribution_y.mean))

        nx = np.shape(distribution_x.covariance)
        ny = np.shape(distribution_y.covariance)

        sigma = np.zeros((nx[0] + ny[0], nx[0] + ny[0]))

        sigma[: nx[0], : nx[0]] = distribution_x.covariance
        sigma[nx[0] :, nx[0] :] = distribution_y.covariance

        return Gaussian.from_moment(mean, sigma)


# class Gaussian:
#     def __init__(self, mean: np.ndarray, covariance: np.ndarray):
#         self.mean = mean
#         self.covariance = covariance
    
#     @classmethod
#     def from_moment(cls, mu: np.ndarray, P: np.ndarray) -> "Gaussian":
#         """Construct from Gaussian density from mean and covariance matrix."""
#         return cls(mu, P)

#     @classmethod
#     def from_sqrt_moment(cls, mu: np.ndarray, S: np.ndarray) -> "Gaussian":
#         """Construct Gaussian denisty from mean and square-root covariance (upper triangular S)."""
#         return cls(mu, S)

#     @classmethod
#     def marginal(cls, density: "Gaussian") -> "Gaussian":
#         mu = density.mean
#         P = density.covariance
#         return cls(mu, P)

#     @classmethod 
#     def conditional(cls, distribution: "Gaussian", idx_x, idx_y, y, sqrt: bool) -> "Gaussian":
#         if sqrt:
#             # The conditional distribution of 𝑦 given 𝑥 is given by 𝑝(𝑦 | 𝑥) = 𝑁(μ𝑦 + S₂ᵀS₁⁻ᵀ(𝑥 - μ𝑥), S₃)
#             μ = distribution.mean
#             S = distribution.covariance 

#             # The joint distribution passed to this function must be in the form p([𝑦; 𝑥]) and not p([𝑥; 𝑦]) 
#             # Extract the blocks S₁, S₂, S₃ from S, this assumes that the square-root covariance is stored as S and not Sᵀ
#             S1 = S[idx_x, idx_x]
#             S2 = S[idx_x, idx_y]
#             S3 = S[idx_y, idx_y]

#             # Compute S₁⁻ᵀ(𝑥 - μ𝑥) by solving the linear system S₁ * w = 𝑦 - μ𝑥
#             w = np.linalg.solve(S1, y - mu[idx_x])

#             # Compute the conditional mean μ_cond = μ𝑦 + S₂ᵀS₁⁻ᵀ(𝑥 - μ𝑥)
#             mu_cond = mu[idx_y] + S2.T @ w

#             # Compute the conditional square-root covariance S_cond = S₃, that is the square-root covariance of p(𝑦 | 𝑥)
#             S_cond = S3

#             return Gaussian.from_sqrt_moment(mu_cond, S_cond)
            
#         else:
#             mu = distribution.mean
#             sigma = distribution.covariance

#             mu_x = mu[idx_x]
#             mu_y = mu[idx_y]

#             sigma_xx = sigma[np.ix_(idx_x, idx_x)]
#             sigma_xy = sigma[np.ix_(idx_x, idx_y)]
#             sigma_yx = sigma[np.ix_(idx_y, idx_x)]
#             sigma_yy = sigma[np.ix_(idx_y, idx_y)]

#             # Expand dimension if necessary, np.linalg.solve requires 2 dimensional array for the first argument
#             sigma_yy = np.atleast_2d(sigma_yy)

#             # Compute the new mean and covariance of the conditional distribution 𝑝(𝑥 | 𝑦)
#             # Dont invert the matrix (Σ𝑦𝑦⁻¹) -  https://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix/

#             # Instead, solve the linear system Σ𝑦𝑦 * w = v to find w = Σ𝑦𝑦⁻¹ * v 
#             w = np.linalg.solve(sigma_yy, y - mu_y)

#             # Compute the conditional mean μ𝑥 | 𝑦 = μ𝑥 + Σ𝑥𝑦 * Σ𝑦𝑦⁻¹ * (𝑦 - μ𝑦)
#             mu_cond = mu_x + sigma_xy @ w  

#             # Again solve the linear system Σ𝑦𝑦 * w = Σ𝑦𝑥 to find w = Σ𝑦𝑦⁻¹ * Σ𝑦𝑥
#             w = np.linalg.solve(sigma_yy, sigma_yx)

#             # Compute the conditional covariance Σ𝑥|𝑦 = Σ𝑥𝑥 - Σ𝑥𝑦 * Σ𝑦𝑦⁻¹ * Σ𝑦𝑥
#             sigma_cond = sigma_xx - sigma_xy * w  

#             # Return the conditional distribution 𝑝(𝑥 | 𝑦)
#             return Gaussian.from_moment(mu_cond, sigma_cond)

#     @classmethod
#     def affine_transform(cls, density: "Gaussian") -> "Gaussian":
#         mu = density.mean
#         P = density.covariance
#         return cls(mu, P)
    
#     @classmethod
#     def unscented_transform(cls, func, density: "Gaussian", sqrt: bool = False) -> "Gaussian":
#         kappa = 0
#         alpha = 1
#         beta = 2
#         nx = len(density.mean)

#         lambda_ = alpha**2 * (nx + kappa) - nx
        
#         Sx = np.linalg.cholesky((nx + lambda_) * density.covariance)

#         # Generate sigma points from the mean and covariance
#         chi = np.zeros((nx, 2 * nx + 1))

#         # The first sigma point is the mean of the input probability distribution
#         chi[:, 0] = density.mean
        
#         for i in range(nx):
#             chi[:, i+1] = density.mean + Sx[:, i]
#             chi[:, i+1+nx] = density.mean - Sx[:, i]

#         # Compute the sigma point weights 
#         mean_weights = np.zeros(2 * nx + 1)
#         covariance_weights = np.zeros(2 * nx + 1)

#         mean_weights[0] = lambda_ / (nx + lambda_)
#         covariance_weights[0] = lambda_ / (nx + lambda_) + 1 - alpha**2 + beta

#         for i in range(1, 2 * nx + 1):
#             mean_weights[i] = 1 / (2 * (nx + lambda_))
#             covariance_weights[i] = mean_weights[i]
        
#         muy = func(chi[:, 0])
#         ny = len(muy)

#         # Propagate the sigma points through the non-linear function 
#         transformed_sigma_points = np.zeros((ny, 2 * nx + 1))
#         transformed_sigma_points[:, 0] = muy

#         for i in range(1, 2 * nx + 1):
#             transformed_sigma_points[:, i] = func(chi[:, i])

#         # Compute the mean and covariance of the transformed sigma points
#         mu = transformed_sigma_points @ mean_weights
#         dy = transformed_sigma_points - mu[:, np.newaxis]  
#         P = dy @ np.diag(covariance_weights) @ dy.T 

#         # Symmetrise the covariance matrix
#         P = (P + P.T) / 2

#         return cls(mu, P)