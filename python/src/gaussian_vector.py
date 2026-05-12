import numpy as np

class GaussianVector:
    """
    Represents a collection of N independent multivariate Gaussian distributions.
    
    mean: (N, D) array where N is number of points, D is dimension.
    covariance: (N, D, D) array of covariance matrices.
    """
    def __init__(self, mean: np.ndarray, covariance: np.ndarray):
        # Ensure mean is (N, D)
        self.mean = np.atleast_2d(mean)
        # Ensure covariance is (N, D, D)
        self.covariance = np.atleast_3d(covariance)
        
        self.N, self.D = self.mean.shape

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the log-likelihood for each point x_i against Gaussian_i.
        x: (N, D) array of observations
        Returns: (N,) array of log-likelihoods
        """

        # x is (2, 1499), self.mean is (2, 1499)
        # Convert to (1499, 2)

        delta = (x - self.mean).T 
        
        cov_single = self.covariance[1]
        
        # 1. Solve for all points at once: (2, 2) \ (1499, 2).T
        # We transpose delta so it's (2, 1499) for the solve, then transpose back
        sol = np.linalg.solve(cov_single, delta)  # Result is (1499, 2)

        # 2. Batch dot product (Mahalanobis distance)
        mahalanobis = np.einsum('ni,ni->n', delta, sol)
        
        # Determinant and Normalization
        _, logdet = np.linalg.slogdet(cov_single)

        D = 2 # Dimension
        normalization = -0.5 * (D * np.log(2 * np.pi) + logdet)
        
        return normalization - 0.5 * mahalanobis

    @classmethod
    def from_single_cov(cls, means: np.ndarray, cov: np.ndarray):
        """
        Helper to create a vector where all points share the same covariance matrix.
        means: (N, D)
        cov: (D, D)
        """
        N = means.shape[0]
        # Broadcast the single cov to (N, D, D)
        covs = np.tile(cov, (N, 1, 1))
        return cls(means, covs)