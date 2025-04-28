function b = isWithinConfidenceRegion(obj, x, n_sigma)

arguments
    obj (1, 1) GaussianInfo
    x (:, 1) double
    n_sigma (1, 1) double = 3
end

n = obj.dim();
c = 2*normcdf(n_sigma) - 1;         % Probability mass enclosed by nSigma standard deviations (same as normcdf(n_sigma) - normcdf(-n_sigma))
r2 = chi2inv(c, n);                 % Squared radius in w coords

z = obj.Xi*x - obj.nu;
b = z.'*z <= r2;
