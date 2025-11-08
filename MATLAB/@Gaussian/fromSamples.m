function out = fromSamples(X)

%
% Input arguments
%
% X:        samples (n x m)
%
% where
%   n is the dimension
%   m is the number of samples
%
% Output arguments
%
% out:      Gaussian object
%

[n, m] = size(X);

% Compute the sample mean
mu = nan(n, 1);

% Compute the sample square-root covariance
S = nan(n, n);

out = Gaussian(mu, S);
% TODO: Merge from MCHA4100 or implement a square-root version directly using a QR decomposition

