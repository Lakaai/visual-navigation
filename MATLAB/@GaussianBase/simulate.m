function X = simulate(obj, m)

%
% Input arguments
%
% obj:      Gaussian object
% m:        number of samples to generate
%
% Output arguments
%
% X:        samples (n x m), where n = obj.dim()
%

if nargin < 2
    m = 1;
end

% Draw m realisations of a Gaussian random variable
n = obj.dim();
X = nan(n, m);
% TODO: Merge from MCHA4100
