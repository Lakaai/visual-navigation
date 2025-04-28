classdef SystemEstimator < SystemBase
    properties
        density GaussianInfo                    % Estimator state
        runEstimator (1, 1) logical = true      % Run state estimator
    end

    methods
        [f, Jx] = dynamicsEst(obj, t, x, u)
        F = augmentedDynamicsEst(obj, t, X, u)
        [xnext, J] = RK4SDEHelper(obj, t, xdw, u, dt, idxQ)
        systemNext = predict(obj, timeNext)
    end

    methods (Abstract)
        [pdw, idx] = processNoise(obj, dt)
    end
end
