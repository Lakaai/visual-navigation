classdef SystemSimulatorEstimator < SystemEstimator
    properties
        x_sim (:, 1) double         % Simulator state
    end

    methods
        u = input(obj, t)
        [f, Jx, Ju] = dynamicsSim(obj, t, x, u)
        systemNext = predict(obj, timeNext)
    end
end
