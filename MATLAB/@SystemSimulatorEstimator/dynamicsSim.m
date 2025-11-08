function [f, Jx, Ju] = dynamicsSim(obj, t, x, u)

% Default implementation reuses dynamics
switch nargout
    case 1
        f = obj.dynamics(t, x, u);
    case 2
        [f, Jx] = obj.dynamics(t, x, u);
    case 3
        [f, Jx, Ju] = obj.dynamics(t, x, u);
end
