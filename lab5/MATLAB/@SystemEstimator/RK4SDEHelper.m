% This helper function evaluates the map
% [         x[k] ] |---> [ x[k+1] ]
% [  dw(idxQ)[k] ] |
% for one step of RK4 integration.
function [xnext, J] = RK4SDEHelper(obj, t, xdw, u, dt, idxQ)
nx = obj.density.dim();             % Dimension of state
nq = length(idxQ);                  % Dimension of process noise
assert(length(xdw) == nx + nq);
x = xdw(1:nx);
dw = zeros(nx, 1);
dw(idxQ) = xdw(nx+1:end);

if nargout < 2
    f1 = obj.dynamicsEst(t, x, u);
    f2 = obj.dynamicsEst(t, x + (f1*dt + dw)/2, u);
    f3 = obj.dynamicsEst(t, x + (f2*dt + dw)/2, u);
    f4 = obj.dynamicsEst(t, x + f3*dt + dw, u);
    xnext = x + (f1 + 2*f2 + 2*f3 + f4)*dt/6 + dw;
else
    % X(t)  = [ x(t),  dx(t)/dx[k],   dx(t)/dw[k] ]
    % dW(t) = [dw(t), ddw(t)/dx[k], ddw(t)/ddw[k] ]
    X = [x, eye(nx), zeros(nx)];    % X[k]  = [ x[k],  dx[k]/dx[k],   dx[k]/dw[k] ]
    dW = [dw, zeros(nx), eye(nx)];  % dW[k] = [dw[k], ddw[k]/dx[k], ddw[k]/ddw[k] ]
    F1 = obj.augmentedDynamicsEst(t, X, u);
    F2 = obj.augmentedDynamicsEst(t, X + (F1*dt + dW)/2, u);
    F3 = obj.augmentedDynamicsEst(t, X + (F2*dt + dW)/2, u);
    F4 = obj.augmentedDynamicsEst(t, X +  F3*dt + dW, u);
    % X[k+1] = [ x[k+1], dx[k+1]/dx[k], dx[k+1]/dw[k] ]
    Xnext = X + (F1 + 2*F2 + 2*F3 + F4)*dt/6 + dW;
    xnext = Xnext(:, 1);
    Jdx = Xnext(:, 2:1+nx);
    Jdw = Xnext(:, 2+nx:end);
    J = [Jdx, Jdw(:, idxQ)];
end
