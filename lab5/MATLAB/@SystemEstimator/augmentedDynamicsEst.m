% Evaluate F(t, X, u) from dX = F(t, X, u)*dt + dW
function F = augmentedDynamicsEst(obj, t, X, u)

nx = size(X, 1);
assert(size(X, 2) == 2*nx + 1);
x = X(:, 1);
[f, J] = obj.dynamicsEst(t, x, u);
assert(size(f, 1) == nx);
assert(size(J, 1) == nx);
assert(size(J, 2) == nx);
F = [f, J*X(:, 2:end)];

