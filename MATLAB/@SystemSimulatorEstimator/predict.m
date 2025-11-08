function systemNext = predict(obj, timeNext)

dt = timeNext - obj.time;
assert(dt >= 0);

systemNext = obj;    % Copy system state (System has value semantics since it doesn't inherit from handle)

if dt < 1e-14
    % Nothing to do
    return;
end

% Update time stamp
systemNext.time = timeNext;

% Update simulator state
options = odeset('MaxStep', 0.1);

% Simulate system with input signal
func = @(t, x) obj.dynamicsSim(t, x, obj.input(t));
[~, x_] = ode45(func, [obj.time timeNext], obj.x_sim, options);
x_ = x_.';
systemNext.x_sim = x_(:, end);

% Update estimator state
if obj.runEstimator
    % Augment state density with independent noise increment dw ~ N(0, Q*dt)
    % [  x ] ~ N([ mu ], [ P,    0 ])
    % [ dw ]    ([  0 ]  [ 0, Q*dt ])
    [pdw, idxQ] = obj.processNoise(dt);
    pxdw = obj.density.join(pdw);   % p(x[k], dw(idxQ)[k]) = p(x[k])*p(dw(idxQ)[k])
    
    % Propagate p(x[k], dw(idxQ)[k]) through RK4 to obtain p(x[k+1])
    t = obj.time;
    u = obj.input(t);
    func = @(xdw) obj.RK4SDEHelper(t, xdw, u, dt, idxQ);
    systemNext.density = pxdw.affineTransform(func);    % p(x[k+1])
end
