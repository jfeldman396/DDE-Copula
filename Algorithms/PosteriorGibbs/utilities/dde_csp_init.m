function state = dde_csp_init(B,spike,alpha,options)
%DDE_CSP_INIT Fresh CSP augmentation conditional on a current loading matrix.
% B includes its unpenalized intercept in column one. All rows enter the
% independent column priors. No inherited stick/category state is assumed.
% Options shape and rate specify slabScale ~ InvGamma(shape,rate), with
% defaults 1 and 5. An optional seed restores the caller RNG after this call;
% without a seed, sampling consumes the caller's current random stream.
if nargin<4, options=struct; end
assert(isstruct(options) && isscalar(options),'DDE:CSPOption','options must be scalar struct.');
assert(all(ismember(fieldnames(options),{'shape','rate','seed'})), ...
    'DDE:CSPOption','Unknown CSP initialization option.');
if ~isfield(options,'shape'), options.shape=1; end
if ~isfield(options,'rate'), options.rate=5; end
assert(ismatrix(B) && size(B,2)>=1 && all(isfinite(B(:))), ...
    'DDE:CSPMatrix','B must be a finite matrix including the intercept column.');
for value={spike,alpha,options.shape,options.rate}
    assert(isscalar(value{1}) && isfinite(value{1}) && value{1}>0, ...
        'DDE:CSPPrior','CSP prior parameters must be finite positive scalars.');
end
if isfield(options,'seed')
    assert(isscalar(options.seed) && isfinite(options.seed) && options.seed>=0 && ...
        options.seed==fix(options.seed) && options.seed<=2^32-1, ...
        'DDE:CSPSeed','seed must be an unsigned 32-bit integer.');
    callerRng=rng; restoreRng=onCleanup(@()rng(callerRng)); %#ok<NASGU>
    rng(options.seed,'twister');
end
K=size(B,2)-1;
v=ones(1,K+1);
for l=1:K
    draw=betarnd(1,alpha);
    attempts=0;
    while ~(draw>0 && draw<1)
        attempts=attempts+1;
        assert(attempts<100,'DDE:CSPNumeric','Could not draw an interior prior stick.');
        draw=betarnd(1,alpha);
    end
    v(l)=draw;
end
state=struct('K',K,'rows',size(B,1),'spike',spike,'alpha',alpha, ...
    'shape',options.shape,'rate',options.rate,'v',v, ...
    'z',zeros(1,K),'slabScale',ones(1,K),'theta',spike*ones(1,K), ...
    'active',false(1,K),'precision',ones(1,K)/spike,'iterations',0);
% One partially collapsed step is an initialization, not an equilibrium draw.
state=dde_csp_gibbs(state,B);
state.initialization='Fresh prior sticks, then collapsed categories and conditional sticks/scales';
end
