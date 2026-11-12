function state = dde_csp_gibbs(state,B)
%DDE_CSP_GIBBS Partially collapsed CSP augmentation with fixed loading matrix.
% z_h in 1:(K+1), active iff z_h>h. Slab scales are integrated out for the
% categorical update, then every slab scale is refreshed conditional on z.
% Required order: z|B,v (collapsed), v|z, slabScale|B,z. Retaining stale slab
% scales after a collapsed z update would invalidate this augmented sampler.
% Effective prior density is Laplace(B(:,h+1);0,theta(h)); the coefficient
% M-step uses precision(h)=1/theta(h). Intercepts are never penalized.
K=state.K; m=size(B,1);
assert(ismatrix(B) && isequal(size(B),[state.rows,K+1]) && all(isfinite(B(:))), ...
    'DDE:CSPMatrix','B dimensions must match the CSP state.');
assert(isscalar(state.spike) && isfinite(state.spike) && state.spike>0 && ...
    isscalar(state.alpha) && isfinite(state.alpha) && state.alpha>0 && ...
    isscalar(state.shape) && isfinite(state.shape) && state.shape>0 && ...
    isscalar(state.rate) && isfinite(state.rate) && state.rate>0, ...
    'DDE:CSPPrior','Invalid CSP prior parameters.');
assert(numel(state.v)==K+1 && state.v(end)==1 && ...
    all(state.v(1:K)>0 & state.v(1:K)<1), ...
    'DDE:CSPStick','Sticks must be interior, with a remainder stick of one.');
state.v=reshape(state.v,1,[]);
logOmega=stick_logweights(state.v);
sumAbs=sum(abs(B(:,2:end)),1);
logSpike=-m*log(2*state.spike)-sumAbs/state.spike;
logSlab=state.shape*log(state.rate)-m*log(2) ...
    +gammaln(state.shape+m)-gammaln(state.shape) ...
    -(state.shape+m)*log(state.rate+sumAbs);
probability=zeros(K,K+1); z=zeros(1,K);
for h=1:K
    logDensity=logSlab(h)*ones(1,K+1); logDensity(1:h)=logSpike(h);
    logProbability=logOmega+logDensity;
    scaled=exp(logProbability-max(logProbability)); scaled=scaled/sum(scaled);
    probability(h,:)=scaled;
    cumulative=cumsum(scaled); cumulative(end)=1;
    z(h)=find(rand<cumulative,1,'first');
end
% Refresh the sticks conditional on the newly sampled categories.
stickShapeA=ones(1,K); stickShapeB=zeros(1,K);
v=ones(1,K+1);
for l=1:K
    stickShapeA(l)=1+sum(z==l);
    stickShapeB(l)=state.alpha+sum(z>l);
    draw=betarnd(stickShapeA(l),stickShapeB(l)); attempts=0;
    while ~(draw>0 && draw<1)
        attempts=attempts+1;
        assert(attempts<100,'DDE:CSPNumeric','Could not draw an interior posterior stick.');
        draw=betarnd(stickShapeA(l),stickShapeB(l));
    end
    v(l)=draw;
end
active=z>(1:K);
slabShape=state.shape+m*double(active);
slabRate=state.rate+sumAbs.*double(active);
slabScale=zeros(1,K);
for h=1:K
    precision=gamrnd(slabShape(h),1/slabRate(h)); attempts=0;
    while ~(isfinite(precision) && precision>0 && isfinite(1/precision))
        attempts=attempts+1;
        assert(attempts<100,'DDE:CSPNumeric','Could not draw a finite positive slab scale.');
        precision=gamrnd(slabShape(h),1/slabRate(h));
    end
    slabScale(h)=1/precision;
end
theta=state.spike*ones(1,K); theta(active)=slabScale(active);
state.z=z; state.v=v; state.logOmega=stick_logweights(v);
state.omega=exp(state.logOmega);
state.active=active; state.slabScale=slabScale; state.theta=theta;
state.precision=1./theta;
state.iterations=state.iterations+1;
state.diagnostics=struct('categoryProbability',probability, ...
    'categoryConditioningLogOmega',logOmega,'logSpikeDensity',logSpike, ...
    'logIntegratedSlabDensity',logSlab,'stickShapeA',stickShapeA, ...
    'stickShapeB',stickShapeB,'slabShape',slabShape,'slabRate',slabRate);
assert(all(z>=1 & z<=K+1) && all(z==fix(z)) && ...
    all(isfinite(theta) & theta>0) && abs(sum(state.omega)-1)<1e-12, ...
    'DDE:CSPInvalidState','CSP update returned invalid augmentation.');
end

function logOmega=stick_logweights(v)
K=numel(v)-1;
prefix=[0,cumsum(log1p(-v(1:K)))];
logOmega=log(v)+prefix;
end
