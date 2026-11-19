function [weights,state,active]=csp_b45_update(B,spike,state,alpha,rho)
%CSP_B45_UPDATE B.4 draws, SAEM sufficient statistics, then B.5 allocation CM.
% Allocations are discrete MAP parameters, not averaged latent indicators.
% Count every coefficient prior factor, including coefficients fixed at zero.
% Intercepts have no prior factor. RHO=1 uses only the current draws.
p=size(B,1);K=size(B,2)-1;L=K+1;
assert(all(isfinite(B),'all')&&spike>0&&alpha>0&&rho>0&&rho<=1,'CSP:B45Input');
if isempty(state)
 state=struct('version','B4_B5_v1','categories',L*ones(1,K),'average',[],'rows',p);
end
assert(isfield(state,'version')&&strcmp(state.version,'B4_B5_v1')&&state.rows==p&&numel(state.categories)==K,'CSP:B45State');
previous=state.categories;assert(all(previous>=1&previous<=L&previous==fix(previous)),'CSP:B45Categories');
mass=sum(abs(B(:,2:end)),1);wasActive=previous>(1:K);
% B.4 conditions on the allocation entering this iteration. Refresh ALL slabs.
v=ones(1,L);
for ell=1:K
 v(ell)=betarnd(1+sum(previous==ell),alpha+sum(previous>ell));attempts=0;
 while ~(v(ell)>0&&v(ell)<1)
  attempts=attempts+1;assert(attempts<100,'CSP:B45Stick');
  v(ell)=betarnd(1+sum(previous==ell),alpha+sum(previous>ell));
 end
end
logomega=log(v)+[0,cumsum(log1p(-v(1:K)))];
shape=1+p*double(wasActive);rate=5+mass.*double(wasActive);precision=zeros(1,K);
for h=1:K
 precision(h)=gamrnd(shape(h),1/rate(h));attempts=0;
 while ~(isfinite(precision(h))&&precision(h)>0&&isfinite(1/precision(h)))
  attempts=attempts+1;assert(attempts<100,'CSP:B45Scale');
  precision(h)=gamrnd(shape(h),1/rate(h));
 end
end
draw=struct('logomega',logomega,'logslab',-log(precision),'precision',precision);
if isempty(state.average)||rho==1
 state.average=draw;
else
 for key=["logomega","logslab","precision"]
  assert(isequal(size(state.average.(key)),size(draw.(key))),'CSP:B45Statistics');
  state.average.(key)=(1-rho)*state.average.(key)+rho*draw.(key);
 end
end
% B.5 evaluates BOTH branches using the same accumulated objective.
a=state.average;scores=zeros(K,L);categories=zeros(1,K);
spikeScore=-p*log(2*spike)-mass/spike;
slabScore=-p*log(2)-p*a.logslab-mass.*a.precision;
for h=1:K
 scores(h,:)=a.logomega+[spikeScore(h)*ones(1,h),slabScore(h)*ones(1,L-h)];
 [~,categories(h)]=max(scores(h,:));
end
active=categories>(1:K);penalty=ones(1,K)/spike;penalty(active)=a.precision(active);
weights=[zeros(p,1),repmat(penalty,p,1)];
state.categories=categories;state.v=v;state.omega=exp(logomega);state.slab=1./precision;
state.draw=draw;state.scores=scores;state.conditioning_categories=previous;
state.slab_shape=shape;state.slab_rate=rate;state.rho=rho;
end
