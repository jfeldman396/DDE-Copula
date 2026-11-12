function [theta,state,active]=pcs_csp_update(B,spike,state,children,alpha)
%RBG_CSP_UPDATE Legacy category MAP, stick Gibbs, sampled IG(1,5) slab scale.
% children follows original hierarchical gating convention for layer 2.
K=size(B,2)-1; L=K+1;
if isempty(state), state=struct('omega',ones(1,L)/L,'slab',ones(1,K)); end
categories=zeros(1,K); mass=sum(abs(B(:,2:end)),1);
for h=1:K
    ls=-children*log(2*spike)-mass(h)/spike;
    la=log(5)-children*log(2)+gammaln(1+children)-(1+children)*log(5+mass(h));
    score=log(state.omega)+[ls*ones(1,h),la*ones(1,L-h)];
    [~,categories(h)]=max(score);
end
v=zeros(1,L); v(end)=1;
for l=1:K, v(l)=betarnd(1+sum(categories==l),alpha+sum(categories>l)); end
state.omega=v.*[1 cumprod(1-v(1:end-1))];
active=categories>1:K;
for h=find(active), state.slab(h)=1/gamrnd(1+children,1/(5+mass(h))); end
scale=spike*ones(1,K); scale(active)=state.slab(active);
theta=repmat(scale,size(B,1),1); state.categories=categories; state.v=v;
end
