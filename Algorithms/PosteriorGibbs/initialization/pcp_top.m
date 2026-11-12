function [next,stats]=pcp_top(A1,old,B2,prop,beta,lambda)
% Whole-layer CP: old-state coordinate product versus exact JOINT Gibbs.
% Batch observations for memory only; never partition the top-layer state.
assert(lambda>=0 && lambda<=1 && beta>0);
N=size(old,1); K=size(old,2); prop=reshape(prop,1,[]);
assert(numel(prop)==K && all(prop>=0 & prop<=1));
useProduct=rand(N,1)<1-lambda; next=old;
if any(useProduct)
    next(useProduct,:)=dde_top_product_proposal(A1(useProduct,:),old(useProduct,:),B2,prop,beta);
end
rows=find(~useProduct);
stats=struct('product_rows',useProduct,'lambda',lambda,'joint_width',K, ...
    'enumerated_width',0,'observation_batches',0);
if isempty(rows), return; end
% Zero loading columns factorize EXACTLY, but remain stochastic coordinates.
% Boundary-prior coordinates are deterministic and enter the intercept.
fixed=prop==0 | prop==1; variable=~fixed;
next(rows,fixed)=repmat(prop(fixed),numel(rows),1);
W=B2(:,2:end); offset=B2(:,1)+W(:,fixed)*prop(fixed)';
coupled=variable & any(W~=0,1); independent=variable & ~coupled;
if any(independent)
    p=dde_logistic(beta*(log(prop(independent))-log1p(-prop(independent))));
    next(rows,independent)=double(rand(numel(rows),sum(independent))<p);
end
H=sum(coupled); stats.enumerated_width=H;
if H==0, return; end
assert(H<=20,'PaperCP:JointWidth','Joint enumeration exceeds configured width.');
bits=double(dec2bin(0:2^H-1,H)-'0');
varying=any(W(:,coupled)~=0,2);
eta=offset(varying)'+bits*W(varying,coupled)';
prior=bits*log(prop(coupled))'+(1-bits)*log1p(-prop(coupled))';
constant=prior-sum(dde_softplus(eta),2);
batch=max(1,floor(2e6/size(bits,1)));
for first=1:batch:numel(rows)
    ii=rows(first:min(first+batch-1,numel(rows)));
    lw=beta*(A1(ii,varying)*eta'+constant');
    p=exp(lw-max(lw,[],2)); p=p./sum(p,2);
    cdf=cumsum(p,2); cdf(:,end)=1;
    chosen=1+sum((1-rand(numel(ii),1))>cdf,2);
    next(ii,coupled)=bits(chosen,:);
    stats.observation_batches=stats.observation_batches+1;
end
end
