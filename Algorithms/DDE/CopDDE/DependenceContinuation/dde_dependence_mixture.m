function [q,ind,m]=dde_dependence_mixture(p,bits,lambda)
% Direct categorical mixture of EXACT marginals and EXACT joint conditional.
% At lambda<1 this is intentionally not Gibbs for the original target.
assert(isscalar(lambda) && lambda>=0 && lambda<=1);
m=min(1,max(0,p*bits)); ind=ones(size(p));
for k=1:size(bits,2)
    ind=ind.*(m(:,k)*bits(:,k)'+(1-m(:,k))*(1-bits(:,k))');
end
ind=ind./sum(ind,2);
if lambda==1, q=p; elseif lambda==0, q=ind; else, q=(1-lambda)*ind+lambda*p; end
end
