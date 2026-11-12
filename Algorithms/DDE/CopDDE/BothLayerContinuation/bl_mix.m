function [q,drawProbability]=bl_mix(logw,exact,bits,old,lambda,method)
% Local block law; CP mixes kernels conditional on the same old block state.
if lambda==1, q=exact; if nargout>1, drawProbability=exact; end; return; end
if ismember(method,{'C1','C2'})
    q=dde_dependence_mixture(exact,bits,lambda);
    if nargout>1, drawProbability=q; end
    return;
end
assert(strcmp(method,'CP'));
[N,m]=size(old); powers=2.^(m-1:-1:0); code=old*powers'; logits=zeros(N,m);
for j=1:m
    zero=code-old(:,j)*powers(j); one=zero+powers(j);
    logits(:,j)=logw((1:N)'+N*one)-logw((1:N)'+N*zero);
end
lp=-dde_softplus(-logits)*bits'-dde_softplus(logits)*(1-bits)';
product=exp(lp-max(lp,[],2)); product=product./sum(product,2);
q=(1-lambda)*product+lambda*exact;
if nargout>1
    useProduct=rand(N,1)<1-lambda; drawProbability=exact;
    drawProbability(useProduct,:)=product(useProduct,:);
end
end
