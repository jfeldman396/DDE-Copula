function [value,gradient]=foa_objective(a,B)
X=[ones(size(a.config,1),1) a.config];eta=X*B';
value=sum(a.T.*eta-a.N.*dde_softplus(eta),'all');
gradient=(a.T-a.N.*dde_logistic(eta))'*X;
end
