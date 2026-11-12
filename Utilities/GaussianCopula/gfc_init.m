function [R,Z,L,eta]=gfc_init(Y,H,seed)
% Exact weak-rank initialization: distinct observed categories have disjoint intervals.
rng(seed,'twister');[N,J]=size(Y);R=nan(N,J);Z=zeros(N,J);
for j=1:J
 ok=~isnan(Y(:,j));[~,~,r]=unique(Y(ok,j),'sorted');R(ok,j)=r;
 counts=accumarray(r,1);cuts=[0;cumsum(counts)/sum(ok)];
 u=cuts(r)+(cuts(r+1)-cuts(r)).*rand(sum(ok),1);
 Z(ok,j)=norminv(u);Z(~ok,j)=randn(sum(~ok),1);
end
[~,S,V]=svd(Z-mean(Z,1),'econ');ev=diag(S).^2/N;
L=V(:,1:H).*sqrt(max(ev(1:H)'-1,.02));
eta=Z*L/(eye(H)+L'*L);
end
