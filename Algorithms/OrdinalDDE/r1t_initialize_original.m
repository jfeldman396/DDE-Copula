function f=r1t_initialize_original(Y,widths,seed)
% Separate initialization sensitivity: original published two-layer initializer.
% D=3 adds a NEW recursive logistic spectral guard; not production equivalence.
rng(seed,'twister');[n,J]=size(Y);Z=zeros(n,J);R=zeros(n,J);
for j=1:J
 [~,~,g]=unique(Y(:,j));R(:,j)=g;cnt=accumarray(g,1);low=[0;cumsum(cnt(1:end-1))];Z(:,j)=norminv((low(g)+rand(n,1).*cnt(g))/n);
end
[prop,B1,B2,gamma,A1,A2]=Normal_init_v2(Z,widths(1),widths(1),widths(2),1e-4);A={A1,A2};B={B1,B2};
if numel(widths)==3
 % Extend the same double-SVD/logit construction to A2, without test data.
 [U,S,V]=svd(A2,'econ');mm=min(size(S,1),max(widths(3)+1,sum(diag(S)>1.01*sqrt(n))));trunc=U(:,1:mm)*S(1:mm,1:mm)*V(:,1:mm)';trunc=min(max(trunc,1e-4),1-1e-4);eta=log(trunc./(1-trunc));eta=eta-mean(eta);[~,~,V]=svd(eta,'econ');L=V(:,1:widths(3));L=L.*(2*(sum(L)>0)-1);A3=double(eta*L>0);X=[ones(n,1),A3];B3=((X'*X+1e-4*eye(size(X,2)))\(X'*eta))';
 % Explicit extension-only diagonal start guard, fixed before fitting.
 for k=1:widths(3),B3(k,k+1)=max(B3(k,k+1),2.5);end
 A{3}=A3;B{3}=B3;prop=mean(A3);
end
for d=1:numel(B),B{d}=min(max(B{d},-5),5);end
f=struct('A',{A},'B',{B},'gamma',max(gamma(:),.01),'prop',min(max(prop(:)',.05),.95),'Z',Z,'R',R,'widths',widths);
end
