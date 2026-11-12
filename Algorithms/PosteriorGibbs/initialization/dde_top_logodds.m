function odds=dde_top_logodds(A1,A2,B2,prop)
%DDE_TOP_LOGODDS Exact ordinary top-coordinate log odds, all at the same state.
N=size(A1,1); K=size(A2,2); eta=[ones(N,1),A2]*B2'; odds=zeros(N,K);
for j=1:K
    loading=B2(:,j+1)'; e0=eta-A2(:,j).*loading;
    odds(:,j)=log(prop(j))-log1p(-prop(j))+A1*loading' ...
        -sum(dde_softplus(e0+loading)-dde_softplus(e0),2);
end
end
