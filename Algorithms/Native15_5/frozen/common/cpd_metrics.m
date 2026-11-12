function m=cpd_metrics(B1,B2,truth)
% Decomposition of the unchanged primary full-width metric.
assert(isequal(size(B1),size(truth.B1_scaled)) && isequal(size(B2),size(truth.B2)));
for layer=1:2
    if layer==1, B=B1; T=truth.B1_scaled; r=1:size(B,1); k=truth.K1_true;
    else, B=B2; T=truth.B2; r=1:truth.K1_true; k=truth.K2_true; end
    e=(B-T).^2; stem=sprintf('B%d_',layer);
    m.([stem 'full'])=mean(e,'all'); m.([stem 'loadings'])=mean(e(:,2:end),'all');
    m.([stem 'intercepts'])=mean(e(:,1));
    m.([stem 'true_block'])=mean(e(r,1:k+1),'all');
    m.([stem 'true_loadings'])=mean(e(r,2:k+1),'all');
    mask=false(size(e)); mask(r,1:k+1)=true;
    m.([stem 'redundant_SSE'])=sum(e(~mask));
    m.([stem 'redundant_contribution'])=sum(e(~mask))/numel(e);
    m.([stem 'true_contribution'])=sum(e(mask))/numel(e);
    m.([stem 'intercept_contribution'])=sum(e(:,1))/numel(e);
    m.([stem 'loading_contribution'])=sum(e(:,2:end),'all')/numel(e);
    assert(abs(m.([stem 'full'])-m.([stem 'true_contribution'])-m.([stem 'redundant_contribution']))<1e-10);
    assert(abs(m.([stem 'full'])-m.([stem 'intercept_contribution'])-m.([stem 'loading_contribution']))<1e-10);
end
end
