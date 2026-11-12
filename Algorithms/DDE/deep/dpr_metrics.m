function m=dpr_metrics(f,input,truth)
% Sequential positive-loading assignment, original sign convention.
previous=1:108; m.K=reshape(f.num_act,1,[]); m.B_aligned=cell(1,3);
for d=1:3
    estimate=f.B{d}(previous,:); target=truth.B_padded{d};
    K=size(target,2)-1; costs=zeros(K,K);
    for k=1:K, costs(k,:)=-sum(estimate(target(:,k+1)>0,2:end),1); end
    [assignment,~]=munkres(costs); previous=assignment;
    estimate=[estimate(:,1),estimate(:,assignment+1)];
    if d==1
        mu=mean(f.Z,1)'; sd=std(f.Z,0,1)'; sd=max(sd,1e-12);
        estimate=[(estimate(:,1)-mu)./sd estimate(:,2:end)./sd];
        perrep=truth.B_padded{1}; model=perrep;
        perrep(:,1)=(perrep(:,1)-input.truth_marginal_mean)./input.truth_marginal_sd;
        perrep(:,2:end)=perrep(:,2:end)./input.truth_marginal_sd;
        model(:,1:19)=truth.B1_model_scaled;
        m.MSE_B1_model=mean((estimate-model).^2,'all'); target=perrep;
    end
    m.B_aligned{d}=estimate;
    m.MSE(d)=mean((estimate-target).^2,'all');
    m.graph_accuracy(d)=mean((estimate~=0)==(target~=0),'all');
    m.correct_dimension(d)=m.K(d)==truth.true_widths(d);
end
m.correct_all=all(m.correct_dimension);
m.metric_definition=['Full maximum-width matrices including intercepts; sequential positive-loading ', ...
    'assignment; empirical final-Z first-layer scaling; per-replication truth-Z ', ...
    'scaling plus exact model-moment B1 MSE as a separate diagnostic.'];
end
