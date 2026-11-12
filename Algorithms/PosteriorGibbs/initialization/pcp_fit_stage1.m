function fit=pcp_fit_stage1(initial,R,tau,method,options,seed,truth,resume)
%FIT_RBG_STAGE1 Matched corrected RL-CSP engine, binary kernel is sole factor.
% Shared changes vs legacy: correct Gaussian temperature, robust ranks,
% converged convex CM, beta-consistent gamma, sample-mean pi, fixed budget.
if nargin<8, resume=[]; end
timer=tic; fit=initial; fit.gamma=fit.gamma(:); fit.prop=reshape(fit.prop,1,[]);
N=size(R,1); J=size(R,2); K1=size(fit.A1,2); K2=size(fit.A2,2);
if isempty(resume)
fit.B1=min(max(fit.B1,-10),10); fit.B2=min(max(fit.B2,[-5 -10*ones(1,K2)]),10);
fit.prop=min(max(fit.prop,options.probability_floor),1-options.probability_floor);
end
dde_assert_rank_consistency(fit.Z,R);
fit.initial=struct('B1',initial.B1,'B2',initial.B2,'gamma',initial.gamma,'prop',initial.prop);
fit.options=options; fit.method=method; fit.seed=seed;
block=4;
state1=[]; state2=[]; active1=true(1,K1); active2=true(1,K2); beta=options.inverse_temperature;
keys={'K1','K2','beta','binary_seconds','seconds_per_sweep','iteration_seconds', ...
    'switching1','switching2','multi_fraction','relative_B1','relative_B2', ...
    'optimizer_flagged_rows','max_scaled_kkt','q_before','q_cm_after','q_gated_after'};
for key=keys, fit.diagnostics.(key{1})=zeros(options.iterations,1); end
fit.phase_seeds=zeros(options.iterations,3);
cm=struct('sample_count',1,'tolerance',options.loading_tolerance,'max_iterations',options.max_loading_iterations);
fit.trace=cell(options.iterations,1); fit.dependence=cell(options.iterations,1);
firstIteration=1;
if ~isempty(resume)
    state1=resume.state.state1; state2=resume.state.state2; active1=resume.state.active1; active2=resume.state.active2;
    firstIteration=resume.iteration+1; beta=resume.beta_next;
    fit.trace(1:resume.iteration)=resume.trace; fit.dependence(1:resume.iteration)=resume.dependence;
    fit.phase_seeds(1:resume.iteration,:)=resume.phase_seeds;
    for name=fieldnames(resume.saved_states)', fit.(name{1})=resume.saved_states.(name{1}); end
    for name=keys, fit.diagnostics.(name{1})(1:resume.iteration)=resume.diagnostics.(name{1}); end
end
for it=firstIteration:options.iterations
    iterationClock=tic; old1=fit.B1; old2=fit.B2;
    phase=mod(seed+10007*it+[1 2 3],2^32-1); fit.phase_seeds(it,:)=phase;
    rng(phase(1),'twister');
    lambda=cf_lambda(method,it,options.warm_iterations);
    [fit.A1,fit.A2,stats]=pcp_binary(fit.A1,fit.A2,fit.Z,fit.B1,fit.B2,fit.gamma,fit.prop,block,options.sweeps,beta,lambda,method);
    rng(phase(2),'twister'); fit.Z=dde_rank_gibbs(fit.Z,R,fit.A1,fit.B1,fit.gamma/beta);
    rng(phase(3),'twister');
    [fit.theta2,state2,active2]=rbg_csp_update(fit.B2,options.spike2,state2,sum(active1));
    [fit.theta,state1,active1]=rbg_csp_update(fit.B1,options.spike1,state1,J);
    X1=[ones(N,1),fit.A1]; X2=[ones(N,1),fit.A2];
    w1=[zeros(J,1),1./fit.theta]; w2=[zeros(K1,1),1./fit.theta2];
    before=objective(fit,X1,X2,beta,w1,w2);
    [fit.B1,s1]=dde_weighted_l1_cm('gaussian',X1,fit.Z,fit.B1,w1,fit.gamma/beta,cm);
    residual=fit.Z-X1*fit.B1'; fit.gamma=((1+.5*beta*sum(residual.^2,1))/(2+.5*beta*N))';
    [fit.B2,s2]=dde_weighted_l1_cm('logistic',X2,fit.A1,fit.B2,w2/beta,ones(K1,1),cm);
    fit.prop=min(max(mean(fit.A2,1),options.probability_floor),1-options.probability_floor);
    after=objective(fit,X1,X2,beta,w1,w2);
    assert(after>=before-1e-8*(1+abs(before)),'RandomBlock:CMObjective');
    first=fit.B1(:,2:end); first(abs(first)<tau)=0; fit.B1(:,2:end)=first;
    second=fit.B2(:,2:end); second(abs(second)<tau)=0; fit.B2(:,2:end)=second;
    fit.B2(~active1,:)=0; % original hierarchical gate, before final pruning
    d=fit.diagnostics; d.K1(it)=sum(active1); d.K2(it)=sum(active2); d.beta(it)=beta;
    d.binary_seconds(it)=stats.runtime_seconds; d.seconds_per_sweep(it)=stats.seconds_per_sweep;
    d.switching1(it)=stats.switching1; d.switching2(it)=stats.switching2; d.multi_fraction(it)=stats.multi_fraction;
    d.relative_B1(it)=norm(fit.B1-old1,'fro')/max(norm(old1,'fro'),1e-8);
    d.relative_B2(it)=norm(fit.B2-old2,'fro')/max(norm(old2,'fro'),1e-8);
    d.optimizer_flagged_rows(it)=sum(~s1.converged)+sum(~s2.converged);
    d.max_scaled_kkt(it)=max([s1.scaled_kkt_residual;s2.scaled_kkt_residual]);
    d.q_before(it)=before; d.q_cm_after(it)=after; d.q_gated_after(it)=objective(fit,X1,X2,beta,w1,w2);
    d.iteration_seconds(it)=toc(iterationClock); fit.diagnostics=d;
    dde_assert_rank_consistency(fit.Z,R);
    fit.trace{it}=dc_trace(fit,it,beta,lambda,state1,state2,active1,active2,stats,truth);
    if it==30, fit.state30=dc_state(fit,state1,state2,active1,active2); end
    if it==50, fit.state50=dc_state(fit,state1,state2,active1,active2); end
    if it>5, beta=min(1,beta+.01*(it-6)); end
    if toc(timer)>=options.max_seconds, break; end
end
for key=keys, fit.diagnostics.(key{1})=fit.diagnostics.(key{1})(1:it); end
fit.phase_seeds=fit.phase_seeds(1:it,:); fit.iterations=it;
fit.B1(:,find(~active1)+1)=0; fit.B2(:,find(~active2)+1)=0;
fit.active1=active1; fit.active2=active2; fit.selected_dimensions=[sum(active1) sum(active2)];
fit.support1=fit.B1(:,2:end)~=0; fit.support2=fit.B2(:,2:end)~=0;
fit.csp_state1=state1; fit.csp_state2=state2; fit.segment_runtime_seconds=toc(timer); fit.runtime_seconds=fit.segment_runtime_seconds; if ~isempty(resume), fit.runtime_seconds=fit.runtime_seconds+resume.prefix_runtime_seconds; end; fit.rank_consistent=true;
fit.description='Matched hybrid CSP MAP/thresholded pretraining; exact-invariant binary kernels do not make the full procedure exact EM';
end

function value=objective(f,X1,X2,beta,w1,w2)
res=f.Z-X1*f.B1'; eta=X2*f.B2';
value=beta*(-.5*sum(log(2*pi*f.gamma')+res.^2./f.gamma','all')+sum(f.A1.*eta-dde_softplus(eta),'all') ...
    +sum(f.A2.*log(f.prop)+(1-f.A2).*log1p(-f.prop),'all')) ...
    -sum(abs(f.B1).*w1,'all')-sum(abs(f.B2).*w2,'all')-sum(2*log(f.gamma)+1./f.gamma);
end
