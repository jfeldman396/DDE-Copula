function fit=review_gfg_fit(Y,H,seed,path,opt)
% Isolated Gaussian-CSP fitter: existing gfc_fit conjugate/CSP body and exact rank Gibbs Z.
% opt identity is strict on resume, except an increasing iteration horizon.
assert(all(isfield(opt,{'iterations','burn','thin','tempering','spike','alpha','sweeps','kernel'})));
assert(strcmp(opt.kernel,'missing_safe_exact_rank_Gibbs') && opt.sweeps==1);
[N,J]=size(Y);R=nan(size(Y));for j=1:J,ok=~isnan(Y(:,j));[~,~,R(ok,j)]=unique(Y(ok,j),'sorted');end
if isfile(path)
 s=load(path,'fit');fit=s.fit;
 assert(isequaln(fit.R,R) && fit.seed==seed && size(fit.L,2)==H,'GFG:ResumeDataIdentity','Resume data, seed and capacity must match.');
 assert(isequaln(rmfield(fit.opt,'iterations'),rmfield(opt,'iterations')),'GFG:ResumeOptionIdentity','All non-horizon options must match on resume.');
 assert(opt.iterations>=fit.opt.iterations && opt.iterations>=fit.iteration,'GFG:ResumeHorizon','The iteration horizon may only increase.');
 fit.opt=opt;
else
 [Ri,Z,L,eta]=gfc_init(Y,H,seed);assert(isequaln(Ri,R));
 fit=struct('R',R,'Z',Z,'L',L,'eta',eta,'theta',ones(1,H),...
  'v',[ones(1,H-1)/(1+opt.alpha) 1],'c',H*ones(1,H),'iteration',0,'seed',seed,'opt',opt,...
  'saved_L',zeros(J,H,0),'saved_C',zeros(J,J,0),'saved_iteration',[],...
  'trace',zeros(0,9),'CPU',0,'wall',0);
end
for it=fit.iteration+1:opt.iterations
 timer=tic;cpu=cputime;rng(mod(seed+10007*it,2^32-1),'twister');beta=1;if opt.tempering,beta=gfc_beta(it);end
 phase=tic;P=eye(H)+beta*(fit.L'*fit.L);U=chol(P);
 fit.eta=(beta*fit.Z*fit.L)/P+randn(N,H)/U';factor_seconds=toc(phase);
 phase=tic;fit.Z=dde_rank_gibbs(fit.Z,R,fit.eta,[zeros(J,1) fit.L],ones(J,1)/beta);
 rank_Z_seconds=toc(phase);dde_assert_rank_consistency(fit.Z,R);
 phase=tic;P=diag(1./fit.theta)+beta*(fit.eta'*fit.eta);U=chol(P);
 fit.L=(beta*fit.Z'*fit.eta)/P+randn(J,H)/U';loading_seconds=toc(phase);
 phase=tic;[fit.theta,fit.v,fit.c]=gfc_csp(fit.L,fit.theta,fit.v,opt.alpha,2,2,opt.spike);csp_seconds=toc(phase);
 C=fit.L*fit.L'+eye(J);sd=sqrt(diag(C));C=C./(sd*sd');
 assert(all(isfinite([fit.Z(:);fit.L(:);fit.eta(:);fit.theta(:)]))&&all(fit.theta>0));
 if it>opt.burn && mod(it-opt.burn,opt.thin)==0
  fit.saved_L(:,:,end+1)=fit.L;fit.saved_C(:,:,end+1)=C;fit.saved_iteration(end+1)=it;
 end
 fit.CPU=fit.CPU+cputime-cpu;fit.wall=fit.wall+toc(timer);fit.iteration=it;
 fit.trace(it,:)=[it beta sum(fit.c>(1:H)) factor_seconds rank_Z_seconds loading_seconds csp_seconds fit.CPU fit.wall];
 if mod(it,50)==0||it==opt.iterations
  temp=[path '.tmp.mat'];save(temp,'fit','-v7.3');movefile(temp,path,'f');
  fprintf('GFG seed%d alpha%.0f spike%.4g iteration%d/%d slabs%d CPU%.2f wall%.2f rankZ%.4f\n',seed,opt.alpha,opt.spike,it,opt.iterations,fit.trace(it,3),fit.CPU,fit.wall,rank_Z_seconds);
 end
 if isfile(fullfile(fileparts(fileparts(path)),'STOP_REQUEST'))
  temp=[path '.tmp.mat'];save(temp,'fit','-v7.3');movefile(temp,path,'f');
  error('GFG:StudyHeld','Coordinator hold; completed iteration saved before stopping.');
 end
end
end
