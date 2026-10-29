function f=lsa_fit(input,arm,path,target,fullAverage,forceOne)
if nargin<5,fullAverage=~strcmp(arm.scheme,'H0');end;if nargin<6,forceOne=false;end
c=lsa_protocol;if nargin<4,target=arm.cap;end
if ~isempty(path)&&isfile(path),s=load(path,'record');f=s.record;else
 f=input.initial;K1=size(f.A1,2);K2=size(f.A2,2);f.B1=min(max(f.B1,-10),10);f.B2=min(max(f.B2,[-5 -10*ones(1,K2)]),10);f.prop=min(max(f.prop,1e-6),1-1e-6);f.state1=[];f.state2=[];f.active1=true(1,K1);f.active2=true(1,K2);f.accum=[];f.iteration=0;f.stable_checks=0;f.stabilized=false;f.history={};f.trace={};f.stoplog={};f.CPU=0;f.wall=0;f.warm_CPU=0;f.warm_wall=0;f.arm=arm;f.seed=input.algorithm_seed;f.J=input.J;f.n=input.n;f.replication=input.replication;
end
if ~isfield(f,"objective_accum"),f.objective_accum=[];end
if f.stabilized,return;end
N=input.n;J=input.J;cm=struct('sample_count',1,'tolerance',1e-6,'max_iterations',1000);R=input.R;
for it=f.iteration+1:min(target,arm.cap)
 timer=tic;cpu=cputime;beta=1;if arm.tempering,beta=cms_beta(it,true);end;lambda=1;if arm.CP,lambda=min(1,(it-1)/59);end
 rng(mod(input.algorithm_seed+10007*it+1,2^32-1),'twister');tc=tic;[f.A1,f.A2]=cmv_binary(f.A1,f.A2,f.Z,f.B1,f.B2,f.gamma,f.prop,input.blocks,1,beta,[lambda lambda],'CP',struct('vi_layers',[false false]));binary=toc(tc);
 rng(mod(input.algorithm_seed+10007*it+2,2^32-1),'twister');tc=tic;X=[ones(N,1) f.A1];
 if strcmp(arm.scheme,'Z'),f.Z=dde_rank_gibbs(f.Z,R,f.A1,f.B1,f.gamma/beta);else,mu=X*f.B1';for j=1:J,f.Z(:,j)=cma_column(f.Z(:,j),R(:,j),mu(:,j),f.gamma(j)/beta,2,'sequential');end;end
 gaussian=toc(tc);S=struct('G',X'*X,'H',X'*f.Z,'ZZ',sum(f.Z.^2,1)');rho=1;
 if ~strcmp(arm.scheme,'H0')&&it>=arm.SA_start
  k=it-arm.SA_start+1;if k>10 && ~forceOne,rho=(k-9)^(-.7);end
  if isempty(f.accum),f.accum=S;else,for key=["G" "H" "ZZ"],f.accum.(key)=(1-rho)*f.accum.(key)+rho*S.(key);end;end;S=f.accum;
 end
 rng(mod(input.algorithm_seed+10007*it+3,2^32-1),'twister');tc=tic;[f.theta2,f.state2,f.active2]=pcs_csp_update(f.B2,.1,f.state2,sum(f.active1),5);[f.theta,f.state1,f.active1]=pcs_csp_update(f.B1,.05,f.state1,J,15);w1=[zeros(J,1),1./f.theta];w2=[zeros(size(f.B2,1),1),1./f.theta2];
 [f.B1,s1]=sob_cm(X,f.Z,f.B1,w1,f.gamma/beta,S);rss=S.ZZ-2*sum(f.B1.*S.H',2)+sum((f.B1*S.G).*f.B1,2);assert(all(rss>=-1e-7*(1+S.ZZ)),'PFS:NegativeRSS');f.gamma=(1+.5*beta*max(rss,0))/(2+.5*beta*N);
 if fullAverage && ~strcmp(arm.scheme,'H0') && it>=arm.SA_start
  ta=tic;f.objective_accum=foa_accumulate(f.objective_accum,f.A1,f.A2,rho);accumSeconds=toc(ta);
  if rho==1
   [f.B2,s2]=cpd_weighted_l1_cm('logistic',[ones(N,1) f.A2],f.A1,f.B2,w2/beta,ones(size(f.B2,1),1),cm);
  else
   a2=f.objective_accum;op=cm;op.observation_weights=a2.N;
   [f.B2,s2]=foa_cm('logistic',[ones(size(a2.config,1),1) a2.config],a2.T./a2.N,f.B2,w2/beta,ones(size(f.B2,1),1),op);
  end
  f.prop=min(max(f.objective_accum.top/N,1e-6),1-1e-6);
 else
  accumSeconds=0;[f.B2,s2]=cpd_weighted_l1_cm('logistic',[ones(N,1) f.A2],f.A1,f.B2,w2/beta,ones(size(f.B2,1),1),cm);f.prop=min(max(mean(f.A2,1),1e-6),1-1e-6);
 end
 tau=max(.3,3*N^(-.3));b=f.B1(:,2:end);b(abs(b)<tau)=0;f.B1(:,2:end)=b;b=f.B2(:,2:end);b(abs(b)<tau)=0;f.B2(:,2:end)=b;f.B2(~f.active1,:)=0;parameter=toc(tc);
 dt=cputime-cpu;dw=toc(timer);f.CPU=f.CPU+dt;f.wall=f.wall+dw;if it<=arm.warmup,f.warm_CPU=f.warm_CPU+dt;f.warm_wall=f.warm_wall+dw;end
 p1=f.B1;p2=f.B2;p1(:,find(~f.active1)+1)=0;p2(:,find(~f.active2)+1)=0;
 f.history{it}=struct('B1',p1,'B2',p2,'gamma',f.gamma,'prop',f.prop,'active1',f.active1,'active2',f.active2);
 [stable,stop]=lsa_stop(f.history,it,arm,c);if stop.eligible,if stable,f.stable_checks=f.stable_checks+1;else,f.stable_checks=0;end;end
 mem=whos('f');aMem=0;nc=0;if ~isempty(f.objective_accum),a2=f.objective_accum;wm=whos('a2');aMem=wm.bytes;nc=size(a2.config,1);end
 f.trace{it}=struct('accumulator_bytes',aMem,'state_bytes',mem.bytes,'configurations',nc,'accumulation_seconds',accumSeconds,'beta',beta,'lambda',lambda,'rho',rho,'K1',sum(f.active1),'K2',sum(f.active2),'CPU',f.CPU,'wall',f.wall,'binary_seconds',binary,'gaussian_seconds',gaussian,'parameter_seconds',parameter,'flags',sum(~s1.converged)+sum(~s2.converged),'KKT',max([s1.scaled_kkt_residual;s2.scaled_kkt_residual]));f.stoplog{it}=stop;f.iteration=it;f.rng=rng;
 f.stabilized=f.stable_checks>=c.required_checks;
 if ~isempty(path)&&(mod(it,c.checkpoint_every)==0||it==target||f.stabilized),cf_save(path,f);end
 if f.stabilized,break;end
end
end
