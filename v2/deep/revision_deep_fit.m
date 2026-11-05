function f=revision_deep_fit(input,hp,file,target)
% Isolated configurable arbitrary-depth extension, preserving exact-block phase.
n=input.n;J=input.J;D=numel(input.initial.A);blocks=min(input.initial.widths,3);seed=input.algorithm_seed;
if isfile(file),load(file,'f');assert(isequaln(f.hp,hp),'Tuning:CheckpointHP','Hyperparameter mismatch');else
 f=input.initial;f.hp=hp;f.iteration=0;f.states=cell(1,D);f.active=cellfun(@(a)true(1,size(a,2)),f.A,'UniformOutput',false);f.accum=[];f.obj=cell(1,D);f.trace={};f.history={};f.CPU=0;f.wall=0;f.stable_checks=0;f.stabilized=false;
 for d=1:D,lo=-10*ones(1,size(f.B{d},2));if d>1,lo(1)=-5;end;f.B{d}=min(max(f.B{d},lo),10);end
end
if f.stabilized,return;end
warmup=30;if hp.cavi,warmup=60;end;SA_start=warmup+21;
cm=struct('sample_count',1,'tolerance',1e-6,'max_iterations',1000);
for it=f.iteration+1:target
 timer=tic;cpu=cputime;beta=r1tune_beta(it,hp.beta0);lambda=1;if hp.cavi,lambda=min(1,(it-1)/59);end
 rng(mod(seed+10007*it+1,2^32-1),'twister');f.A=r1t_binary(f.A,f.Z,f.B,f.gamma,f.prop,blocks,beta,lambda,false,'CAVI');
 rng(mod(seed+10007*it+2,2^32-1),'twister');X=[ones(n,1),f.A{1}];mu=X*f.B{1}';for j=1:J,f.Z(:,j)=cma_column(f.Z(:,j),input.R(:,j),mu(:,j),f.gamma(j)/beta,2,'sequential');end
 S=struct('G',X'*X,'H',X'*f.Z,'ZZ',sum(f.Z.^2,1)');rho=1;
 if it>=SA_start
  k=it-SA_start+1;if k>10,rho=(k-9)^(-.7);end
  if isempty(f.accum),f.accum=S;else,for key=["G" "H" "ZZ"],f.accum.(key)=(1-rho)*f.accum.(key)+rho*S.(key);end;end;S=f.accum;
 end
 rng(mod(seed+10007*it+3,2^32-1),'twister');weights=cell(1,D);
 for d=D:-1:1
  spike=hp.spikes(d);if d==1,children=J;else,children=sum(f.active{d-1});end
  [theta,f.states{d},f.active{d}]=pcs_csp_update(f.B{d},spike,f.states{d},children,hp.alpha(d));weights{d}=[zeros(size(f.B{d},1),1),1./theta];
 end
 [f.B{1},s1]=sob_cm(X,f.Z,f.B{1},weights{1},f.gamma/beta,S);rss=S.ZZ-2*sum(f.B{1}.*S.H',2)+sum((f.B{1}*S.G).*f.B{1},2);assert(all(rss>=-1e-7*(1+S.ZZ)));f.gamma=(1+.5*beta*max(rss,0))/(2+.5*beta*n);
 flags=sum(~s1.converged);maxKKT=max(s1.scaled_kkt_residual);rejections=sum(s1.rejected);maxCMiterations=max(s1.iterations);
 for d=2:D
  if it>=SA_start
   f.obj{d}=foa_accumulate(f.obj{d},f.A{d-1},f.A{d},rho);
   if rho==1,[f.B{d},ss]=cpd_weighted_l1_cm('logistic',[ones(n,1),f.A{d}],f.A{d-1},f.B{d},weights{d}/beta,ones(size(f.B{d},1),1),cm);
   else,aa=f.obj{d};op=cm;op.observation_weights=aa.N;[f.B{d},ss]=foa_cm('logistic',[ones(size(aa.config,1),1),aa.config],aa.T./aa.N,f.B{d},weights{d}/beta,ones(size(f.B{d},1),1),op);end
  else,[f.B{d},ss]=cpd_weighted_l1_cm('logistic',[ones(n,1),f.A{d}],f.A{d-1},f.B{d},weights{d}/beta,ones(size(f.B{d},1),1),cm);end
  flags=flags+sum(~ss.converged);maxKKT=max(maxKKT,max(ss.scaled_kkt_residual));rejections=rejections+sum(ss.rejected);maxCMiterations=max(maxCMiterations,max(ss.iterations));
 end
 if it>=SA_start,f.prop=min(max(f.obj{D}.top/n,1e-6),1-1e-6);else,f.prop=min(max(mean(f.A{D}),1e-6),1-1e-6);end
 for d=1:D,b=f.B{d}(:,2:end);b(abs(b)<hp.tau)=0;f.B{d}(:,2:end)=b;if d>1,f.B{d}(~f.active{d-1},:)=0;end;end
 f.CPU=f.CPU+cputime-cpu;f.wall=f.wall+toc(timer);f.iteration=it;
 effective=f.B;for d=1:D,effective{d}(:,find(~f.active{d})+1)=0;end
 f.history{it}=struct('B',{effective},'gamma',f.gamma,'prop',f.prop,'active',{f.active});
 [stable,eligible,drift]=deepstop(f.history,it,warmup);if eligible,if stable,f.stable_checks=f.stable_checks+1;else,f.stable_checks=0;end;end;f.stabilized=f.stable_checks>=3;
 f.trace{it}=struct('iteration',it,'beta',beta,'lambda',lambda,'rho',rho,'flags',flags,'KKT',maxKKT,'rejections',rejections,'maxCMiterations',maxCMiterations,'active',cellfun(@sum,f.active),'nonzero',cellfun(@(b)sum(any(b(:,2:end)~=0,1)),f.B),'drift',drift,'CPU',f.CPU,'wall',f.wall);
 if mod(it,20)==0||it==target||f.stabilized,save(file,'f','-v7.3');fprintf('TUNE_DEEP it=%d CPU=%.1f K=%s flags=%d KKT=%.3g\n',it,f.CPU,mat2str(cellfun(@sum,f.active)),flags,maxKKT);end
 if f.stabilized,break;end
end
end
function [stable,eligible,drift]=deepstop(history,it,warmup)
D=numel(history{it}.B);drift=nan(1,D+2);stable=false;eligible=it>=warmup+40&&mod(it-warmup,5)==0;if ~eligible,return;end
h=history(it-19:it);activeStable=all(cellfun(@(x)isequal(x.active,h{1}.active),h));
for d=1:D,x=cellfun(@(v)v.B{d},h,'UniformOutput',false);old=mean(cat(3,x{1:10}),3);new=mean(cat(3,x{11:20}),3);drift(d)=norm(new-old,'fro')/max(norm(old,'fro'),1e-8);end
for index=1:2,keys={'gamma','prop'};key=keys{index};x=cellfun(@(v)v.(key),h,'UniformOutput',false);old=mean(cat(3,x{1:10}),3);new=mean(cat(3,x{11:20}),3);drift(D+index)=norm(new-old,'fro')/max(norm(old,'fro'),1e-8);end
stable=activeStable&&all(drift(1:D)<=.005)&&drift(D+1)<=.01&&drift(D+2)<=.005;
end
