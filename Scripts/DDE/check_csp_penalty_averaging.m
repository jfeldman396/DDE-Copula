function report=check_csp_penalty_averaging(out)
% Check B.4 conditionals, B.5 against the direct accumulated prior, and resume.
assert(java.io.File(out).isAbsolute()&&~isfolder(out)&&~isfile(out));
repo=fileparts(fileparts(fileparts(mfilename('fullpath'))));oldpath=path;oldrng=rng;cleanup=onCleanup(@()restore(oldpath,oldrng));
addpath(fullfile(repo,'Algorithms','DDE'));cpdde_paths(repo);verify_revision_sources;mkdir(out);
rng(719);B=[zeros(1,4);0 .1 .3 0;0 1.5 0 -.4;0 .2 .1 -.2];spike=.05;alpha=3;
s=[];draws={};weights=[];maximum=0;priorMaximum=0;
for t=1:25
 rho=1;if t>3,rho=(t-2)^(-.7);end
 if t==2,s.categories=[1 4 2];end % Force active/inactive full conditionals.
 if isempty(s),previous=4*ones(1,3);else,previous=s.categories;end
 [w,s,active]=csp_b45_update(B,spike,s,alpha,rho);
 assert(isequal(s.slab_shape,1+4*double(previous>(1:3))));
 assert(isequal(s.slab_rate,5+sum(abs(B(:,2:end)),1).*double(previous>(1:3))));
 assert(s.rows==4&&all(w(:,1)==0)); % zero row still contributes its prior factors
 draws{end+1}=s.draw;weights=[(1-rho)*weights,rho];
 for key=["logomega","logslab","precision"]
  manual=zeros(size(s.average.(key)));
  for j=1:numel(draws),manual=manual+weights(j)*draws{j}.(key);end
  discrepancy=max(abs(manual-s.average.(key)),[],'all');maximum=max(maximum,discrepancy);
 end
 % Independent enumeration: evaluate the complete Laplace log densities at
 % each past draw, then sum them with the actual SA weights (not B.5 code).
 for h=1:3
  expected=zeros(1,4);
  for ell=1:4
   for j=1:numel(draws)
    scale=spike;if ell>h,scale=exp(draws{j}.logslab(h));end
    expected(ell)=expected(ell)+weights(j)*(draws{j}.logomega(ell)+sum(-log(2*scale)-abs(B(:,h+1))/scale));
   end
  end
  priorMaximum=max(priorMaximum,max(abs(expected-s.scores(h,:))));
  assert(expected(s.categories(h))>=max(expected)-1e-11);
  assert(expected(s.categories(h))>=expected(previous(h))-1e-11);
  coefficient=1/spike;if active(h),coefficient=s.average.precision(h);end
  assert(all(w(:,h+1)==coefficient));
 end
end
assert(maximum<1e-11&&priorMaximum<1e-10);
% Inactive scales must be freshly drawn from IG(1,5), not retained stale.
precisions=zeros(1000,1);for t=1:1000,s.categories=[1 2 3];[~,s]=csp_b45_update(B,spike,s,alpha,1);precisions(t)=1/s.slab(1);assert(all(s.slab_shape==1)&&all(s.slab_rate==5));end
assert(numel(unique(precisions))==1000&&abs(mean(precisions)-.2)<.03);
[w,empty,active]=csp_b45_update(zeros(3,1),spike,[],alpha,1);assert(isequal(w,zeros(3,1))&&isempty(active)&&empty.v==1);
% End-to-end missing data, unequal block partitions, and exact split resume.
rng(1234);N=30;J=18;K1=6;K2=2;
initial=struct('A1',double(rand(N,K1)<.5),'A2',double(rand(N,K2)<.5),'B1',randn(J,K1+1)*.2,'B2',randn(K1,K2+1)*.2,'Z',randn(N,J),'gamma',ones(J,1),'prop',[.4 .6]);
R=ceil(4*normcdf(initial.Z));R(1:3:end,1)=NaN;
input=struct('initial',initial,'R',R,'n',N,'J',J,'algorithm_seed',10291,'replication',1,'blocks',[4 2],'layer_partitions',{{[2 4],[2]}});
arm=pfs_arm(1);cfg=r1tune_config(struct('tau',.1,'spike1',.02,'spike2',.04,'beta0',.9,'alpha1',6,'alpha2',2));
instant=cfg;instant.csp_penalty_averaging=false;
a=full_cavi_fit(input,arm,'',90,true,false,true,cfg);b=full_cavi_fit(input,arm,'',90,true,false,true,instant);compare_states(a,b);
whole=full_cavi_fit(input,arm,'',95,true,false,true,cfg);
checkpoint=fullfile(out,'resume.mat');record=a;save(checkpoint,'record','-v7.3');
for it=91:95
 previous=a;a=full_cavi_fit(input,arm,checkpoint,it,true,false,true,cfg);rho=(it-90+1)^(-.7);
 assert(a.trace{it}.rho==rho&&a.trace{it}.csp_penalties_averaged);
 for layer=1:2
  key=sprintf('state%d',layer);s=a.(key);old=previous.(key);
  for stat=["logomega","logslab","precision"]
   assert(isequal(s.average.(stat),(1-rho)*old.average.(stat)+rho*s.draw.(stat)));
  end
  sizes=[J K1];assert(s.rows==sizes(layer));
 end
end
compare_states(whole,a);
% Old checkpoints are invalid even BEFORE SAEM, since B.4-B.5 changes warm-up.
record=rmfield(a,'csp_update_version');bad=fullfile(out,'legacy.mat');save(bad,'record','-v7.3');
expect_error(@()full_cavi_fit(input,arm,bad,96,true,false,true,cfg),'CAVI:CSPVersionMismatch');
record.iteration=10;save(bad,'record','-v7.3');expect_error(@()full_cavi_fit(input,arm,bad,96,true,false,true,cfg),'CAVI:CSPVersionMismatch');
record=a;record.state1.average=[];save(bad,'record','-v7.3');expect_error(@()full_cavi_fit(input,arm,bad,96,true,false,true,cfg),'CAVI:MissingCSPHistory');
expect_error(@()full_cavi_fit(input,arm,checkpoint,96,true,false,true,instant),'CAVI:CSPModeMismatch');
report=struct('passed',true,'csp_update_version','B4_B5_v1','B4_active_and_inactive_conditionals',true,'all_coefficient_prior_rows_counted',true,'B5_matches_weighted_complete_prior',true,'allocation_CM_nondecreasing',true,'both_layers_checked',true,'rho1_same_as_current_draws',true,'empty_top_layer',true,'missing_responses',true,'unequal_block_partitions',true,'resume_exact',true,'legacy_checkpoints_rejected',true,'max_statistic_absolute_error',maximum,'max_prior_absolute_error',priorMaximum);
f=fopen(fullfile(out,'REPORT.json'),'w');assert(f>0);fprintf(f,'%s\n',jsonencode(report,PrettyPrint=true));fclose(f);disp(report);
end
function compare_states(a,b)
keys={'A1','A2','Z','B1','B2','gamma','prop','theta','theta2','active1','active2','state1','state2','accum','objective_accum','csp_penalty_weights','history','iteration','stabilized','stable_checks','rng'};
for i=1:numel(keys),key=keys{i};assert(isequaln(a.(key),b.(key)),['State mismatch: ' key]);end
end
function expect_error(action,identifier)
try,action();catch err,assert(strcmp(err.identifier,identifier),'Unexpected error: %s',err.identifier);return;end
error('CAVI:ExpectedError','Expected %s',identifier);
end
function restore(p,r),path(p);rng(r);end
