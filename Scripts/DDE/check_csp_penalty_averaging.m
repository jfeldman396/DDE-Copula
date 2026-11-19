function report=check_csp_penalty_averaging(out)
% Small deterministic check of both penalties, missing data and checkpoint resume.
% Example: addpath('Scripts/DDE'); check_csp_penalty_averaging('/new/absolute/check');
assert(java.io.File(out).isAbsolute()&&~isfolder(out)&&~isfile(out));
repo=fileparts(fileparts(fileparts(mfilename('fullpath'))));oldpath=path;oldrng=rng;cleanup=onCleanup(@()restore(oldpath,oldrng));
addpath(fullfile(repo,'Algorithms','DDE'));cpdde_paths(repo);verify_revision_sources;mkdir(out);
rng(1234);N=30;J=18;K1=6;K2=2;
initial=struct('A1',double(rand(N,K1)<.5),'A2',double(rand(N,K2)<.5),'B1',randn(J,K1+1)*.2,'B2',randn(K1,K2+1)*.2,'Z',randn(N,J),'gamma',ones(J,1),'prop',[.4 .6]);
R=ceil(4*normcdf(initial.Z));R(1:3:end,1)=NaN;
input=struct('initial',initial,'R',R,'n',N,'J',J,'algorithm_seed',10291,'replication',1,'blocks',[4 2],'layer_partitions',{{[2 4],[2]}});
arm=pfs_arm(1);cfg=r1tune_config(struct('tau',.1,'spike1',.02,'spike2',.04,'beta0',.9,'alpha1',6,'alpha2',2));
assert(cfg.csp_penalty_averaging);legacy=cfg;legacy.csp_penalty_averaging=false;
a=full_cavi_fit(input,arm,'',90,true,false,true,cfg);
b=full_cavi_fit(input,arm,'',90,true,false,true,legacy);
compare_states(a,b,false);assert(isempty(b.csp_penalty_accum));
assert(isequal(a.csp_penalty_accum.w1,[zeros(J,1),1./a.theta]));
assert(isequal(a.csp_penalty_accum.w2,[zeros(K1,1),1./a.theta2]));
% An uninterrupted fit must equal a fit resumed through individual SAEM steps.
whole=full_cavi_fit(input,arm,'',95,true,false,true,cfg);
checkpoint=fullfile(out,'resume.mat');record=a;save(checkpoint,'record','-v7.3');
w1=a.csp_penalty_accum.w1;w2=a.csp_penalty_accum.w2;weights=1;maximum=0;
for it=91:95
 a=full_cavi_fit(input,arm,checkpoint,it,true,false,true,cfg);rho=(it-90+1)^(-.7);
 assert(a.trace{it}.rho==rho&&a.trace{it}.csp_penalties_averaged);
 w1=cat(3,w1,[zeros(J,1),1./a.theta]);w2=cat(3,w2,[zeros(K1,1),1./a.theta2]);
 weights=[(1-rho)*weights,rho];assert(abs(sum(weights)-1)<1e-12);
 expected1=sum(w1.*reshape(weights,1,1,[]),3);expected2=sum(w2.*reshape(weights,1,1,[]),3);
 error1=norm(a.csp_penalty_accum.w1-expected1,'fro')/max(norm(expected1,'fro'),eps);
 error2=norm(a.csp_penalty_accum.w2-expected2,'fro')/max(norm(expected2,'fro'),eps);maximum=max([maximum,error1,error2]);assert(maximum<1e-12);
 assert(all(a.csp_penalty_accum.w1(:,1)==0)&&all(a.csp_penalty_accum.w2(:,1)==0));
end
compare_states(whole,a,true);
% Old post-SAEM states and changed modes must fail, not restart accumulation.
record=rmfield(a,{'csp_penalty_accum','csp_penalty_averaging'});bad=fullfile(out,'old_post_saem.mat');save(bad,'record','-v7.3');
expect_error(@()full_cavi_fit(input,arm,bad,96,true,false,true,cfg),'CAVI:MissingCSPHistory');
record=a;record.csp_penalty_accum=[];bad=fullfile(out,'empty_accumulator.mat');save(bad,'record','-v7.3');
expect_error(@()full_cavi_fit(input,arm,bad,96,true,false,true,cfg),'CAVI:MissingCSPHistory');
expect_error(@()full_cavi_fit(input,arm,checkpoint,96,true,false,true,legacy),'CAVI:CSPModeMismatch');
% Before SAEM, a checkpoint without the new fields is safe to upgrade.
record=full_cavi_fit(input,arm,'',80,true,false,true,cfg);record=rmfield(record,{'csp_penalty_accum','csp_penalty_averaging'});
before=fullfile(out,'before_saem.mat');save(before,'record','-v7.3');upgraded=full_cavi_fit(input,arm,before,95,true,false,true,cfg);compare_states(whole,upgraded,true);
report=struct('passed',true,'both_layers_checked',true,'rho1_unchanged',true,'missing_responses',true,'unequal_block_partitions',true,'resume_exact',true,'unsafe_checkpoints_rejected',true,'max_weighted_sum_relative_error',maximum);
f=fopen(fullfile(out,'REPORT.json'),'w');assert(f>0);fprintf(f,'%s\n',jsonencode(report,PrettyPrint=true));fclose(f);disp(report);
end
function compare_states(a,b,includePenalty)
keys={'A1','A2','Z','B1','B2','gamma','prop','theta','theta2','active1','active2','state1','state2','accum','objective_accum','history','iteration','stabilized','stable_checks','rng'};
if includePenalty,keys{end+1}='csp_penalty_accum';end
for i=1:numel(keys),key=keys{i};assert(isequaln(a.(key),b.(key)),['State mismatch: ' key]);end
end
function expect_error(action,identifier)
try,action();catch err,assert(strcmp(err.identifier,identifier),'Unexpected error: %s',err.identifier);return;end
error('CAVI:ExpectedError','Expected %s',identifier);
end
function restore(p,r),path(p);rng(r);end
