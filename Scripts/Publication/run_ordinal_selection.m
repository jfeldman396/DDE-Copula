function run_ordinal_selection(out,indices)
% Recreate the original bounded search and architecture-based reselection.
% Fresh synthetic data only; held-out rows never enter hyperparameter ranking.
% OUT may be reused only to resume this same search; completed stages are immutable.
if nargin<2,indices=1:4;end
assert(java.io.File(out).isAbsolute()&&all(ismember(indices,1:4)));
repo=fileparts(fileparts(fileparts(mfilename('fullpath'))));
oldpath=path;oldrng=rng;oldroot=getenv('DDE_ORDINAL_SELECTION_ROOT');
cleanup=onCleanup(@()restore_selection(oldpath,oldrng,oldroot));
addpath(fullfile(repo,'Algorithms','DDE'),fullfile(repo,'Algorithms','EmpiricalDDE'));verify_revision_sources;cpdde_paths(repo);
addpath(fullfile(repo,'Algorithms','DDE','CopDDE','JointSamplerAcceleration'),'-begin');
addpath(fullfile(repo,'Algorithms','OrdinalDDE'));
setenv('DDE_ORDINAL_SELECTION_ROOT',out);
base=fullfile(out,'results','reviewer1_tail_tuning');
if ~isfolder(base),mkdir(base);end
if ~isfolder(fullfile(base,'validity_selection')),mkdir(fullfile(base,'validity_selection'));end
prior=fullfile(out,'results','reviewer1_tail_study');if ~isfolder(prior),mkdir(prior);end
families={'clayton','survival_clayton','student_t','gaussian'};
for fi=indices
 file=fullfile(prior,sprintf('data_%s_n1000_r1.mat',families{fi}));
 if ~isfile(file),data=r1t_data(families{fi},1000,30,1);save(file,'data','-v7.3');end
end
run_r1tt_resume('study',indices);
run_r1tt_valid_selection(indices);
plan=jsondecode(fileread(fullfile(repo,'Scripts','Publication','config','ordinal.json')));
summary=struct([]);
for fi=indices
 family=families{fi};s=load(fullfile(base,'validity_selection',[family '_D2_valid_final.mat']),'final');f=s.final;
 expected=plan.cases(fi).hp;names=fieldnames(expected);
 for k=1:numel(names),a=f.hp.(names{k});b=expected.(names{k});assert(isequal(a(:),b(:)),'Ordinal:SelectionMismatch','Fresh selection differs from reported settings');end
 rejected3=0;eligible2=0;
 for c=1:10
  a=load(fullfile(base,sprintf('%s_D3_c%02d_stage240.mat',family,c)),'record');v=r1tt_valid_architecture(a.record.f,3);rejected3=rejected3+~v.eligible;
  a=load(fullfile(base,sprintf('%s_D2_c%02d_stage240.mat',family,c)),'record');v=r1tt_valid_architecture(a.record.f,2);eligible2=eligible2+v.eligible;
 end
 row=struct('family',family,'candidate',f.candidate,'hp',f.hp,'validation_pmse',f.validation_pmse,'test_pmse',f.test_score.pmse,'eligible_D2_screen',eligible2,'ineligible_D3_screen',rejected3,'matches_reported_settings',true,'test_used_for_selection',false);
 if isempty(summary),summary=row;else,summary(end+1)=row;end
end
fid=fopen(fullfile(out,'SELECTION_CHECK.json'),'w');assert(fid>=0);fprintf(fid,'%s\n',jsonencode(summary,PrettyPrint=true));fclose(fid);
end
function root=r1tt_setup
root=getenv('DDE_ORDINAL_SELECTION_ROOT');assert(~isempty(root));
end
function restore_selection(p,r,e),path(p);rng(r);setenv('DDE_ORDINAL_SELECTION_ROOT',e);end

function prep=r1tt_prepare(family,fi)
root=r1tt_setup;out=fullfile(root,'results','reviewer1_tail_tuning');file=fullfile(out,[family '_prepared.mat']);
if isfile(file),load(file,'prep');return;end
prior=load(fullfile(root,'results','reviewer1_tail_study',sprintf('data_%s_n1000_r1.mat',family)),'data');Y=prior.data.Ytrain;assert(isequal(size(Y),[1000 30]));clear prior;
splitseed=518200+fi;rng(splitseed,'twister');order=randperm(1000);fit_rows=order(1:700);validation_rows=order(701:1000);Yfit=Y(fit_rows,:);Yvalidation=Y(validation_rows,:);seed=831000+100*fi;
initial=cell(1,3);for D=2:3,widths=[9 3 1];initial{D}=r1t_initialize_original(Yfit,widths(1:D),seed);end
prep=struct('family',family,'family_index',fi,'split_seed',splitseed,'fit_rows',fit_rows,'validation_rows',validation_rows,'Ytrain_full',Y,'Yfit',Yfit,'Yvalidation',Yvalidation,'initial',{initial},'initial_seed',seed,'algorithm_seed',seed+33,'score_seed',971000+100*fi,'known_U_used',false,'test_used',false);
save(file,'prep','-v7.3');
end

function run_r1tt_resume(mode,indices)
if nargin<1,mode='study';end;if nargin<2,indices=1:4;end
root=r1tt_setup;dest=fullfile(root,'results','reviewer1_tail_tuning');families={'clayton','survival_clayton','student_t','gaussian'};
if strcmp(mode,'runtime'),indices=1;end
for fi=indices
 prep=r1tt_prepare(families{fi},fi);
 for D=2:3
  completed_final=fullfile(dest,sprintf('%s_D%d_final.mat',prep.family,D));completed_selection=fullfile(dest,sprintf('%s_D%d_selection.mat',prep.family,D));if isfile(completed_final)&&isfile(completed_selection),fprintf('IMMUTABLE_COMPLETED_SKIP %s D%d\n',prep.family,D);continue;end
  widths=[9 3 1];alpha=widths(1:D);hps={};for tau=[.1 .4],for t1=[.02 .08],for t2=[.02 .08],hps{end+1}=struct('tau',tau,'t1',t1,'t2',t2,'beta0',.7,'alpha',alpha);end;end;end
  if strcmp(mode,'runtime')
   cp=fullfile(dest,sprintf('%s_D%d_c01_checkpoint.mat',prep.family,D));[f,elapsed]=r1tt_fit_case(prep.initial{D},hps{1},prep.algorithm_seed,cp,60);timer=tic;score=r1tt_score(f,prep.Yfit,prep.Yvalidation,prep.score_seed+10*D,3,5,100);scoring_elapsed=toc(timer);runtime=struct('D',D,'iterations',f.iteration,'fit_seconds',elapsed,'score_seconds',scoring_elapsed,'hp',hps{1},'active',cellfun(@sum,f.active));save(fullfile(dest,sprintf('runtime_D%d.mat',D)),'runtime');disp(runtime);continue;
  end
  records=cell(1,10);for c=1:8,records{c}=candidate(prep,D,c,hps{c},240);end
  scores=cellfun(@(r)r.score.pmse,records(1:8));[~,order]=sort(scores);
  assert(isfinite(scores(order(2))),'Tuning:NoCandidates','Need at least two complete candidates');
  refinement_parents=order(1:2);for c=9:10,hp=hps{refinement_parents(c-8)};hp.beta0=.9;hps{c}=hp;records{c}=candidate(prep,D,c,hp,240);end
  scores=cellfun(@(r)r.score.pmse,records);[~,order]=sort(scores);finalists=order(1:2);assert(all(isfinite(scores(finalists))));
  finalscore=zeros(1,2);
  for k=1:2
   c=finalists(k);records{c}=candidate(prep,D,c,hps{c},600);r=records{c};r.finalist_score=r1tt_score(r.f,prep.Yfit,prep.Yvalidation,prep.score_seed+10*D,5,5,100);records{c}=r;record=r;save(fullfile(dest,sprintf('%s_D%d_c%02d_result.mat',prep.family,D,c)),'record','-v7.3');save(fullfile(dest,sprintf('%s_D%d_c%02d_finalist_rerank.mat',prep.family,D,c)),'record','-v7.3');finalscore(k)=r.finalist_score.pmse;
  end
  [~,winner]=min(finalscore);chosen=finalists(winner);selection=struct('family',prep.family,'D',D,'candidate',chosen,'hp',hps{chosen},'refinement_parents',refinement_parents,'finalists',finalists,'finalist_pmse',finalscore,'fit_rows',prep.fit_rows,'validation_rows',prep.validation_rows,'selected_without_test',true,'records',{records});
  selectfile=fullfile(dest,sprintf('%s_D%d_selection.mat',prep.family,D));save(selectfile,'selection','-v7.3');fprintf('SELECT %s D%d c%d pMSE=%.6g HP=%s\n',prep.family,D,chosen,min(finalscore),jsonencode(hps{chosen}));
  fullfileout=fullfile(dest,sprintf('%s_D%d_final.mat',prep.family,D));if isfile(fullfileout),fprintf('EXISTS final %s D%d\n',prep.family,D);continue;end
  initial=r1t_initialize_original(prep.Ytrain_full,alpha,prep.initial_seed);checkpoint=fullfile(dest,sprintf('%s_D%d_full_checkpoint.mat',prep.family,D));[f,fit_elapsed]=r1tt_fit_case(initial,hps{chosen},prep.algorithm_seed,checkpoint,600);
  % First access to reserved test arrays in this tuning workflow, after selection.
  prior=load(fullfile(root,'results','reviewer1_tail_study',sprintf('data_%s_n1000_r1.mat',prep.family)),'data');data=prior.data;selected_model=r1tune_selected_model(f);pred=r1t_predict(selected_model,prep.Ytrain_full,5000,prep.score_seed+77000);test_score=r1tt_score(f,prep.Ytrain_full,data.Ytest,prep.score_seed+88000+10*D,5,5,100);
  final=struct('family',prep.family,'D',D,'hp',hps{chosen},'candidate',chosen,'validation_pmse',min(finalscore),'f',f,'selected_model',selected_model,'pred',pred,'test_score',test_score,'fit_elapsed',fit_elapsed,'data',data,'selected_without_test',true);save(fullfileout,'final','-v7.3');fprintf('FINAL %s D%d it=%d testpMSE=%.6g\n',prep.family,D,f.iteration,test_score.pmse);
 end
end
fprintf('TAIL_TUNING_COMPLETE mode=%s\n',mode);
end
function record=candidate(prep,D,c,hp,target)
root=r1tt_setup;dest=fullfile(root,'results','reviewer1_tail_tuning');stem=sprintf('%s_D%d_c%02d',prep.family,D,c);file=fullfile(dest,[stem '_result.mat']);checkpoint=fullfile(dest,[stem '_checkpoint.mat']);
stage240=fullfile(dest,[stem '_stage240.mat']);if target==240&&isfile(stage240),s=load(stage240,'record');record=s.record;assert(isequaln(record.hp,hp));return;end
if isfile(file),s=load(file,'record');record=s.record;assert(isequaln(record.hp,hp));if record.f.iteration>=target||record.f.stabilized,return;end;end
fprintf('CANDIDATE %s target=%d HP=%s\n',stem,target,jsonencode(hp));timer=tic;
try
 [f,fit_elapsed]=r1tt_fit_case(prep.initial{D},hp,prep.algorithm_seed,checkpoint,target);score=r1tt_score(f,prep.Yfit,prep.Yvalidation,prep.score_seed+10*D,3,5,100);record=struct('family',prep.family,'D',D,'candidate',c,'hp',hp,'f',f,'score',score,'target',target,'fit_elapsed',fit_elapsed,'total_elapsed',toc(timer),'status','complete','test_used',false);save(file,'record','-v7.3');save(fullfile(dest,sprintf('%s_stage%d.mat',stem,target)),'record','-v7.3');fprintf('CANDIDATE_DONE %s it=%d pMSE=%.6g\n',stem,f.iteration,score.pmse);
catch ME
 failure=getReport(ME,'extended','hyperlinks','off');status='failed';save(fullfile(dest,sprintf('%s_failure_target%d.mat',stem,target)),'failure','hp','status','target');fprintf('CANDIDATE_FAILED %s\n%s\n',stem,failure);record=struct('family',prep.family,'D',D,'candidate',c,'hp',hp,'score',struct('pmse',Inf),'status','failed','test_used',false);
end
end

function run_r1tt_valid_selection(indices)
% Versioned endpoint-validity reselection; no changes to fitting transitions.
if nargin<1,indices=1:4;end
root=r1tt_setup;base=fullfile(root,'results','reviewer1_tail_tuning');out=fullfile(base,'validity_selection');families={'clayton','survival_clayton','student_t','gaussian'};
for fi=indices
 family=families{fi};prep=r1tt_prepare(family,fi);D=2;validfile=fullfile(out,[family '_D2_valid_final.mat']);if isfile(validfile),fprintf('VALID_IMMUTABLE_SKIP %s\n',family);continue;end
 pool={};
 for c=1:10
  file=fullfile(base,sprintf('%s_D2_c%02d_stage240.mat',family,c));load(file,'record');a=architecture(record.f,2);entry=struct('candidate',c,'record',record,'architecture',a,'eligible',a.eligible,'score',record.score.pmse);pool{end+1}=entry;
 end
 eligible=find(cellfun(@(r)r.eligible,pool));[~,ord]=sort(cellfun(@(r)r.score,pool(eligible)));queue=eligible(ord);initial_pool=pool;save(fullfile(out,[family '_D2_initial_pool.mat']),'initial_pool','queue','-v7.3');fprintf('VALID_POOL %s eligible=%s\n',family,mat2str(queue));
 promoted={};attempted=[];attempts={};next=1;done=false;
 % Promote the best two eligible screen records first. More are considered
 % only if those do not produce an eligible full-data refit.
 while next<=min(2,numel(queue)),promoted{end+1}=promote(prep,pool{queue(next)}.record);next=next+1;end
 while ~done
  available=find(cellfun(@(r)r.architecture.eligible,promoted));if ~isempty(available),available=available(~ismember(cellfun(@(r)r.candidate,promoted(available)),attempted));end
  if isempty(available)
   if next<=numel(queue),promoted{end+1}=promote(prep,pool{queue(next)}.record);next=next+1;continue;else,break;end
  end
  [~,ix]=sort(cellfun(@(r)r.finalist_score.pmse,promoted(available)));available=available(ix);
  for i=available
   r=promoted{i};attempted(end+1)=r.candidate;[full,arch,reused]=fullfit(prep,r);attempts{end+1}=struct('candidate',r.candidate,'hp',r.hp,'validation_pmse',r.finalist_score.pmse,'architecture',arch,'reused_raw_final',reused);
   if ~arch.eligible,fprintf('VALID_FULL_REJECT %s c%d widths=%s\n',family,r.candidate,mat2str(arch.widths));continue;end
   selection=struct('family',family,'D',2,'candidate',r.candidate,'hp',r.hp,'validation_score',r.finalist_score,'architecture_training',r.architecture,'architecture_full',arch,'pool_candidates',queue,'attempted_candidates',attempted,'selected_without_test',true,'protocol','validity-only v2 endpoint reselection');save(fullfile(out,[family '_D2_valid_selection.mat']),'selection','-v7.3');
   if ~reused
    prior=load(fullfile(root,'results','reviewer1_tail_study',sprintf('data_%s_n1000_r1.mat',family)),'data');data=prior.data;f=full.f;selected_model=r1tune_selected_model(f);pred=r1t_predict(selected_model,prep.Ytrain_full,5000,prep.score_seed+77000);test_score=r1tt_score(f,prep.Ytrain_full,data.Ytest,prep.score_seed+88020,5,5,100);final=struct('family',family,'D',2,'candidate',r.candidate,'hp',r.hp,'validation_pmse',r.finalist_score.pmse,'f',f,'selected_model',selected_model,'pred',pred,'test_score',test_score,'fit_elapsed',full.fit_elapsed,'data',data,'selected_without_test',true);
   else,final=full;final.validation_pmse=r.finalist_score.pmse;end
   final.validity=arch;final.validity_selection=selection;save(validfile,'final','-v7.3');fprintf('VALID_FINAL %s c%d widths=%s pMSE=%.6g reused=%d\n',family,r.candidate,mat2str(arch.widths),final.test_score.pmse,reused);done=true;break;
  end
 end
 status='NO_ELIGIBLE_FULL_REFIT';if done,status='VALID_FULL_REFIT_AVAILABLE';end
 save(fullfile(out,[family '_D2_validity_status.mat']),'status','pool','promoted','attempts','queue','attempted','-v7.3');fprintf('VALID_STATUS %s %s\n',family,status);
end
fprintf('VALIDITY_SELECTION_COMPLETE\n');
end
function r=promote(prep,screen)
root=r1tt_setup;base=fullfile(root,'results','reviewer1_tail_tuning');out=fullfile(base,'validity_selection');family=prep.family;c=screen.candidate;dest=fullfile(out,sprintf('%s_D2_c%02d_promotion.mat',family,c));if isfile(dest),load(dest,'r');return;end
old600=fullfile(base,sprintf('%s_D2_c%02d_stage600.mat',family,c));
if isfile(old600),load(old600,'record');r=record;elseif screen.f.stabilized,r=screen;else
 cp=fullfile(out,sprintf('%s_D2_c%02d_checkpoint.mat',family,c));if ~isfile(cp),copyfile(fullfile(base,sprintf('%s_D2_c%02d_checkpoint.mat',family,c)),cp);end
 [f,elapsed]=r1tt_fit_case(prep.initial{2},screen.hp,prep.algorithm_seed,cp,600);r=screen;r.f=f;r.target=600;r.fit_elapsed=elapsed;
end
r.architecture=architecture(r.f,2);
if r.architecture.eligible
 oldscore=fullfile(base,sprintf('%s_D2_c%02d_finalist_rerank.mat',family,c));
 if isfile(oldscore),ss=load(oldscore,'record');assert(isequal(ss.record.f.B,r.f.B)&&isequal(ss.record.f.gamma,r.f.gamma)&&isequal(ss.record.f.active,r.f.active));r.finalist_score=ss.record.finalist_score;
 else,r.finalist_score=r1tt_score(r.f,prep.Yfit,prep.Yvalidation,prep.score_seed+20,5,5,100);end
 fprintf('VALID_PROMOTE %s c%d it=%d widths=%s score=%.6g\n',family,c,r.f.iteration,mat2str(r.architecture.widths),r.finalist_score.pmse);
else,r.finalist_score=struct('pmse',Inf,'synthetic_sd',NaN,'synthetic_se',NaN);fprintf('VALID_PROMOTE_REJECT %s c%d widths=%s\n',family,c,mat2str(r.architecture.widths));end
save(dest,'r','-v7.3');
end
function [full,arch,reused]=fullfit(prep,r)
root=r1tt_setup;base=fullfile(root,'results','reviewer1_tail_tuning');out=fullfile(base,'validity_selection');family=prep.family;c=r.candidate;dest=fullfile(out,sprintf('%s_D2_c%02d_fullfit.mat',family,c));reused=false;
if isfile(dest),load(dest,'full','arch','reused');return;end
old=load(fullfile(base,[family '_D2_final.mat']),'final');
if isequaln(old.final.hp,r.hp),full=old.final;reused=true;arch=architecture(full.f,2);save(dest,'full','arch','reused','-v7.3');return;end
initial=r1t_initialize_original(prep.Ytrain_full,[9 3],prep.initial_seed);cp=fullfile(out,sprintf('%s_D2_c%02d_full_checkpoint.mat',family,c));[f,fit_elapsed]=r1tt_fit_case(initial,r.hp,prep.algorithm_seed,cp,600);arch=architecture(f,2);full=struct('family',family,'D',2,'candidate',c,'hp',r.hp,'f',f,'fit_elapsed',fit_elapsed,'status','RAW_FULL_REFIT_AUDIT');save(dest,'full','arch','reused','-v7.3');
end
function a=architecture(f,D)
[eligible,d]=r1_architecture(f,D);a=r1tt_valid_architecture(f,D);assert(eligible==a.eligible&&isequal(d.widths,a.widths)&&isequal(d.path_widths,a.observed_path_widths));a.shared_reason=d.reason;
end

function [f,elapsed]=r1tt_fit_case(initial,hp,seed,checkpoint,target)
input=struct('initial',initial,'n',size(initial.Z,1),'J',size(initial.Z,2),'R',initial.R,'replication',1,'algorithm_seed',seed,'blocks',[3 3]);D=numel(initial.B);timer=tic;
if D==2
 input.initial=struct('A1',initial.A{1},'A2',initial.A{2},'B1',initial.B{1},'B2',initial.B{2},'Z',initial.Z,'gamma',initial.gamma,'prop',initial.prop);
 arm=struct('scheme','Z','cap',600,'tempering',true,'CP',true,'warmup',60,'SA_start',81);
 cfg=struct('tau',hp.tau,'spike1',hp.t1,'spike2',hp.t2,'beta0',hp.beta0,'alpha1',hp.alpha(1),'alpha2',hp.alpha(2));raw=r1tune_fit(input,arm,checkpoint,target,true,false,true,cfg);
 f=struct('A',{{raw.A1,raw.A2}},'B',{{raw.B1,raw.B2}},'gamma',raw.gamma,'prop',raw.prop,'Z',raw.Z,'active',{{raw.active1,raw.active2}},'iteration',raw.iteration,'CPU',raw.CPU,'wall',raw.wall,'stabilized',raw.stabilized,'trace',{raw.trace},'stable_checks',raw.stable_checks,'hp',hp);
else,f=r1tt_fit_deep(input,hp,checkpoint,target);end
elapsed=toc(timer);dde_assert_rank_consistency(f.Z,input.R);
end

function f=r1tt_fit_deep(input,hp,file,target)
% Isolated configurable arbitrary-depth extension, preserving exact-block phase.
n=input.n;J=input.J;D=numel(input.initial.A);blocks=min(input.initial.widths,3);seed=input.algorithm_seed;
if isfile(file),load(file,'f');assert(isequaln(f.hp,hp),'Tuning:CheckpointHP','Hyperparameter mismatch');else
 f=input.initial;f.hp=hp;f.iteration=0;f.states=cell(1,D);f.active=cellfun(@(a)true(1,size(a,2)),f.A,'UniformOutput',false);f.accum=[];f.obj=cell(1,D);f.trace={};f.history={};f.CPU=0;f.wall=0;f.stable_checks=0;f.stabilized=false;
 for d=1:D,lo=-10*ones(1,size(f.B{d},2));if d>1,lo(1)=-5;end;f.B{d}=min(max(f.B{d},lo),10);end
end
if f.stabilized,return;end
cm=struct('sample_count',1,'tolerance',1e-6,'max_iterations',1000);
for it=f.iteration+1:target
 timer=tic;cpu=cputime;beta=r1tune_beta(it,hp.beta0);lambda=min(1,(it-1)/59);
 rng(mod(seed+10007*it+1,2^32-1),'twister');f.A=r1t_binary(f.A,f.Z,f.B,f.gamma,f.prop,blocks,beta,lambda,false,'CAVI');
 rng(mod(seed+10007*it+2,2^32-1),'twister');X=[ones(n,1),f.A{1}];f.Z=dde_rank_gibbs(f.Z,input.R,f.A{1},f.B{1},f.gamma/beta);
 S=struct('G',X'*X,'H',X'*f.Z,'ZZ',sum(f.Z.^2,1)');rho=1;
 if it>=81
  k=it-80;if k>10,rho=(k-9)^(-.7);end
  if isempty(f.accum),f.accum=S;else,for key=["G" "H" "ZZ"],f.accum.(key)=(1-rho)*f.accum.(key)+rho*S.(key);end;end;S=f.accum;
 end
 rng(mod(seed+10007*it+3,2^32-1),'twister');weights=cell(1,D);
 for d=D:-1:1
  if d==1,spike=hp.t1;children=J;else,spike=hp.t2;children=sum(f.active{d-1});end
  [theta,f.states{d},f.active{d}]=pcs_csp_update(f.B{d},spike,f.states{d},children,hp.alpha(d));weights{d}=[zeros(size(f.B{d},1),1),1./theta];
 end
 [f.B{1},s1]=sob_cm(X,f.Z,f.B{1},weights{1},f.gamma/beta,S);rss=S.ZZ-2*sum(f.B{1}.*S.H',2)+sum((f.B{1}*S.G).*f.B{1},2);assert(all(rss>=-1e-7*(1+S.ZZ)));f.gamma=(1+.5*beta*max(rss,0))/(2+.5*beta*n);
 flags=sum(~s1.converged);maxKKT=max(s1.scaled_kkt_residual);rejections=sum(s1.rejected);maxCMiterations=max(s1.iterations);
 for d=2:D
  if it>=81
   f.obj{d}=foa_accumulate(f.obj{d},f.A{d-1},f.A{d},rho);
   if rho==1,[f.B{d},ss]=cpd_weighted_l1_cm('logistic',[ones(n,1),f.A{d}],f.A{d-1},f.B{d},weights{d}/beta,ones(size(f.B{d},1),1),cm);
   else,aa=f.obj{d};op=cm;op.observation_weights=aa.N;[f.B{d},ss]=foa_cm('logistic',[ones(size(aa.config,1),1),aa.config],aa.T./aa.N,f.B{d},weights{d}/beta,ones(size(f.B{d},1),1),op);end
  else,[f.B{d},ss]=cpd_weighted_l1_cm('logistic',[ones(n,1),f.A{d}],f.A{d-1},f.B{d},weights{d}/beta,ones(size(f.B{d},1),1),cm);end
  flags=flags+sum(~ss.converged);maxKKT=max(maxKKT,max(ss.scaled_kkt_residual));rejections=rejections+sum(ss.rejected);maxCMiterations=max(maxCMiterations,max(ss.iterations));
 end
 if it>=81,f.prop=min(max(f.obj{D}.top/n,1e-6),1-1e-6);else,f.prop=min(max(mean(f.A{D}),1e-6),1-1e-6);end
 for d=1:D,b=f.B{d}(:,2:end);b(abs(b)<hp.tau)=0;f.B{d}(:,2:end)=b;if d>1,f.B{d}(~f.active{d-1},:)=0;end;end
 f.CPU=f.CPU+cputime-cpu;f.wall=f.wall+toc(timer);f.iteration=it;
 effective=f.B;for d=1:D,effective{d}(:,find(~f.active{d})+1)=0;end
 f.history{it}=struct('B',{effective},'gamma',f.gamma,'prop',f.prop,'active',{f.active});
 [stable,eligible,drift]=deepstop(f.history,it);if eligible,if stable,f.stable_checks=f.stable_checks+1;else,f.stable_checks=0;end;end;f.stabilized=f.stable_checks>=3;
 f.trace{it}=struct('iteration',it,'beta',beta,'lambda',lambda,'rho',rho,'flags',flags,'KKT',maxKKT,'rejections',rejections,'maxCMiterations',maxCMiterations,'active',cellfun(@sum,f.active),'nonzero',cellfun(@(b)sum(any(b(:,2:end)~=0,1)),f.B),'drift',drift,'CPU',f.CPU,'wall',f.wall);
 if mod(it,20)==0||it==target||f.stabilized,save(file,'f','-v7.3');fprintf('TUNE_DEEP it=%d CPU=%.1f K=%s flags=%d KKT=%.3g\n',it,f.CPU,mat2str(cellfun(@sum,f.active)),flags,maxKKT);end
 if f.stabilized,break;end
end
end
function [stable,eligible,drift]=deepstop(history,it)
D=numel(history{it}.B);drift=nan(1,D+2);stable=false;eligible=it>=100&&mod(it-60,5)==0;if ~eligible,return;end
h=history(it-19:it);activeStable=all(cellfun(@(x)isequal(x.active,h{1}.active),h));
for d=1:D,x=cellfun(@(v)v.B{d},h,'UniformOutput',false);old=mean(cat(3,x{1:10}),3);new=mean(cat(3,x{11:20}),3);drift(d)=norm(new-old,'fro')/max(norm(old,'fro'),1e-8);end
for index=1:2,keys={'gamma','prop'};key=keys{index};x=cellfun(@(v)v.(key),h,'UniformOutput',false);old=mean(cat(3,x{1:10}),3);new=mean(cat(3,x{11:20}),3);drift(D+index)=norm(new-old,'fro')/max(norm(old,'fro'),1e-8);end
stable=activeStable&&all(drift(1:D)<=.005)&&drift(D+1)<=.01&&drift(D+2)<=.005;
end

function a=r1tt_valid_architecture(f,requestedD)
model=r1tune_selected_model(f);D=numel(model.B);w=cellfun(@(b)sum(any(b(:,2:end)~=0,1)),model.B);reach=any(model.B{1}(:,2:end)~=0,1);paths=sum(reach);for d=2:D,reach=any(model.B{d}(reach,2:end)~=0,1);paths(d)=sum(reach);end
prev=[size(model.B{1},1),w(1:end-1)];positive=all(w>0)&&D==requestedD;bound=prev>=3*w;no_dangling=isequal(w,paths);a=struct('eligible',positive&&all(bound)&&no_dangling,'requested_depth',requestedD,'widths',w,'observed_path_widths',paths,'CSP',cellfun(@sum,f.active),'raw_nonzero',cellfun(@(b)sum(any(b(:,2:end)~=0,1)),f.B),'positive_requested_layers',positive,'width_bounds',bound,'no_dangling',no_dangling,'scope','Necessary width/nonzero-path checks only; sufficient graph-identifiability conditions not asserted');
end

function [eligible,detail]=r1_architecture(raw,requestedDepth)
% Necessary architecture checks on the actual CSP-pruned predictive model.
% These checks do not certify the sufficient pure-children graph conditions.
model=r1tune_selected_model(raw);
if isfield(model,'B') && iscell(model.B),B=model.B;else,B={model.B1,model.B2};end
D=numel(B);if nargin<2,requestedDepth=D;end
J=size(B{1},1);widths=cellfun(@(b)sum(any(b(:,2:end)~=0,1)),B);
reach=any(B{1}(:,2:end)~=0,1);paths=zeros(1,D);paths(1)=sum(reach);
for d=2:D
 assert(size(B{d},1)==numel(reach),'Architecture:Shape','Adjacent matrix shapes do not match.');
 reach=any(B{d}(reach,2:end)~=0,1);paths(d)=sum(reach);
end
previous=[J widths(1:end-1)];bounds=previous>=3*widths;
positive=all(widths>0);connected=isequal(paths,widths);depthMatches=D==requestedDepth;
eligible=positive && all(bounds) && connected && depthMatches;
reasons={};if ~positive,reasons{end+1}='zero requested layer';end
if ~all(bounds),reasons{end+1}='adjacent width bound failed';end
if ~connected,reasons{end+1}='upper node has no nonzero path to observations';end
if ~depthMatches,reasons{end+1}='requested depth mismatch';end
detail=struct('eligible',eligible,'widths',widths,'path_widths',paths,'J',J,...
 'requested_depth',requestedDepth,'positive_layers',positive,'width_bounds',bounds,...
 'no_dangling',connected,'reason',strjoin(reasons,'; '),...
 'scope','Necessary positive-width, adjacent 3-child width bounds and observed-path checks only; not a pure-children theorem certificate.');
end

function [A,diagnostic]=r1t_binary(A,Z,B,gamma,prop,blocks,beta,lambda,record,method)
% Arbitrary depth, exact enumerated BLOCK conditional at beta=lambda=1.
% Every layer is visited top-to-bottom; own state-independent row permutation.
if nargin<9,record=false;end;if nargin<10,method='CP';end
D=numel(A);n=size(Z,1);diagnostic=struct('blocks',{{}},'beta',beta,'lambda',lambda);
for d=D:-1:1
 K=size(A{d},2);[~,order]=sort(rand(n,K),2);
 if d>1,childEta=[ones(n,1),A{d}]*B{d}';else,childEta=[ones(n,1),A{1}]*B{1}';end
 if d<D,parentEta=[ones(n,1),A{d+1}]*B{d+1}';else,parentEta=repmat(log(prop./(1-prop)),n,1);end
 for first=1:blocks(d):K
  if record,beforeA=A;end
  S=order(:,first:min(first+blocks(d)-1,K));m=size(S,2);bits=double(dec2bin(0:2^m-1,m)-'0');C=size(bits,1);old=A{d}((1:n)'+n*(S-1));eta0=childEta;
  for k=1:m,eta0=eta0-old(:,k).*B{d}(:,S(:,k)+1)';end
  logw=zeros(n,C);
  for c=1:C
   eta=eta0;
   for k=1:m,eta=eta+bits(c,k)*B{d}(:,S(:,k)+1)';end
   if d==1,logw(:,c)=-.5*sum((Z-eta).^2./gamma',2);else,logw(:,c)=sum(A{d-1}.*eta-dde_softplus(eta),2);end
   for k=1:m,h=parentEta((1:n)'+n*(S(:,k)-1));logw(:,c)=logw(:,c)+bits(c,k)*h;end
  end
  logw=beta*logw;P=exp(logw-max(logw,[],2));P=P./sum(P,2);draw=P;
  if lambda<1
   if strcmp(method,'CAVI'),[~,draw]=pilot_mix(logw,P,bits,old,lambda,'VI');else,[~,draw]=bl_mix(logw,P,bits,old,lambda,'CP');end
  end
  cum=cumsum(draw,2);cum(:,end)=1;chosen=1+sum((1-rand(n,1))>cum,2);next=bits(chosen,:);A{d}((1:n)'+n*(S-1))=next;
  for k=1:m,childEta=childEta+(next(:,k)-old(:,k)).*B{d}(:,S(:,k)+1)';end
  if record,diagnostic.blocks{end+1}=struct('layer',d,'S',S,'old',old,'bits',bits,'P',P,'logw',logw,'beforeA',{beforeA});end
 end
end
end
