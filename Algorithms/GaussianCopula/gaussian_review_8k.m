function gaussian_review_8k(preflight,data_input,out,prior4k,planfile)
here=fileparts(mfilename('fullpath'));addpath(here);addpath(fullfile(fileparts(fileparts(here)),'Utilities','GaussianCopula'));assert(isfile(data_input)&&isfile(planfile));if ~isfolder(out),mkdir(out);end
plan=jsondecode(fileread(planfile));d=load(data_input);Y=d.Y;X=d.X;assert(isequal(size(Y),[1500 98])&&sum(isnan(Y),'all')==223);assert(isequal(X,fillmissing(Y,'constant',3)));
opt=struct('iterations',8000,'burn',1000,'thin',5,'tempering',true,'spike',.05,'alpha',5,'sweeps',1,'kernel','missing_safe_exact_rank_Gibbs');
if preflight
 opt.iterations=3;opt.burn=2;opt.tempering=false;
 a=review_gfg_fit_8k(Y,32,2026110299,fullfile(out,'pilot_split.mat'),opt);opt.iterations=6;a=review_gfg_fit_8k(Y,32,2026110299,fullfile(out,'pilot_split.mat'),opt);b=review_gfg_fit_8k(Y,32,2026110299,fullfile(out,'pilot_direct.mat'),opt);
 assert(isequaln(a.L,b.L)&&isequaln(a.Z,b.Z)&&isequaln(a.c,b.c)&&isequaln(a.saved_L,b.saved_L));
 rng(8);L=randn(98,32)/5;[Q,~]=qr(randn(32));assert(norm(L*L'-(L*Q)*(L*Q)','fro')<1e-10);
 Z=randn(1500,32)*L'+randn(1500,98);syn=empirical(Z,X);assert(isequal(sort(syn),sort(X)));
 [p,~]=r1tune_pmse(X,syn,X,2026110298,5,5);assert(p>=0&&p<=.25);
 writejson(fullfile(out,'PREFLIGHT.json'),struct('passed',true,'split_resume_parity',true,'rotation_covariance_preserved',true,'exact_margins',true,'score_finite',true));return;
end
assert(isfile(fullfile(out,'PREFLIGHT.json')));rankings=struct([]);
for id=1:6
 folder=fullfile(out,sprintf('c%d',id));if ~isfolder(folder),mkdir(folder);end
 done=fullfile(folder,'COMPLETE.json');if isfile(done),s=jsondecode(fileread(done));if isempty(rankings),rankings=s;else,rankings(end+1)=s;end;continue;end
 cfg=plan.configs(id);opt.alpha=cfg.alpha;opt.spike=cfg.spike;
 fits=cell(1,2);for c=1:2
 writejson(fullfile(out,'STATUS.json'),struct('stage','FITTING','config',id,'chain',c,'final_selection',false));
 fits{c}=review_gfg_fit_8k(Y,32,2026110300+c,fullfile(folder,sprintf('chain%d.mat',c)),opt);
 end
 ids=find(fits{1}.saved_iteration>4000);fixed=load(fullfile(prior4k,sprintf('c%d/POINT_MODEL.mat',id)),'reference','Q');reference=fixed.reference;Q=fixed.Q;
 meanL=zeros(98,32);meanstd=zeros(98,32);count=0;maxerr=0;off=triu(true(98),1);corrs=zeros(numel(ids),2,nnz(off));
 for c=1:2
  ii=find(fits{c}.saved_iteration>4000);assert(numel(ii)==800);
  for k=1:numel(ii)
   raw=fits{c}.saved_L(:,:,ii(k));a=raw./sqrt(1+sum(raw.^2,2));[U,~,V]=svd(a'*reference,'econ');T=U*V'*Q;
   maxerr=max(maxerr,norm(raw*raw'-(raw*T)*(raw*T)','fro')/max(1,norm(raw*raw','fro')));
   meanL=meanL+raw*T;meanstd=meanstd+a*T;count=count+1;C=fits{c}.saved_C(:,:,ii(k));corrs(k,c,:)=C(off);
  end
 end
 meanL=meanL/count;meanstd=meanstd/count;assert(maxerr<1e-10);
 standardized_point=meanL./sqrt(1+sum(meanL.^2,2));diagnostics=gfc_mcmc_diag(corrs);writejson(fullfile(folder,'DIAGNOSTICS.json'),diagnostics);
 save(fullfile(folder,'POINT_MODEL.mat'),'meanL','meanstd','standardized_point','reference','Q','cfg','maxerr','-v7.3');writematrix(standardized_point,fullfile(folder,'STANDARDIZED_POINT_LOADINGS.csv'));
 scorepath=fullfile(folder,'PMSE_REPLICATES.csv');if isfile(scorepath),scores=readtable(scorepath);assert(isequal(scores.replicate,(1:height(scores))'));else,scores=table;end
 for rep=height(scores)+1:50
  assert(~isfile(fullfile(out,'STOP_REQUEST')),'ReviewGaussian:Stopped');
  rng(plan.generation_seeds(rep),'twister');Z=randn(1500,32)*meanL'+randn(1500,98);synthetic=empirical(Z,X);assert(isequal(sort(synthetic),sort(X)));
  [pMSE,detail]=r1tune_pmse(X,synthetic,X,plan.classifier_seeds(rep),100,5);replicate=rep;AUC=detail.AUC;Brier=detail.Brier;
  scores=[scores;table(replicate,pMSE,AUC,Brier)];tmp=[scorepath '.tmp.csv'];writetable(scores,tmp);movefile(tmp,scorepath,'f');
  writejson(fullfile(out,'STATUS.json'),struct('stage','SCORING','config',id,'replicate',rep,'final_selection',false));fprintf('GAUSSIAN cfg%d rep%d pMSE %.7f\n',id,rep,pMSE);
 end
 s=struct('id',id,'alpha',cfg.alpha,'spike',cfg.spike,'pMSE_mean',mean(scores.pMSE),'pMSE_SD',std(scores.pMSE),'pMSE_MC_SE',std(scores.pMSE)/sqrt(50),'correlation_Rhat_max',diagnostics.split_rank_Rhat_max,'correlation_ESS_min',diagnostics.rank_ESS_min,'CPU_seconds',sum(cellfun(@(f)f.CPU,fits)),'rotation_relative_error_max',maxerr);writejson(done,s);if isempty(rankings),rankings=s;else,rankings(end+1)=s;end;
end
T=sortrows(struct2table(rankings),{'pMSE_mean','id'});writetable(T,fullfile(out,'RANKING.csv'));writejson(fullfile(out,'STATUS.json'),struct('stage','GRID_COMPLETE_PENDING_DIAGNOSTIC_REVIEW','completed',6,'selected_id_by_pMSE',T.id(1),'final_selection',false,'reason','Inspect invariant correlation diagnostics before accepting final comparison.'));
end
function S=empirical(Z,X)
Q=tiedrank(Z)/(size(Z,1)+1);S=zeros(size(Z));for j=1:size(Z,2),v=sort(X(:,j));S(:,j)=v(max(1,ceil(Q(:,j)*size(X,1))));end
end
function writejson(p,s)
fid=fopen([p '.tmp'],'w');assert(fid>=0);fprintf(fid,'%s\n',jsonencode(s,'PrettyPrint',true));fclose(fid);movefile([p '.tmp'],p,'f');
end
