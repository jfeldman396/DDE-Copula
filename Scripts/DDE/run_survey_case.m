function base=run_survey_case(data_input,sampler,index,out,target)
% Exact 12/4 empirical grid case; original data are available from the author.
% Omit TARGET for full fitting and all50 pMSE replicates; specify only for smoke tests.
assert(ismember(sampler,{'gibbs','maxima'})&&ismember(index,0:11));
assert(java.io.File(out).isAbsolute()&&~isfolder(out)&&~isfile(out));
repo=fileparts(fileparts(fileparts(mfilename('fullpath'))));oldpath=path;oldrng=rng;cleanup=onCleanup(@()restore(oldpath,oldrng));
addpath(fullfile(repo,'Algorithms','DDE'));verify_revision_sources;cpdde_paths(repo);
plan=jsondecode(fileread(fullfile(repo,'Scripts','Publication','config','empirical.json')));
train=load(data_input);assert(isequal(size(train.Y),[1500 98])&&sum(isnan(train.Y),'all')==223&&isequal(train.X,fillmissing(train.Y,'constant',3)));
% Verify the value identity rather than timestamp-dependent MAT container bytes.
y=train.Y;y(isnan(y))=0;bytes=[uint8(isnan(train.Y(:)));reshape(typecast(y(:),'uint8'),[],1);reshape(typecast(train.X(:),'uint8'),[],1)];
md=java.security.MessageDigest.getInstance('SHA-256');md.update(bytes);hash=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
factor=jsondecode(fileread(fullfile(repo,'Scripts','Publication','config','factor.json')));assert(strcmp(hash,factor.prepared_values_sha256));
metadata=readtable(fullfile(repo,'Results','Publication','data','ivae','item_metadata.csv'),'TextType','string');train.item_ids=cellstr(metadata.item_id);train.row_ids=(1:1500)';
train.role='training';train.split_id='NEW_FULL1500_PMSE_GRID192_20261102';train.forbidden_row_ids=[];train.input_path=data_input;train.input_file_sha256=hash;
e=plan.cases(index+1);assert(e.config_id==index);
options=struct('depth',2,'max_widths',[12 4],'init_method','spectral','initialization_seed',e.fit_seed,'algorithm_seed',e.fit_seed,'maximum_training_n',1500,'cap',plan.cap,'target',100,'permit_fitting',true,'cfg',e.cfg,'blocks',[6 4],'sampler',sampler);options.layer_partitions={[6 6],[4]};
mkdir(out);if nargin<5,targets=plan.chunk_iterations:plan.chunk_iterations:plan.cap;else,targets=target;end
for budget=targets
 options.target=budget;[model,audit]=pmse_fit_training(repo,train,options,fullfile(out,'fit'));
 if audit.stabilized||budget==plan.cap,break;end
end
base=struct('sampler',sampler,'config_id',index,'cfg',e.cfg,'fit_seed',e.fit_seed,'widths',model.widths,'iterations',model.iterations,'stabilized',audit.stabilized,'eligible',audit.eligible,'CM_flags',audit.CM_flags,'complete',nargin<5);
if nargin>=5,base.status='SMOKE_ONLY_NO_SCORES';writejson(fullfile(out,'STATUS.json'),base);return;end
if ~audit.eligible||~audit.stabilized,base.status='PRESERVED_NOT_RANKABLE';writejson(fullfile(out,'COMPLETE.json'),base);return;end
rows=struct([]);
for rep=1:plan.score_replicates
 rng(plan.generation_seeds(rep),'twister');synthetic=generate_X_Cop_pred(1500,model.prop,model.B1,model.B2,model.gamma,train.X);
 assert(isequal(sort(synthetic,1),sort(train.X,1)));
 [v,d]=r1tune_pmse(train.X,synthetic,train.X,plan.classifier_seeds(rep),plan.score_trees,plan.score_folds);
 row=struct('replicate',rep,'generation_seed',plan.generation_seeds(rep),'classifier_seed',plan.classifier_seeds(rep),'pMSE',v,'AUC',d.AUC,'Brier',d.Brier,'exact_empirical_margins',true);
 if isempty(rows),rows=row;else,rows(end+1)=row;end
 writetable(struct2table(rows),fullfile(out,'PMSE_REPLICATES.csv'));
end
base.status='COMPLETE_STABILIZED_VALID_ALL50_PMSE';base.pMSE_mean=mean([rows.pMSE]);base.pMSE_SD=std([rows.pMSE]);base.pMSE_MC_SE=std([rows.pMSE])/sqrt(50);
writejson(fullfile(out,'COMPLETE.json'),base);
end
function writejson(p,o),f=fopen(p,'w');assert(f>=0);c=onCleanup(@()fclose(f));fprintf(f,'%s\n',jsonencode(o,PrettyPrint=true));end
function restore(p,r),path(p);rng(r);end
