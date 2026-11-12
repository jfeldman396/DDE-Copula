function [model,audit]=pmse_fit_training(root,train,options,folder)
% Sole adapter entry: fresh train-only initialization or identity-bound resume.
% Root must explicitly release Stage1 fitting after data/oracle validation.
cpdde_paths(root);options=cpdde_options(options,size(train.Y,2));
assert(options.permit_fitting,'ConditionalDDE:FittingNotReleased','Stage1 data and predictive scoring must be validated before fitting.');
if ~isfolder(folder),mkdir(folder);end
sources=cpdde_sources(root);
identity_file=fullfile(folder,'TRAINING_IDENTITY.mat');
[input,training_audit,marginals]=cpdde_initial(train,options);
assert(isequal(options.max_widths,[12 4])&&isfield(options,'layer_partitions')&&isequal(options.layer_partitions,{[6 6],[4]}));
input.layer_partitions=options.layer_partitions;
input.initial.source_identity=cpdde_digest({sources});
protocol=options;protocol=rmfield(protocol,{'target','permit_fitting'});
identity=struct('training_identity',training_audit.training_identity, ...
 'initial_fingerprint',training_audit.unbounded_initial_fingerprint, ...
 'protocol',protocol,'sources',sources,'train_audit',training_audit,'marginals',{marginals});
if isfile(identity_file)
 saved=load(identity_file,'identity');assert(isequaln(saved.identity,identity),'ConditionalDDE:ResumeIdentityMismatch','Training data, initializer, settings or source changed.');
else,save(identity_file,'identity','-v7.3');end
initial_file=fullfile(folder,'FRESH_TRAINING_INPUT.mat');
if ~isfile(initial_file),save(initial_file,'input','training_audit','marginals','options','-v7.3');end
if strcmp(options.sampler,'gibbs'),arm=pfs_arm(9);else,assert(strcmp(options.sampler,'maxima'));arm=pfs_arm(1);end;arm.cap=options.cap;
checkpoint=fullfile(folder,'RAW_CHECKPOINT.mat');
if isfile(checkpoint)
 saved=load(checkpoint,'record');f=saved.record;
 assert(strcmp(f.training_identity,input.training_identity)&&strcmp(f.initialization_fingerprint,training_audit.unbounded_initial_fingerprint));
 assert(strcmp(f.source_identity,input.initial.source_identity)&&f.seed==input.algorithm_seed&&f.n==input.n&&f.J==input.J);
 assert(f.arm.cap==arm.cap&&strcmp(f.arm.scheme,arm.scheme)&&f.arm.CP&&f.arm.tempering&&f.arm.warmup==60&&f.arm.SA_start==81);
 assert(f.iteration<=options.target,'ConditionalDDE:LaterCheckpoint','A later endpoint cannot replace an earlier requested budget.');
end
raw=full_cavi_fit(input,arm,checkpoint,options.target,true,false,true,options.cfg);
if options.depth==1,assert(isempty(raw.A2)&&size(raw.A2,1)==input.n&&size(raw.B2,2)==1&&isempty(raw.prop));end
dde_assert_rank_consistency(raw.Z,input.R);
[model,audit,full_selected]=cpdde_export(raw,training_audit,marginals,sources);
model.kernel=options.sampler;audit.sampler=options.sampler;model.binary_layer_partitions=options.layer_partitions;audit.binary_layer_partitions=options.layer_partitions;
audit.initialization=training_audit;audit.cap=options.cap;audit.target=options.target;
audit.rank_constraints_verified=true;
audit.cap_reached_without_stability=raw.iteration==options.cap&&~raw.stabilized;
destination=fullfile(folder,sprintf('MODEL_it%04d_target%04d.mat',raw.iteration,options.target));
if isfile(destination)
 saved=load(destination,'model','audit');assert(isequaln(saved.model,model)&&isequaln(saved.audit,audit),'ConditionalDDE:ImmutableModelMismatch');
else,save(destination,'model','audit','full_selected','-v7.3');end
fid=fopen(fullfile(folder,sprintf('AUDIT_it%04d_target%04d.json',raw.iteration,options.target)),'w');assert(fid>=0);cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',jsonencode(audit,'PrettyPrint',true));
end

