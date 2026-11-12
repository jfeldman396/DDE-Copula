function result=run_original_rl_saved_case(inputFile,outputDir)
% Original fixed-width truncated-lasso RL-SAEM comparator; full upstream stopping protocol.
% Saved synthetic input must contain record with data/algorithm seeds.
assert(java.io.File(inputFile).isAbsolute()&&java.io.File(outputDir).isAbsolute());
assert(isfile(inputFile)&&~isfile(outputDir)&&~isfolder(outputDir));
baseline=verify_v2_sources;verify_cavi_sources;manifest=verify_original_rl_sources;
root=fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))),'Algorithms','DDE');oldpath=path;oldcwd=pwd;oldrng=rng;
assert(isempty(gcp('nocreate')),'V2:ActivePool','Run in a MATLAB session without an active parallel pool.');
settings=parallel.Settings;oldAuto=settings.Pool.AutoCreate;settings.Pool.AutoCreate=false;
cleanup=onCleanup(@()restore(oldpath,oldcwd,oldrng,settings,oldAuto));
s=load(inputFile,'record');input=s.record;[N,J]=size(input.R);
assert(J==150,'V2:Scope','This wrapper reproduces the J150 original-comparator rerun.');
t=load(fullfile(root,'cavi','truth',sprintf('truth_J%d.mat',J)),'truth');truth=t.truth;
mkdir(outputDir);copyfile(fullfile(root,'CopDDE'),fullfile(outputDir,'CopDDE'));
copyfile(fullfile(root,'comparators','original_rl'),fullfile(outputDir,'original_rl'));
addpath(genpath(fullfile(outputDir,'CopDDE')));
addpath(fullfile(outputDir,'original_rl'),'-begin');cd(outputDir);
rng(input.data_seed,'twister');
[~,X]=generate_X_Cop_marg_emp(N,truth.prop(:)',truth.lambdas,truth.gamma,truth.B1_scaled(:,1:11),truth.B2(1:10,1:4));
R=zeros(N,J);for j=1:J,[~,~,R(:,j)]=unique(X(:,j),'sorted');end
assert(isequal(R,input.R),'V2:PairedData','Paired data regeneration mismatch');
a=input.initial;rng(input.data_seed+700000,'twister');
[a.prop,a.B1,a.B2,a.gamma,a.A1,a.A2]=Normal_init_v2(a.Z,floor(J/3),floor(J/3),floor(J/9),1e-4);
rng(input.algorithm_seed,'twister');timer=tic;
[gamma,~,~,~,~,Z,prop,B1,B2,A1,A2,it]=get_SAEM_RL(X,a.Z,input.R,max(input.R,[],1),a.prop,a.B1,a.B2,a.gamma(:)',N^.25,N^.25,max(.3,3*N^(-.3)),a.A1,a.A2,1,ceil(size(a.A2,2)/2),50,.7);
seconds=toc(timer);assert(all(isfinite([prop(:);B1(:);B2(:)])));
job=struct('input_file',inputFile,'source_manifest',manifest,'baseline_manifest',baseline,'protocol','original_fixed_width_truncated_lasso_RL_SAEM');
result=struct('job',job,'gamma',gamma,'Z',Z,'prop',prop,'B1',B1,'B2',B2,'A1',A1,'A2',A2,'iterations',it,'seconds',seconds);
save('result.mat','result','-v7.3');
end
function restore(p,d,r,settings,oldAuto)
cd(d);path(p);rng(r);settings.Pool.AutoCreate=oldAuto;
end
