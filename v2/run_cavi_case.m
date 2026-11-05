function result=run_cavi_case(J,N,replication,armID,outputDir,iterationLimit)
%RUN_CAVI_CASE Fresh maxima fit: arm1 tempering+CAVI; arm2 CAVI; arm3 tempering.
% Example (starts a fit): run_cavi_case(50,500,1,1,'/new/absolute/output')
% Never points to production caches; refuses every pre-existing destination.
validateattributes(J,{'numeric'},{'scalar','integer'});assert(ismember(J,[50 100 150]));
validateattributes(N,{'numeric'},{'scalar','integer'});assert(ismember(N,[500 1000 2000 4000 8000 16000]));
validateattributes(replication,{'numeric'},{'scalar','integer','>=',1,'<=',100});
validateattributes(armID,{'numeric'},{'scalar','integer','>=',1,'<=',3});
outputDir=char(outputDir);outFile=java.io.File(outputDir);assert(outFile.isAbsolute(),'V2:AbsoluteOutput','Use an absolute output path.');
assert(~isfolder(outputDir)&&~isfile(outputDir),'V2:FreshOutput','Output must not exist.');
baselineManifest=verify_v2_sources;manifest=verify_cavi_sources;root=fileparts(mfilename('fullpath'));
oldpath=path;oldcwd=pwd;oldrng=rng;cleanup=onCleanup(@()restore(oldpath,oldcwd,oldrng));
[ok,msg]=mkdir(outputDir);assert(ok,msg);outFile=java.io.File(outputDir);outputDir=char(outFile.getCanonicalPath());
% Copy only the published source into this run so legacy configuration mkdir
% side effects and relative result paths are confined to the new destination.
copyfile(fullfile(root,'CopDDE'),fullfile(outputDir,'CopDDE'));
copyfile(fullfile(root,'cavi'),fullfile(outputDir,'cavi'));
addpath(genpath(fullfile(outputDir,'CopDDE')));addpath(fullfile(outputDir,'cavi'),'-begin');cd(outputDir);
mkdir(fullfile('results','layer_specific_saem_study','inputs'));
mkdir(fullfile('results','paper_cp_comparison'));copyfile(fullfile(outputDir,'cavi','truth','*.mat'),fullfile('results','paper_cp_comparison'));

arm=pfs_arm(armID);if nargin<6,iterationLimit=arm.cap;end;validateattributes(iterationLimit,{'numeric'},{'scalar','integer','>=',1,'<=',arm.cap});
job=struct('protocol','maxima_CAVI_full_objective_v1','J',J,'n',N,'replication',replication,'arm',arm,'source_manifest',manifest,'baseline_manifest',baselineManifest,'iteration_limit',iterationLimit,'entry_point','run_cavi_case','shared_prefix_used',false);
save('job.mat','job');
input=lsa_input(J,N,replication);
fit=full_cavi_fit(input,arm,fullfile(outputDir,'checkpoint.mat'),iterationLimit,true,false,armID~=3);
dde_assert_rank_consistency(fit.Z,input.R);
result=struct('complete',fit.stabilized||fit.iteration>=arm.cap,'job',job,'fit',fit,'metrics',psr_score(fit,pcp_truth(J)));
save('result.mat','result','-v7.3');
end
function restore(p,d,r)
cd(d);path(p);rng(r);
end
