function result=run_v2_case(J,N,replication,armID,outputDir)
%RUN_V2_CASE Run ONE fresh paper-study case using the unchanged LSA engine.
% Example (starts a fit): run_v2_case(50,500,1,1,'/new/absolute/output')
% Never points to production caches; refuses every pre-existing destination.
validateattributes(J,{'numeric'},{'scalar','integer'});assert(ismember(J,[50 100 150]));
validateattributes(N,{'numeric'},{'scalar','integer'});assert(ismember(N,[500 1000 2000 4000 8000 16000]));
validateattributes(replication,{'numeric'},{'scalar','integer','>=',1,'<=',100});
validateattributes(armID,{'numeric'},{'scalar','integer','>=',1,'<=',12});
outputDir=char(outputDir);outFile=java.io.File(outputDir);assert(outFile.isAbsolute(),'V2:AbsoluteOutput','Use an absolute output path.');
assert(~isfolder(outputDir)&&~isfile(outputDir),'V2:FreshOutput','Output must not exist.');
manifest=verify_v2_sources;root=fileparts(mfilename('fullpath'));
oldpath=path;oldcwd=pwd;oldrng=rng;cleanup=onCleanup(@()restore(oldpath,oldcwd,oldrng));
[ok,msg]=mkdir(outputDir);assert(ok,msg);outFile=java.io.File(outputDir);outputDir=char(outFile.getCanonicalPath());
% Copy only the published source into this run so legacy configuration mkdir
% side effects and relative result paths are confined to the new destination.
copyfile(fullfile(root,'CopDDE'),fullfile(outputDir,'CopDDE'));
addpath(genpath(fullfile(outputDir,'CopDDE')));cd(outputDir);
mkdir(fullfile('results','layer_specific_saem_study','inputs'));
arm=pfs_arm(armID);
job=struct('protocol','layer_specific_saem_v1','J',J,'n',N,'replication',replication,'arm',arm,'source_manifest',manifest,'entry_point','run_v2_case','shared_prefix_used',false);
save('job.mat','job');
input=lsa_input(J,N,replication);
fit=lsa_fit(input,arm,fullfile(outputDir,'checkpoint.mat'),arm.cap,~strcmp(arm.scheme,'H0'),false);
dde_assert_rank_consistency(fit.Z,input.R);
result=struct('job',job,'fit',fit,'metrics',psr_score(fit,pcp_truth(J)));
save('result.mat','result','-v7.3');
end
function restore(p,d,r)
cd(d);path(p);rng(r);
end
