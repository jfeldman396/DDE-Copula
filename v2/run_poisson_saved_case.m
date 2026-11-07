function result=run_poisson_saved_case(inputFile,outputDir)
% Original fixed-width Poisson comparator; full upstream stopping protocol.
% Saved synthetic input must contain record with data/algorithm seeds.
assert(java.io.File(inputFile).isAbsolute()&&java.io.File(outputDir).isAbsolute());
assert(isfile(inputFile)&&~isfile(outputDir)&&~isfolder(outputDir));
baseline=verify_v2_sources;verify_cavi_sources;manifest=verify_poisson_sources;
root=fileparts(mfilename('fullpath'));oldpath=path;oldcwd=pwd;oldrng=rng;
cleanup=onCleanup(@()restore(oldpath,oldcwd,oldrng));
s=load(inputFile,'record');input=s.record;[N,J]=size(input.R);
assert(ismember(J,[50 100 150]));
t=load(fullfile(root,'cavi','truth',sprintf('truth_J%d.mat',J)),'truth');truth=t.truth;
mkdir(outputDir);copyfile(fullfile(root,'CopDDE'),fullfile(outputDir,'CopDDE'));
copyfile(fullfile(root,'comparators','poisson'),fullfile(outputDir,'poisson'));
addpath(genpath(fullfile(outputDir,'CopDDE')));
addpath(fullfile(outputDir,'poisson','utilities'),'-begin');
addpath(fullfile(outputDir,'poisson','main algorithms','Poisson'),'-begin');cd(outputDir);
rng(input.data_seed,'twister');
[~,X]=generate_X_Cop_marg_emp(N,truth.prop(:)',truth.lambdas,truth.gamma,truth.B1_scaled(:,1:11),truth.B2(1:10,1:4));
R=zeros(N,J);for j=1:J,[~,~,R(:,j)]=unique(X(:,j),'sorted');end
assert(isequal(R,input.R),'V2:PairedData','Paired data regeneration mismatch');
a=input.initial;rng(input.algorithm_seed,'twister');timer=tic;
[prop,B1,B2,A1,A2,it]=get_SAEM_poisson(X,a.prop,a.B1,a.B2,N^.25,N^.25,max(.3,3*N^(-.3)),a.A1,a.A2,1,ceil(size(a.A2,2)/2));
seconds=toc(timer);assert(all(isfinite([prop(:);B1(:);B2(:)])));
job=struct('input_file',inputFile,'source_manifest',manifest,'baseline_manifest',baseline,'protocol','original_fixed_width_Poisson');
result=struct('job',job,'prop',prop,'B1',B1,'B2',B2,'A1',A1,'A2',A2,'iterations',it,'seconds',seconds);
save('result.mat','result','-v7.3');
end
function restore(p,d,r)
cd(d);path(p);rng(r);
end
