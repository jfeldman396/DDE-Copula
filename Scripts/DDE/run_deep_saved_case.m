function result=run_deep_saved_case(inputFile,outputDir,iterationLimit)
% Reproduce the no-floor fit from an archived synthetic input and its seeds.
% Input creation and dimension-failure accounting remain separate protocols.
if nargin<3,iterationLimit=160;end
validateattributes(iterationLimit,{'numeric'},{'scalar','integer','>=',1,'<=',160});
inputFile=char(inputFile);outputDir=char(outputDir);
assert(java.io.File(inputFile).isAbsolute()&&java.io.File(outputDir).isAbsolute(),'V2:AbsolutePaths','Use absolute paths.');
assert(isfile(inputFile),'V2:MissingInput','Saved input does not exist.');
assert(~isfolder(outputDir)&&~isfile(outputDir),'V2:FreshOutput','Output must not exist.');
baseline=verify_revision_sources;cavi=baseline;deep=baseline;
s=load(inputFile,'input');input=s.input;
assert(input.J==108&&numel(input.initial.A)==3);
assert(isequal(input.initial.widths,[size(input.initial.A{1},2),size(input.initial.A{2},2),size(input.initial.A{3},2)]));
assert(all(input.initial.widths>=1));
root=fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))),'Algorithms','DDE');oldpath=path;oldcwd=pwd;oldrng=rng;
cleanup=onCleanup(@()restore(oldpath,oldcwd,oldrng));
[ok,msg]=mkdir(outputDir);assert(ok,msg);
copyfile(fullfile(root,'CopDDE'),fullfile(outputDir,'CopDDE'));
copyfile(fullfile(root,'cavi'),fullfile(outputDir,'cavi'));
copyfile(fullfile(root,'deep'),fullfile(outputDir,'deep'));
addpath(genpath(fullfile(outputDir,'CopDDE')));addpath(fullfile(outputDir,'cavi'),'-begin');addpath(fullfile(outputDir,'deep'),'-begin');cd(outputDir);
hp=struct('beta0',.9,'cavi',true,'spikes',[.01 .03 .005],'alpha',[36 12 4],'tau',3*input.n^(-.3));
job=struct('protocol','saved_input_nofloor_gap5','input_file',inputFile,'iteration_limit',iterationLimit,'baseline_manifest',baseline,'cavi_manifest',cavi,'deep_manifest',deep);
save('job.mat','job');
fit=revision_deep_fit(input,hp,fullfile(outputDir,'checkpoint.mat'),iterationLimit);
result=struct('complete',fit.stabilized||fit.iteration==160,'job',job,'fit',fit);
save('result.mat','result','-v7.3');
end
function restore(p,d,r)
cd(d);path(p);rng(r);
end
