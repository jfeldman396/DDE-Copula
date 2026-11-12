function metrics=score_deep_saved_case(resultFile,inputFile,truthFile)
% Score an archived synthetic fit against its matching input and truth.
% Does not write to input, result, or production directories.
verify_v2_sources;verify_deep_sources;
root=fileparts(mfilename('fullpath'));oldpath=path;
cleanup=onCleanup(@()path(oldpath));
addpath(genpath(fullfile(root,'CopDDE')));addpath(fullfile(root,'deep'),'-begin');
r=load(resultFile,'result');s=load(inputFile,'input');t=load(truthFile,'record');
f=r.result.fit;input=s.input;truth=t.record;
assert(size(f.Z,1)==input.n && size(f.Z,2)==input.J);
assert(numel(f.B)==3 && isequal(truth.true_widths,[18 6 2]));
fit=struct('B',{f.history{end}.B},'Z',f.Z,'num_act',cellfun(@sum,f.active));
for d=1:3
 b=zeros(size(truth.B_padded{d}));
 b(1:size(fit.B{d},1),1:size(fit.B{d},2))=fit.B{d};fit.B{d}=b;
end
metrics=dpr_metrics(fit,input,truth);
metrics.initial_widths=input.initial.widths;
end
