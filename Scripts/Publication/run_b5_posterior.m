function run_b5_posterior(fitDir,out)
% Four chains for the reported B5 first-window analysis. Data remain local.
assert(java.io.File(out).isAbsolute()&&~isfolder(out)&&~isfile(out));
repo=fileparts(fileparts(fileparts(mfilename('fullpath'))));oldpath=path;cleanup=onCleanup(@()path(oldpath));
addpath(fullfile(repo,'Algorithms','DDE'),fullfile(repo,'Algorithms','EmpiricalDDE'));verify_revision_sources;cpdde_paths(repo);
addpath(fullfile(repo,'Algorithms','PosteriorGibbs'));
a=load(fullfile(fitDir,'FRESH_TRAINING_INPUT.mat'),'input');s=load(fullfile(fitDir,'RAW_CHECKPOINT.mat'),'record');f=s.record;assert(f.iteration==740&&isequal(size(f.B1),[98 13])&&isequal(size(f.B2),[12 5]));
mkdir(out);
for chain=1:4
 seed=930031+1500+chain*1000000+17000000;if chain==1,seed=35931531;end
 options=struct('seed',seed,'iterations',2000,'spike1',.05,'spike2',.1,'alpha1',12,'alpha2',4);
 run_posterior_gibbs(a.input,f,options,fullfile(out,sprintf('B5_chain%d',chain)));
end
end
