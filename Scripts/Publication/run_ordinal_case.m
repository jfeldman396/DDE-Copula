function result=run_ordinal_case(family,method,out,target)
% Refit one reported ordinal model at its frozen selected hyperparameters.
assert(ismember(method,{'gibbs','maxima'}));
assert(java.io.File(out).isAbsolute()&&~isfolder(out)&&~isfile(out));
repo=fileparts(fileparts(fileparts(mfilename('fullpath'))));oldpath=path;oldrng=rng;cleanup=onCleanup(@()restore(oldpath,oldrng));
addpath(fullfile(repo,'Algorithms','DDE'),fullfile(repo,'Algorithms','EmpiricalDDE'));verify_revision_sources;cpdde_paths(repo);
% Ordinal data have no missing values; use the original group-maxima kernel.
addpath(fullfile(repo,'Algorithms','DDE','CopDDE','JointSamplerAcceleration'),'-begin');
addpath(fullfile(repo,'Algorithms','OrdinalDDE'));
plan=jsondecode(fileread(fullfile(repo,'Scripts','Publication','config','ordinal.json')));
ix=find(strcmp(family,{plan.cases.family}));assert(isscalar(ix));e=plan.cases(ix);hp=e.hp;
mkdir(out);data=r1t_data(family,plan.n,plan.J,plan.replication);initial=r1t_initialize_original(data.Ytrain,[9 3],e.initial_seed);
input=struct('initial',struct('A1',initial.A{1},'A2',initial.A{2},'B1',initial.B{1},'B2',initial.B{2},'Z',initial.Z,'gamma',initial.gamma,'prop',initial.prop),'n',plan.n,'J',plan.J,'R',initial.R,'replication',1,'algorithm_seed',e.algorithm_seed,'blocks',[3 3]);
if strcmp(method,'gibbs'),scheme='Z';else,scheme='H1';end
arm=struct('scheme',scheme,'cap',plan.cap,'tempering',true,'CP',true,'warmup',60,'SA_start',81);
cfg=struct('tau',hp.tau,'spike1',hp.t1,'spike2',hp.t2,'beta0',hp.beta0,'alpha1',hp.alpha(1),'alpha2',hp.alpha(2));
if nargin<4,target=plan.cap;end
raw=r1tune_fit(input,arm,fullfile(out,'checkpoint.mat'),target,true,false,true,cfg);
f=struct('A',{{raw.A1,raw.A2}},'B',{{raw.B1,raw.B2}},'gamma',raw.gamma,'prop',raw.prop,'Z',raw.Z,'active',{{raw.active1,raw.active2}});
result=struct('family',family,'method',method,'hp',hp,'raw',raw,'data',data,'complete',target==plan.cap||raw.stabilized,'settings',e);
if result.complete
 result.selected_model=r1tune_selected_model(f);
 result.pred=r1t_predict(result.selected_model,data.Ytrain,plan.synthetic_n,e.prediction_seed);
end
save(fullfile(out,'result.mat'),'result','-v7.3');
end
function restore(p,r),path(p);rng(r);end
