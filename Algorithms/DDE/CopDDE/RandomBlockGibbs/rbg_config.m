function c=rbg_config(overrides)
if nargin<1, overrides=struct; end
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
c=struct('source_dir',fullfile(root,'results','exact_gibbs_study'), ...
    'output_dir',fullfile(root,'results','random_block_gibbs','pilot'),'cache_dir','', ...
    'n_values',[2000 8000],'replications',1:5,'methods',{{'P','G1','G2','G4'}}, ...
    'seed',8324900,'repeat',1,'job_index',1,'job_count',1,'make_reports',false, ...
    'time_budget_reference','','verbose',true);
c.stage1=struct('iterations',50,'sweeps',1,'inverse_temperature',.7, ...
    'spike1',.02,'spike2',.05,'loading_tolerance',1e-6,'max_loading_iterations',2000, ...
    'probability_floor',1e-6,'max_seconds',Inf);
w=width_graph_config; c.stage2=w.refinement; c.seed_amplitude=.005;
for cell=fieldnames(overrides)'
    key=cell{1}; assert(isfield(c,key),'RandomBlock:Config','Unknown option %s',key);
    if isstruct(c.(key))
        for sub=fieldnames(overrides.(key))', name=sub{1}; assert(isfield(c.(key),name)); c.(key).(name)=overrides.(key).(name); end
    else, c.(key)=overrides.(key); end
end
assert(all(ismember(c.methods,{'P','G1','G2','G4','G6'})) && c.job_index<=c.job_count);
assert(~strcmp(c.source_dir,c.output_dir));
end
