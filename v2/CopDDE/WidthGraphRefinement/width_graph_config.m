function c=width_graph_config(overrides)
%WIDTH_GRAPH_CONFIG Matched fixed-width/fixed-penalty experiment, pilot first.
if nargin<1, overrides=struct; end
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
c=struct('source_dir',fullfile(root,'results','exact_gibbs_study'), ...
    'output_dir',fullfile(root,'results','width_graph_refinement','pilot'), ...
    'cache_dir','','n_values',[2000 8000],'replications',1:5, ...
    'seed',7324900,'resume',true,'job_index',1,'job_count',1, ...
    'make_reports',true,'verbose',true);
c.preparation=struct('seed_amplitude',0.005);
c.refinement=struct('iterations',100,'sweeps',1,'samples',1, ...
    'loading_tolerance',1e-6,'max_loading_iterations',2000, ...
    'probability_floor',1e-6,'verbose',false);
for key=fieldnames(overrides)'
    name=key{1}; assert(isfield(c,name),'WidthGraph:UnknownConfig','Unknown config %s',name);
    if isstruct(c.(name))
        for sub=fieldnames(overrides.(name))'
            k=sub{1}; assert(isfield(c.(name),k)); c.(name).(k)=overrides.(name).(k);
        end
    else, c.(name)=overrides.(name); end
end
assert(all(ismember(c.n_values,[500 2000 8000])) && all(ismember(c.replications,1:20)));
assert(numel(unique(c.n_values))==numel(c.n_values) && numel(unique(c.replications))==numel(c.replications));
assert(c.job_index>=1 && c.job_index<=c.job_count && c.job_count==fix(c.job_count));
assert(c.preparation.seed_amplitude>0 && c.preparation.seed_amplitude<.1);
assert(~strcmp(c.source_dir,c.output_dir));
end
