function c=pcp_protocol
c.root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
c.output_dir=fullfile(c.root,'results','paper_cp_comparison');
if ~isfolder(c.output_dir), mkdir(c.output_dir); end
c.J=[50 100 150]; c.n=[500 1000 2000 4000 8000 16000]; c.replications=1:100;
base=rbg_config; c.options=base.stage1; c.options.warm_iterations=30;
c.options.inverse_temperature=.7;
c.description=['CP both layers; random first-layer blocks of four; whole-layer joint top ', ...
    'Gibbs; original paper temperature and J-specific spike scales; corrected ', ...
    'shared CM/rank engine from the continuation studies, 50 fixed iterations.'];
c.beta=zeros(1,50); beta=.7;
for it=1:50
    c.beta(it)=beta;
    if it>5, beta=min(1,beta+.01*(it-6)); end
end
assert(c.beta(1)==.7 && c.beta(14)<1 && c.beta(15)==1);
end
