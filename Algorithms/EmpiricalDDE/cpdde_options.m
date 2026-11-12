function o=cpdde_options(o,J)
% Explicit Stage1 defaults; no automatic tuning or fitting authorization.
if nargin<1,o=struct;end
d=struct('depth',2,'max_widths',[floor(J/3),floor(floor(J/3)/3)], ...
 'init_method','spectral','initialization_seed',2026103037,'algorithm_seed',2026103037, ...
 'epsilon',1e-4,'regression_ridge',1e-6,'minimum_gamma',1e-3, ...
 'blocks',[3 3],'maximum_training_n',1200,'cap',600,'target',120, ...
 'permit_fitting',false,'cfg',struct('tau',.2,'spike1',.02,'spike2',.08, ...
 'beta0',.7,'alpha1',32,'alpha2',10));
for k=string(fieldnames(d))',if ~isfield(o,k),o.(k)=d.(k);end;end
assert(ismember(o.depth,[1 2]));o.max_widths=reshape(o.max_widths,1,[]);
if o.depth==1
 assert(numel(o.max_widths)>=1);o.max_widths=[o.max_widths(1),0];
else,assert(numel(o.max_widths)==2 && o.max_widths(2)>0);end
assert(o.max_widths(1)>0 && all(o.max_widths==fix(o.max_widths)));
assert(J>=3*o.max_widths(1) && o.max_widths(1)>=3*o.max_widths(2));
assert(ismember(o.init_method,{'spectral','random_binary'}));
assert(o.epsilon>0&&o.epsilon<.1&&o.minimum_gamma>0&&o.regression_ridge>0);
assert(o.cap>=110&&o.target>=1&&o.target<=o.cap&&o.cap==fix(o.cap)&&o.target==fix(o.target));
assert(numel(o.blocks)==2&&all(o.blocks>=1));
assert(o.initialization_seed>=0&&o.algorithm_seed>=0);
o.cfg=r1tune_config(o.cfg);
end
