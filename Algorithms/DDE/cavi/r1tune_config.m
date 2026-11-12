function cfg=r1tune_config(cfg)
% Default settings reproduce the frozen current estimator exactly.
if nargin<1,cfg=struct;end
defaults=struct('spike1',.05,'spike2',.1,'tau',NaN,'beta0',.7,'alpha1',15,'alpha2',5);
for key=string(fieldnames(defaults))'
 if ~isfield(cfg,key),cfg.(key)=defaults.(key);end
end
assert(cfg.spike1>0 && cfg.spike2>0 && cfg.alpha1>0 && cfg.alpha2>0);
assert(cfg.beta0>0 && cfg.beta0<=1);
assert(isnan(cfg.tau) || (isfinite(cfg.tau) && cfg.tau>=0));
end
