function cpdde_paths(root)
% Resolve frozen dependencies inside this checkout; caller restores its path.
addpath(fullfile(root,'Utilities'),fullfile(root,'Utilities','GaussianCopula'));
addpath(genpath(fullfile(root,'Algorithms','DDE','CopDDE')));
addpath(fullfile(root,'Algorithms','DDE','cavi'),fullfile(root,'Algorithms','DDE','deep'));
addpath(fullfile(root,'Algorithms','EmpiricalDDE'),'-begin');
end
