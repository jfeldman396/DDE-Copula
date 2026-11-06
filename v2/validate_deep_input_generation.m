function validate_deep_input_generation(archivedInputDir,newOutputDir)
% Check fresh inputs against archived synthetic inputs without fitting models.
assert(isfolder(archivedInputDir));
assert(java.io.File(newOutputDir).isAbsolute());
assert(~isfolder(newOutputDir)&&~isfile(newOutputDir));
mkdir(newOutputDir);
fields={'X','R','Rlevels','Z_true','A_true','initial', ...
 'truth_marginal_mean','truth_marginal_sd','algorithm_seed','initialization_seed'};
for n=[500 16000]
 fresh=create_deep_nofloor_input(n,1,fullfile(newOutputDir,sprintf('input_n%d_r1.mat',n)));
 s=load(fullfile(archivedInputDir,sprintf('input_n%05d_r001.mat',n)),'input');
 for k=1:numel(fields)
  assert(isequaln(fresh.(fields{k}),s.input.(fields{k})),['Mismatch: ' fields{k}]);
 end
 fprintf('EXACT_PARITY n=%d rep=1\n',n);
end
try
 create_deep_nofloor_input(500,76,fullfile(newOutputDir,'input_n500_r76.mat'));
 error('Validation:ExpectedFailure','Expected zero-width initialization failure');
catch err
 assert(strcmp(err.identifier,'Gap5:InvalidWidth'));
 fprintf('EXPECTED_FAILURE n=500 rep=76\n');
end
end
