function stage_dde_sources(root,outputDir,includeDeep)
% Build the same isolated source tree, reusing byte-identical original helpers.
copyfile(fullfile(root,'CopDDE'),fullfile(outputDir,'CopDDE'));
copyfile(fullfile(root,'cavi'),fullfile(outputDir,'cavi'));
if includeDeep,copyfile(fullfile(root,'deep'),fullfile(outputDir,'deep'));end
original=fullfile(fileparts(fileparts(root)),'Utilities');
files={
 'deep/input_generation/generate_X_cop_D.m';
 'deep/input_generation/binary_init.m';
 'deep/input_generation/Normal_init_D.m';
 'deep/input_generation/logit.m';
 'CopDDE/Utilities/thres.m';
 'CopDDE/Utilities/munkres.m';
 'CopDDE/Utilities/sim_block_loadings.m';
 'CopDDE/Utilities/rescale_B1.m';
 'CopDDE/Utilities/logit.m';
 'CopDDE/Utilities/logistic.m';
 'CopDDE/Utilities/generate_X_Cop_marg_emp.m';
 'CopDDE/Utilities/format_estimates.m';
};
for k=1:numel(files)
 relative=files{k};if startsWith(relative,'deep/') && ~includeDeep,continue;end
 [folder,name,ext]=fileparts(relative);destination=fullfile(outputDir,folder);
 if ~isfolder(destination),mkdir(destination);end
 copyfile(fullfile(original,[name ext]),fullfile(destination,[name ext]));
end
end
