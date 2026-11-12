function sources=cpdde_sources(root)
% Bind the current resolved numerical dependency closure to each new fit.
names={'agr_log_interval','agr_slice','agr_truncated_standard_normal','binary','bl_mix','cf_save','cma_column','cma_conditional','cma_logh','cmv_cavi','cpd_weighted_l1_cm','cpdde_accumulate','cpdde_digest','cpdde_engine_partitioned','cpdde_export','cpdde_initial','cpdde_options','cpdde_paths','cpdde_sources','dde_assert_binary_model','dde_assert_rank_consistency','dde_dependence_mixture','dde_logistic','dde_rank_gibbs','dde_softplus','foa_accumulate','foa_cm','full_cavi_binary_partitioned','generate_X_Cop_pred','logistic','lsa_protocol','lsa_stop','pcs_csp_update','pfs_arm','pilot_mix','pmse_fit_training','r1st_cma_observed','r1tune_beta','r1tune_config','r1tune_pmse','r1tune_selected_model','sao_cm','sob_cm'};
sources=struct('name',{},'path',{},'sha256',{});
for i=1:numel(names)
 p=which(names{i});assert(startsWith(p,root),'Source resolved outside checkout: %s',names{i});
 fid=fopen(p,'rb');assert(fid>=0);bytes=fread(fid,Inf,'*uint8');fclose(fid);
 md=java.security.MessageDigest.getInstance('SHA-256');md.update(bytes);
 h=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2)',1,[]));
 sources(i)=struct('name',names{i},'path',p,'sha256',h);
end
end
