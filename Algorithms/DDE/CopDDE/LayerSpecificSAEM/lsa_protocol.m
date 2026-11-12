function c=lsa_protocol
c=struct('out','results/layer_specific_saem_study','J',[50 100 150],'n',[500 1000 2000 4000 8000 16000],'replications',1:100,'post_cap',100,'minimum_post',40,'window',10,'check_every',5,'required_checks',3,'weight_tolerance',.005,'gamma_tolerance',.01,'prop_tolerance',.005,'alpha',[15 5],'spikes',[.05 .1],'checkpoint_every',10,'single_threaded',true,'initial_workers',18);
end
