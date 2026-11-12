function [z,info]=cma_column(z,R,mu,gamma,sweeps,order)
% Missing entries impose NO rank constraints; use original kernel on observed rows.
ok=~isnan(R);
if any(ok)
 [z(ok),info]=r1st_cma_observed(z(ok),R(ok),mu(ok),gamma,sweeps,order);
else
 info=struct('random_TN_draws',0,'regenerated_observations',0,'groups',0,'seconds',0);
end
z(~ok)=mu(~ok)+sqrt(gamma)*randn(sum(~ok),1);
info.missing_unconstrained_draws=sum(~ok);
dde_assert_rank_consistency(z,R);
end
