function dde_assert_rank_consistency(Z,R)
%DDE_ASSERT_RANK_CONSISTENCY Strict inequalities across distinct observed ranks.
assert(isequal(size(Z),size(R)), 'DDE:RankSize', 'Z and R must match.');
assert(all(isfinite(Z),'all'), 'DDE:FiniteZ', 'Z must be finite.');
for j=1:size(R,2)
    observed = ~isnan(R(:,j));
    if ~any(observed), continue; end
    [~,~,g] = unique(R(observed,j),'sorted');
    lo = accumarray(g,Z(observed,j),[],@min);
    hi = accumarray(g,Z(observed,j),[],@max);
    assert(all(hi(1:end-1)<lo(2:end)), 'DDE:RankViolation', ...
        'Latent Z violates strict rank constraints in column %d.',j);
end
end
