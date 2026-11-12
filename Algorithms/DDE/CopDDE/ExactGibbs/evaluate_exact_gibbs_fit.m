function metrics = evaluate_exact_gibbs_fit(fit, truth, alignment)
%EVALUATE_EXACT_GIBBS_FIT Align, rescale, and evaluate one two-layer fit.
% All primary errors and support accuracies EXCLUDE intercepts and include
% the full maximum-width, zero-padded matrices. Pass Method A's metrics as
% ALIGNMENT when evaluating B: the assignments, selected support, and widths
% then remain those of pretraining. B.runtime_seconds is its incremental cost.

if nargin < 3, alignment = []; end
required = {'B1_unscaled','B1_scaled','B2','K1_true','K2_true'};
for k = 1:numel(required)
    assert(isfield(truth, required{k}), 'Missing truth.%s.', required{k});
end
[J, K1p] = size(truth.B1_unscaled);
K1 = K1p - 1;
K2 = size(truth.B2, 2) - 1;
assert(isequal(size(truth.B1_scaled), [J,K1+1]));
assert(isequal(size(truth.B2), [K1,K2+1]));
assert(isfield(fit,'Z') && size(fit.Z,2) == J && size(fit.Z,1) > 1, ...
    'Empirical marginal rescaling requires the final latent Gaussian Z.');
assert(all(isfinite(fit.Z(:))), 'Final Z must be finite for rescaling.');

% Refinement stores compact matrices but retains the original node indices.
if isfield(fit, 'architecture')
    arch = fit.architecture;
    index1 = arch.indices1(:)'; index2 = arch.indices2(:)';
    assert(numel(unique(index1)) == numel(index1) && ...
        all(index1 >= 1 & index1 <= K1));
    assert(numel(unique(index2)) == numel(index2) && ...
        all(index2 >= 1 & index2 <= K2));
    assert(isequal(size(fit.B1), [J,numel(index1)+1]));
    assert(isequal(size(fit.B2), [numel(index1),numel(index2)+1]));
    B1 = zeros(J,K1+1); B2 = zeros(K1,K2+1);
    B1(:,1) = fit.B1(:,1); B1(:,index1+1) = fit.B1(:,2:end);
    B2(index1,1) = fit.B2(:,1);
    B2(index1,index2+1) = fit.B2(:,2:end);
    prop = 0.5*ones(1,K2); prop(index2) = fit.prop(:)';
    selected1 = false(J,K1); selected2 = false(K1,K2);
    selected1(:,index1) = logical(arch.support1);
    selected2(index1,index2) = logical(arch.support2);
    width1 = numel(index1); width2 = numel(index2);
else
    assert(isequal(size(fit.B1), [J,K1+1]));
    assert(isequal(size(fit.B2), [K1,K2+1]));
    B1 = fit.B1; B2 = fit.B2; prop = fit.prop(:)';
    selected1 = get_field(fit,'support1',B1(:,2:end) ~= 0);
    selected2 = get_field(fit,'support2',B2(:,2:end) ~= 0);
    width1 = sum(get_field(fit,'active1',any(selected1,1)));
    width2 = sum(get_field(fit,'active2',any(selected2,1)));
end
assert(numel(prop) == K2 && numel(fit.gamma) == J);
assert(isequal(size(selected1),[J,K1]) && isequal(size(selected2),[K1,K2]));
assert(all(isfinite(B1(:))) && all(isfinite(B2(:))));

% Reuse the original utility. Its assignments are intentionally retained for
% B so a post-refinement label permutation cannot manufacture improvement.
[B1a,B2a,gammaa,propa,a1,a2] = format_estimates( ...
    truth.B1_unscaled,truth.B2,B1,B2,fit.gamma,prop,true,fit.Z,true);
if ~isempty(alignment)
    a1 = alignment.assignment1; a2 = alignment.assignment2;
    assert(isequal(sort(a1(:)'),1:K1) && isequal(sort(a2(:)'),1:K2));
    B1a = [B1(:,1),B1(:,a1+1)];
    B2a = [B2(a1,1),B2(a1,a2+1)];
    propa = prop(a2);
    [B1a,gammaa] = rescale_B1(propa,B2a,B1a,fit.gamma,true,fit.Z);
end
selected1 = logical(selected1(:,a1));
selected2 = logical(selected2(a1,a2));
if ~isempty(alignment)
    selected1 = alignment.selected_support1;
    selected2 = alignment.selected_support2;
    width1 = alignment.K1; width2 = alignment.K2;
end
truth1 = truth.B1_scaled(:,2:end);
truth2 = truth.B2(:,2:end);
err1 = (B1a(:,2:end)-truth1).^2;
err2 = (B2a(:,2:end)-truth2).^2;
support1 = truth1 ~= 0; support2 = truth2 ~= 0;
k1t = truth.K1_true; k2t = truth.K2_true;

metrics = struct();
metrics.mse_B1 = mean(err1(:));
metrics.mse_B2 = mean(err2(:));
metrics.mse_B1_true_block = mean(err1(:,1:k1t),'all');
metrics.mse_B2_true_block = mean(err2(1:k1t,1:k2t),'all');
metrics.support_accuracy1 = mean(selected1 == support1,'all');
metrics.support_accuracy2 = mean(selected2 == support2,'all');
metrics.support_accuracy1_true_block = mean(selected1(:,1:k1t) == support1(:,1:k1t),'all');
metrics.support_accuracy2_true_block = mean(selected2(1:k1t,1:k2t) == support2(1:k1t,1:k2t),'all');
metrics.support_tpr1 = conditional_mean(selected1,support1);
metrics.support_tpr2 = conditional_mean(selected2,support2);
metrics.support_tnr1 = conditional_mean(~selected1,~support1);
metrics.support_tnr2 = conditional_mean(~selected2,~support2);
metrics.K1 = width1; metrics.K2 = width2;
metrics.runtime_seconds = get_field(fit,'runtime_seconds',NaN);
metrics.refinement_runtime_seconds = 0;
if isfield(fit,'architecture')
    metrics.refinement_runtime_seconds = metrics.runtime_seconds;
    if ~isempty(alignment)
        metrics.runtime_seconds = alignment.runtime_seconds + metrics.refinement_runtime_seconds;
    elseif isfield(fit,'pretraining_runtime_seconds')
        metrics.runtime_seconds = fit.pretraining_runtime_seconds + metrics.refinement_runtime_seconds;
    else
        metrics.runtime_seconds = NaN; % A's runtime is needed for the total.
    end
end
metrics.assignment1 = a1; metrics.assignment2 = a2;
metrics.B1_aligned = B1a; metrics.B2_aligned = B2a;
metrics.gamma_aligned = gammaa; metrics.prop_aligned = propa;
metrics.selected_support1 = selected1; metrics.selected_support2 = selected2;
metrics.coefficient_support1 = B1a(:,2:end) ~= 0;
metrics.coefficient_support2 = B2a(:,2:end) ~= 0;
metrics.alignment_frozen = ~isempty(alignment);
metrics.rank_consistent = get_field(fit,'rank_consistent',NaN);
metrics.initial_rank_repaired_columns = numel(get_field(fit,'initial_rank_repaired_columns',[]));
metrics.metric_definition = ['Non-intercept entries; full maximum-width zero padding; ' ...
    'first layer uses existing empirical-Z marginal rescaling; selected support is structural.'];
end

function value = get_field(s, name, fallback)
if isfield(s,name), value = s.(name); else, value = fallback; end
end

function out = conditional_mean(value, mask)
if any(mask(:)), out = mean(value(mask)); else, out = NaN; end
end
