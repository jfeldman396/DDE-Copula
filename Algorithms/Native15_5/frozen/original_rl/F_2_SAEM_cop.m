function [f] = F_2_SAEM_cop(A1_sample_k, A2_sample_long, N, K2, C, temp)
% Negative tempered Bernoulli log-likelihood
% Here temp in (0,1], and the tempered target is p(y|eta)^temp

    if nargin < 6 || isempty(temp)
        temp = 1;
    end

    f = @loglike;

    function ll = loglike(x)
        x = reshape(x, [1, K2+1, 1]);

        eta = reshape(sum(bsxfun(@times, A2_sample_long, x), 2), [N, C]);

        % stable log(1 + exp(eta))
        log1pexp_eta = log1p(exp(-abs(eta))) + max(eta, 0);

        nll = -(sum(A1_sample_k .* eta, 'all') - sum(log1pexp_eta, 'all')) / C;

        ll = temp * nll;
    end
end