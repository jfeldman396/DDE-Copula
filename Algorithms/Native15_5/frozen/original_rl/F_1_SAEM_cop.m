function [f] = F_1_SAEM_cop(Xj, A1_sample_long, N, K1, C, gamma, temp)
    
    % Negative tempered Gaussian log-likelihood
    % Here temp in (0,1], and the tempered target is p(y|eta)^temp

    if nargin < 7 || isempty(temp)
        temp = 1;
    end

    f = @loglike;

    function ll = loglike(x)
        x = reshape(x, [1, K1+1, 1]);

        eta = reshape(sum(bsxfun(@times, A1_sample_long, x), 2), [N, C]);

        nll = sum((Xj - eta).^2, 'all') / (2 * gamma * C);

        ll = temp * nll;
    end
end
