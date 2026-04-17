function [X_sim, Z] = generate_X_Cop_pred(N, prop_true, B1, B2, gamma, X_data)
% GENERATE_X_COP_PRED Generate synthetic data from the fitted copula model.
%
% Inputs
%   N         : number of synthetic observations
%   prop_true : 1 x K2 or K2 x 1 vector of Bernoulli probs for A2
%   B1        : p x (K1+1) coefficient matrix for Z | A1
%   B2        : K1 x (K2+1) coefficient matrix for A1 | A2
%   gamma     : p x 1 or 1 x p vector of residual variances
%   X_data    : observed training data used for empirical marginals
%
% Outputs
%   X_sim     : N x p synthetic data matrix
%   Z         : N x p latent Gaussian matrix

    % number of features
    p = size(X_data, 2);
    prop_true = prop_true(:).';   % force row vector
    gamma = gamma(:).';           % force row vector
    K2 = numel(prop_true);

    % --- Step 1: Generate A2 (binary covariates) ---
    A2 = zeros(N, K2);
    for k = 1:K2
        A2(:, k) = binornd(1, prop_true(k), N, 1);
    end

    % --- Step 2: Generate A1 (binary covariate depending on A2) ---
    p_true = logistic([ones(N,1), A2] * B2');
    A1 = binornd(1, p_true);

    % --- Step 3: Latent Gaussian Z ---
    mu = [ones(N, 1), A1] * B1';        % means
    Z  = randn(size(mu)) .* sqrt(gamma) + mu;

    % --- Step 4: Convert latent Gaussian draws to uniforms via ranks ---
    Q = zeros(N, p);
    for j = 1:p
        Q(:, j) = tiedrank(Z(:, j)) / (N + 1);
    end

    % --- Step 5: Transform uniforms to empirical marginals from X_data ---
    X_sim = zeros(N, p);
    for j = 1:p
        x_emp = sort(X_data(:, j));
        n_emp = numel(x_emp);

        % empirical inverse CDF: map Q(:,j) in (0,1) to observed quantiles
        idx = max(1, ceil(Q(:, j) * n_emp));
        X_sim(:, j) = x_emp(idx);
    end
end

function y = logistic(x)
% LOGISTIC Logistic function
    y = 1 ./ (1 + exp(-x));
end