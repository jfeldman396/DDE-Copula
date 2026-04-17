function [Z,X,A1,A2,p_true] = generate_X_Cop_marg(N, prop_true, lambdas, B1, B2)

    K2 = size(prop_true, 2);

    % Generate A2 (latent binary)
    A2 = zeros(N, K2);
    for k = 1:K2
        A2(:,k) = binornd(1, prop_true(k), N, 1);
    end

    % Generate A1 (dependent binary)
    p_true = logistic([ones(N,1), A2] * B2');
    A1 = binornd(1, p_true);

    % Latent Gaussian layer
    mu = [ones(N, 1), A1] * B1';
    Z = randn(size(mu)) + mu;
    p = size(Z,2);

    % Convert latent Z to uniforms
    Q = marginal_cdf_Z(Z, prop_true, B2, B1, ones(p,1));

    % Allocate X
    X = zeros(N,p);

    % Cycle through marginals:
    %   1: zero-inflated
    %   2: right-skewed (Gamma)
    %   3: Poisson
    for j = 1:p
        modtype = mod(j-1, 3) + 1;

        switch modtype
            case 1  % ---- Zero-inflated ----
               pi0 = 0.3; % probability of extra zeros
                qj = Q(:,j);
            
                X(:,j) = zeros(N,1);  % pre-allocate
            
                % indices that fall in the "Poisson region"
                idx_nonzero = qj >= pi0;
            
                % rescale only those uniforms to [0,1]
                u_adj = (qj(idx_nonzero) - pi0) ./ (1 - pi0);
            
                % transform them using Poisson inverse CDF
                X(idx_nonzero,j) = poissinv(u_adj, lambdas(j));
            case 2  % ---- Right-skewed ----
                shape = 2; scale = lambdas(j);  % mean ~ lambda
                X(:,j) = round(gaminv(Q(:,j), shape, scale));

            case 3  % ---- Standard Poisson ----
                X(:,j) = poissinv(Q(:,j), lambdas(j));
        end
    end
end