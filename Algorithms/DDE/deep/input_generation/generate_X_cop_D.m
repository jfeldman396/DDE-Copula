function [X,Z, A] = generate_X_cop_D(N, lambdas,prop_true, D, B_cell, gamma)
% Generate data from a Normal-D-latent-layer DDE
% @param N: sample size
% @param prop_true: length K{D} vector for the top-latent-layer proportions
% @param B_cell: length D cell conatining each layers' coefficients
% @param gamma: legnth J vector of the bottom layer variances
    p = size(B_cell{1},1);
    K_top = size(prop_true, 2);
    A = cell(D,1);

    % generate A{D}
    A_top = zeros(N, K_top);
    for k = 1:K_top
        A_top(:,k) = binornd(1, prop_true(k), N, 1);
    end
    A{D} = A_top;

    % recursively generate A{d}
    for d = (D-1):-1:1
        A{d} = binornd(1, logistic([ones(N,1), A{d+1}] * B_cell{d+1}'));
    end

    % generate the observed data X
    Z = normrnd([ones(N,1), A{1}] * B_cell{1}', repmat(sqrt(gamma'),N,1));


     Q = zeros(N,p);
    for j = 1:p
        r = tiedrank(Z(:,j));   % handles ties if they occur
        Q(:,j) = r / (N + 1);   % keeps values strictly inside (0,1)
    end

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

                X(:,j) = zeros(N,1);

                % indices that fall in the "Poisson region"
                idx_nonzero = qj >= pi0;

                % rescale only those uniforms to [0,1]
                u_adj = (qj(idx_nonzero) - pi0) ./ (1 - pi0);

                % transform them using Poisson inverse CDF
                X(idx_nonzero,j) = poissinv(u_adj, lambdas(j));

            case 2  % ---- Right-skewed ----
                shape = 2;
                scale = lambdas(j);
                X(:,j) = round(gaminv(Q(:,j), shape, scale));

            case 3  % ---- Standard Poisson ----
                X(:,j) = poissinv(Q(:,j), lambdas(j));
        end
end