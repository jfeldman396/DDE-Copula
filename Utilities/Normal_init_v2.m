function [prop_ini, B1_ini, B2_ini, gamma_ini, A1_est, A2_est] = Normal_init(X, K1_max, K1, K2,epsilon)
% Algorithm 1 (spectral initialization) for Normal-two-latent-layer DDEs
% @X: N x J continuous data matrix
% @K1_max: upper bound for K1; set to K1 if known
% @K1, K2: true latent dimensions K1, K2 for simulations
% @B1, B2, G1, G2: true layer-wise coefficients and graphical matrices
% @epsilon: truncation threshold for the double-svd

    [N,J] = size(X);

    %% 1st layer
    % svd
    X_centered = X- ones(N,1) * mean(X);
    [~, S, V] = svd(X_centered, "econ");

    % varimax
    m = K1_max;
    V_top_m = V(:,1:m);

    [R_V, ~] = rotatefactors(V_top_m(:,1:K1_max), 'method', 'varimax', 'Maxit', 1000);
    B1_est = thres(R_V, 1/2.5/sqrt(J));

    % adjust sign flip
    Sign_flip_multiplier_B = [repmat(2*(mean(B1_est(:,1:end), 1) > 0)-1, J, 1)];
    B1_est = B1_est .* Sign_flip_multiplier_B;
  
    % estimate A1
    A1_est_nonbin = X_centered * B1_est * inv(B1_est' * B1_est);
    A1_est = double(A1_est_nonbin > 0);
    G1_est = (B1_est ~= 0);
    
    % re-estimate B1
    A1_centered = A1_est - ones(N,1) * mean(A1_est, 1);
    B1_re_est = (inv(A1_centered' * A1_centered) * A1_centered' * X_centered)';
    B1_re_est = thres(B1_re_est .* G1_est, 0);
    b1 = mean(X, 1)' - B1_re_est * mean(A1_est, 1)';
    
    B1_ini = [b1, B1_re_est];
    gamma_ini = mean((X - [ones(N,1), A1_est]*B1_ini').^2, 1);


    %% 2nd layer
    % first svd
    [U,S,V] = svd(A1_est, "econ");
    m = max(K2+1, sum(diag(S) > 1.01*sqrt(N)));
    A1_top_m = U(1:end, 1:m) * S(1:m, 1:m) * V(1:end, 1:m)';
    A1_trunc = max(min(A1_top_m, 1-epsilon), epsilon);
    
    % second svd
    A_inv = logit(A1_trunc);
    A_inv_adj = A_inv - ones(N,1)*mean(A_inv);
    [~, ~, V_adj] = svd(A_inv_adj, 'econ'); 

    % trivial case with only one latent variable
    if K2 == 1
        B2_est = V_adj(:,1);
        Sign_flip_multiplier_B = [repmat(2*(mean(B2_est(:,1:end), 1) > 0)-1, K1_max, 1)];
        B2_est = B2_est .* Sign_flip_multiplier_B;

        A2_est = A_inv_adj * B2_est * inv(B2_est' * B2_est);
        A2_est = double(A2_est > 0);
        A2_centered = A2_est - ones(N,1) * mean(A2_est, 1);
        B2_re_est = (inv(A2_centered' * A2_centered) * A2_centered' * A_inv_adj)'/2;

    else
        % varimax
        [R_V, ~] = rotatefactors(V_adj(:,1:K2), 'method', 'varimax', 'Maxit', 1000);
        B2_est = thres(R_V, 1/3.5/sqrt(K1_max));
        
        % adjust sign flip
        Sign_flip_multiplier_B = [repmat(2*(mean(B2_est(:,1:end), 1) > 0)-1, K1_max, 1)];
        B2_est = B2_est .* Sign_flip_multiplier_B;

        G2_est = double(B2_est ~=0);
        
        % estimate A2
        A2_est = A_inv_adj * B2_est * inv(B2_est' * B2_est);
        A2_est = double(A2_est > 0);
    
        % re-estimate B2
        A2_centered = A2_est - ones(N,1) * mean(A2_est, 1);
        B2_re_est = (inv(A2_centered' * A2_centered) * A2_centered' * A_inv_adj)'/2;
       
        for k = 1:K2
            B2_re_est(k,k) = max(B2_re_est(k,k), 2.5);
        end
    end
    b2 = mean(A_inv_adj, 1)' - B2_re_est * mean(A2_est, 1)';
    B2_ini = [b2, B2_re_est];
    prop_ini = mean(A2_est > 0);
end