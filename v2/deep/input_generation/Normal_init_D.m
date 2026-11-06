function [prop_ini, B_cell_ini, gamma_ini, A_cell_ini] = Normal_init_D(X, D, K_cell,  epsilon)

    [N,J] = size(X);
    A_cell_ini = cell(D,1); B_cell_ini = cell(D,1);

    %% 1st layer
    % svd
    K1 = K_cell{1};  K1_max = K1;
    X_centered = X- ones(N,1) * mean(X);
    [~, ~, V] = svd(X_centered, "econ");

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

    A_cell_ini{1} = A1_est; B_cell_ini{1} = [b1, B1_re_est];
    gamma_ini = mean((X - [ones(N,1), A1_est]*B_cell_ini{1}').^2, 1);

    %% d-th layer
    for d =2:D
        [A_cell_ini{d}, B_cell_ini{d}] = binary_init(N, K_cell{d-1}, K_cell{d}, A_cell_ini{d-1}, 2, epsilon);
    end

    prop_ini = mean(A_cell_ini{D} > 0);
end