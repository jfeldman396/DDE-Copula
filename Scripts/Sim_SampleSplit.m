clear;
addpath('/Users/jfeldm01/Library/CloudStorage/OneDrive-Kearney/Documents/Deep-Discrete-Encoders-main/CopDDE/Utilities/')
addpath('/Users/jfeldm01/Library/CloudStorage/OneDrive-Kearney/Documents/Deep-Discrete-Encoders-main/CopDDE//Algorithms/')
addpath('/Users/jfeldm01/Library/CloudStorage/OneDrive-Kearney/Documents/Deep-Discrete-Encoders-main/utilities/')
addpath('/Users/jfeldm01/Library/CloudStorage/OneDrive-Kearney/Documents/Deep-Discrete-Encoders-main/main algorithms/Poisson/')

%%% add paths on your own

K2_true = 3;
K1_true = 10;

J = 100; 

%% ------------------------------------------------------------
% Construct true loading matrices
% -------------------------------------------------------------
B2_sub = zeros(K1_true, K2_true);

max_val = 4;
idx = floor(linspace(1, K1_true, 4));
for i = 1:(length(idx) - 1)
    if mod(i,2) == 0
        B2_sub(idx(i):idx(i+1), i)   = max_val;
        B2_sub(idx(i):idx(i+1), i-1) = -max_val/3;
    else
        B2_sub(idx(i):idx(i+1), i) = max_val;
        if i < length(idx) - 2
            B2_sub(idx(i):idx(i+1), i+1) = -max_val/3;
        end
    end
end

B1_sub = sim_block_loadings(J, K1_true,10,10,5,0,1); % change first 5 to 10 when J = 100

K1_max = floor(J/3);
K2_max = floor(K1_max/3);

B1_true_aug_unscale = zeros(J, K1_max);
B2_true_aug = zeros(K1_max, K2_max);

B1_true_aug_unscale(:,1:K1_true)= B1_sub;
B2_true_aug(1:K1_true,1:K2_true) = B2_sub;

B1_true_aug_unscale = [[-2*ones(floor(J/3),1); -4*ones(floor(J/3),1); -2*ones(J - 2*floor(J/3),1)], B1_true_aug_unscale];
B2_true_aug = [[-2*ones(floor(K1_true/3),1); -ones(floor(K1_true/3),1); -2*ones(K1_true- 2*floor(K1_true/3),1); zeros(K1_max - K1_true,1)], B2_true_aug];

gamma = ones(J,1);

prop_true = 0.5*ones(K2_true,1);

[B1_true_scale, gamma_scale,~] = rescale_B1(prop_true, B2_true_aug(1:K1_true,1:K2_true + 1), B1_true_aug_unscale(:, 1:K1_true + 1), gamma, false,prop_true);
B1_true_aug_scale = zeros(J, K1_max+1);
B1_true_aug_scale(:,1:K1_true+1)= B1_true_scale;

B2_true = B2_true_aug(1:K1_true, 1:K2_true + 1);
lambdas = randi([1, 10], 1, J);
%% main simulation
C = 100; n_vec = [6000,8000,10000,12000]; 


B1_final_CSP = zeros(J, K1_max+1, length(n_vec), C); B2_final_CSP = zeros(K1_max, K2_max + 1,length(n_vec), C);

time_CSP = zeros(length(n_vec), C);

num_act1_CSP = zeros(length(n_vec), C); num_act2_CSP = zeros(length(n_vec), C);

results = cell(length(n_vec), C);
for aa = 1:4
    N = n_vec(aa);
    epsilon = 0.0001;
    tau = max(.3,3*N^(-0.3));
    
    
    for c= 1:C
        warning('off', 'stats:kmeans:FailedToConverge');
        rng(50+c);
        tt = tic;
        [Z_true,X,~] = generate_X_Cop_marg_emp(N,prop_true.',...
            lambdas,gamma_scale, B1_true_scale, B2_true_aug(1:K1_true,1:K2_true + 1));

        idx = randperm(N(1));
        split_1 = idx(1:4000);
        split_2 = idx(4000 + 1:end);

        X1 = X(split_1,:);
        X2 = X(split_2,:);

        R1 = NaN(length(split_1), J); R2 = NaN(length(split_2), J);

        for j = 1:J % do this for both splits
            yj1 = X1(:, j);
            isn = isnan(yj1);

            yj1_nonan = yj1(~isn);
            [~, ~, ic1] = unique(yj1_nonan, 'sorted');

            R1(~isn, j) = ic1;

            yj2 = X2(:, j);
            isn = isnan(yj2);

            yj2_nonan = yj2(~isn);
            [~, ~, ic2] = unique(yj2_nonan, 'sorted');

            R2(~isn, j) = ic2;
        end

        %% ----------------------------------------------------
        % 3) Number of rank levels per variable
        % -----------------------------------------------------
        Rlevels1 = zeros(1, J);Rlevels2 = zeros(1, J);
        for j = 1:J
            Rlevels1(j) = max(R1(:, j), [], 'omitnan');
            Rlevels2(j) = max(R2(:, j), [], 'omitnan');
            if isempty(Rlevels1(j)) || isnan(Rlevels1(j))
                Rlevels1(j) = 0;
            end
            if isempty(Rlevels2(j)) || isnan(Rlevels2(j))
                Rlevels2(j) = 0;
            end
        end

        Z_init1 = simulate_gaussian_mixture(X1,J);
        tic;
        % spectral initialization
        [prop_in, B1_in, B2_in, gamma_in, A1_in, A2_in] = Normal_init_v2(Z_init1, K1_max, K1_max, K2_max,epsilon);
        

        %% CSP
    
        [num_act1_CSP(aa,c),num_act2_CSP(aa,c), theta, theta2,gamma_CSP,p,q,A1_sample_long, A2_sample_long,Z,prop_CSP, B1_CSP, B2_CSP, A1_new, A2_new, it_num] = get_SAEM_RL_CSP(X1, Z_init1, R1, Rlevels1, prop_in, gamma_in,B1_in, B2_in, A1_in, A2_in, 1, 50,.05,.1,.7,tau); % change to .05 and .1 for J = 100
        
        [num_act1_CSP(aa,c),num_act2_CSP(aa,c)]
        [B1_final_CSP(:,:,aa,c),B2_final_CSP(:,:,aa,c),~] = format_estimates(B1_true_aug_unscale, B2_true_aug, B1_CSP, B2_CSP,gamma_CSP,prop_CSP,true,Z,true);

        %% now esitmate graphs
        n1 = num_act1_CSP(aa,c); n2 = num_act2_CSP(aa,c);

        B1_tmp = B1_final_CSP(:,:,aa,c); B2_tmp = B2_final_CSP(:,:,aa,c);

        B1_tmp =B1_tmp(:,1:n1+1); B2_tmp = B2_tmp(1:n1, 1:n2+1);

        G1 = B1_tmp ~=0; G2 = B2_tmp ~=0;

        G2

        %% initialize Z
        Z_init2 = simulate_gaussian_mixture(X2,J);

        %% and theta
        [prop_in, B1_in, B2_in, gamma_in, A1_in, A2_in] = Normal_init_G(Z_init2, n1, n1, n2, B1_tmp, B2_tmp, G1, G2, epsilon);
        tol = n2/2;
        %% fit unpenalized model
        [gamma,p,q,A1_sample_long, A2_sample_long,Z,prop, B1, B2, A1_new, A2_new, t] = get_SAEM_RL_G(X2, Z_init2, R2, Rlevels2, prop_in, B1_in, B2_in, gamma_in, A1_in, A2_in, C, tol, 50, 1, G1, G2);

        
        results{aa,c}.B1 = B1; 
        results{aa,c}.B2 = B2;
        results{aa,c}.G1 = G1;
        results{aa,c}.G2 = G2;
        results{aa,c}.time = toc(tt);
        results{aa,c}.time
    end
end


