addpath('/Users/jfeldm01/Library/CloudStorage/OneDrive-Kearney/Documents/Deep-Discrete-Encoders-main/CopDDE/Utilities/')
addpath('/Users/jfeldm01/Library/CloudStorage/OneDrive-Kearney/Documents/Deep-Discrete-Encoders-main/CopDDE//Algorithms/')
addpath('/Users/jfeldm01/Library/CloudStorage/OneDrive-Kearney/Documents/Deep-Discrete-Encoders-main/utilities/')
addpath('/Users/jfeldm01/Library/CloudStorage/OneDrive-Kearney/Documents/Deep-Discrete-Encoders-main/main algorithms/Poisson/')

%%% add paths on your own

K2_true = 3;
K1_true = 10;

J = 150; 

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

B1_sub = sim_block_loadings(J, K1_true,10,10,5,0,1); % first entry by J: J = 50 -> 5, J = 100 -> 10, J = 150 -> 15

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
C = 100; n_vec = [500 1000 2000 4000 8000,16000]; n_parallel = 6;


B1_final_CSP = zeros(J, K1_max+1, length(n_vec), C); B2_final_CSP = zeros(K1_max, K2_max + 1,length(n_vec), C);
B1_final_noCSP = zeros(J, K1_max+1, length(n_vec), C); B2_final_noCSP = zeros(K1_max, K2_max + 1,length(n_vec), C);
B1_final_Gibbs = zeros(J, K1_max+1, length(n_vec), C); B2_final_Gibbs = zeros(K1_max, K2_max + 1,length(n_vec), C);
B1_final_pois= zeros(J, K1_max+1, length(n_vec), C); B2_final_pois = zeros(K1_max, K2_max + 1,length(n_vec), C);

time_CSP = zeros(length(n_vec), C);
time_noCSP = zeros(length(n_vec), C);
time_Gibbs = zeros(length(n_vec), C);
time_pois = zeros(length(n_vec), C);

num_act1_CSP = zeros(length(n_vec), C); num_act2_CSP = zeros(length(n_vec), C);
num_act1_Gibbs = zeros(length(n_vec), C); num_act2_Gibbs = zeros(length(n_vec), C);

for aa = 1:6
    N = n_vec(aa);
    tol = ceil(K2_max/2); epsilon = 0.0001;
    pen_1 = N^(2/8); pen_2 = N^(2/8); tau = max(.3,3*N^(-0.3));
    
    
    parfor(c= 1:C)
        rng(50+c);
        
        [Z_true,X,~] = generate_X_Cop_marg_emp(N,prop_true.',...
            lambdas,gamma_scale, B1_true_scale, B2_true_aug(1:K1_true,1:K2_true + 1));

         R = NaN(N, J);

        for j = 1:J
            yj = X(:, j);
            isn = isnan(yj);

            yj_nonan = yj(~isn);
            [~, ~, ic] = unique(yj_nonan, 'sorted');

            R(~isn, j) = ic;
        end

        %% ----------------------------------------------------
        % 3) Number of rank levels per variable
        % -----------------------------------------------------
        Rlevels = zeros(1, J);
        for j = 1:J
            Rlevels(j) = max(R(:, j), [], 'omitnan');
            if isempty(Rlevels(j)) || isnan(Rlevels(j))
                Rlevels(j) = 0;
            end
        end

        Z_init = simulate_gaussian_mixture(X,J);
        tic;
        % spectral initialization
        [prop_in, B1_in, B2_in, gamma_in, A1_in, A2_in] = Normal_init_v2(Z_init, K1_max, K1_max, K2_max,epsilon);
        

        %% CSP
        t = tic;
        [num_act1_CSP(aa,c),num_act2_CSP(aa,c), theta, theta2,gamma_CSP,p,q,A1_sample_long, A2_sample_long,Z,prop_CSP, B1_CSP, B2_CSP, A1_new, A2_new, it_num] = get_SAEM_RL_CSP(X, Z_init, R, Rlevels, prop_in, gamma_in,B1_in, B2_in, A1_in, A2_in, 1, 50,.05,.1,.7,tau); % t1 = .02,.05,.05, t2 = .04,.1,.1, J =50,100,150 
        time_CSP(aa,c) = toc(t);
        
        [num_act1_CSP(aa,c),num_act2_CSP(aa,c)]
        [B1_final_CSP(:,:,aa,c),B2_final_CSP(:,:,aa,c),~] = format_estimates(B1_true_aug_unscale, B2_true_aug, B1_CSP, B2_CSP,gamma_CSP,prop_CSP,true,Z,true);
    
        %% CSP with VI
        t = tic;
        [num_act1_Gibbs(aa,c),num_act2_Gibbs(aa,c), theta, theta2,gamma_Gibbs,p,q,A1_sample_long, A2_sample_long,Z,prop_Gibbs, B1_Gibbs, B2_Gibbs, A1_new, A2_new, it_num] = get_SAEM_RL_CSP_Gibbs(X, Z_init, R, Rlevels, prop_in, gamma_in,B1_in, B2_in, A1_in, A2_in, 1, 50,.05,.1,.7,tau);
        time_Gibbs(aa,c) = toc(t);

        [B1_final_Gibbs(:,:,aa,c),B2_final_Gibbs(:,:,aa,c),~] = format_estimates(B1_true_aug_unscale, B2_true_aug, B1_Gibbs, B2_Gibbs,gamma_Gibbs,prop_Gibbs,true,Z,true);
        
        %% no CSP -- MLE
        t = tic;
        [gamma_noCSP,p,q,A1_sample_long, A2_sample_long,Z,prop_noCSP, B1_noCSP, B2_noCSP, A1_new, A2_new, it_noCSP] = ...
            get_SAEM_RL(X, Z_init, R, Rlevels, prop_in, B1_in, B2_in, gamma_in, pen_1, pen_2, tau, A1_in, A2_in, 1, tol, 50,.7);
        time_noCSP(aa,c) = toc(t);
        
        [B1_final_noCSP(:,:,aa,c),B2_final_noCSP(:,:,aa,c),~] = format_estimates(B1_true_aug_unscale, B2_true_aug, B1_noCSP, B2_noCSP,gamma_noCSP,prop_noCSP,true,Z,true);

        %% Poisson DDE
        t= tic;
        [prop_pois, B1_pois, B2_pois, A1_final, A2_final, it_pois] = get_SAEM_poisson( ...
             X, prop_in, B1_in, B2_in, pen_1, ...
             pen_2, tau, A1_in, A2_in, 1, tol);

        [B1_final_pois(:,:,aa,c),B2_final_pois(:,:,aa,c),~] = format_estimates(B1_true_aug_unscale, B2_true_aug, B1_pois, B2_pois,gamma_noCSP,prop_pois,false,Z,true);

        time_pois(aa,c) = toc(t);


        fprintf('%d-th iteration complete \n', c);
    end    

end

% save("J50_overspecified")

%-------------------------------------------

% Evaluations
G1_true_aug = B1_true_aug_scale ~= 0;
G2_true_aug = B2_true_aug ~= 0;

MSE_B1s_CSP    = [];
MSE_B1s_Gibbs  = [];
MSE_B1s_noCSP  = [];
MSE_B1s_pois   = [];

MSE_B2s_CSP    = [];
MSE_B2s_Gibbs  = [];
MSE_B2s_noCSP  = [];
MSE_B2s_pois   = [];

acc_G1s_CSP    = [];
acc_G1s_Gibbs  = [];
acc_G1s_noCSP  = [];
acc_G1s_pois   = [];

acc_G2s_CSP    = [];
acc_G2s_Gibbs  = [];
acc_G2s_noCSP  = [];
acc_G2s_pois   = [];

n_aa = size(B1_final_CSP, 3);
n_rep = size(B1_final_CSP, 4);

for aa = 1:n_aa

    MSE_B1_CSP    = [];
    MSE_B1_Gibbs  = [];
    MSE_B1_noCSP  = [];
    MSE_B1_pois   = [];

    MSE_B2_CSP    = [];
    MSE_B2_Gibbs  = [];
    MSE_B2_noCSP  = [];
    MSE_B2_pois   = [];

    acc_G1_CSP    = [];
    acc_G1_Gibbs  = [];
    acc_G1_noCSP  = [];
    acc_G1_pois   = [];

    acc_G2_CSP    = [];
    acc_G2_Gibbs  = [];
    acc_G2_noCSP  = [];
    acc_G2_pois   = [];

    for c = 1:n_rep

        % -----------------------------
        % CSP
        % -----------------------------
        B1_i_CSP = B1_final_CSP(:,:,aa,c);
        G1_i_CSP = B1_i_CSP ~= 0;
        B2_i_CSP = B2_final_CSP(:,:,aa,c);
        G2_i_CSP = B2_i_CSP ~= 0;

        MSE_B1_CSP = [MSE_B1_CSP, mean((B1_i_CSP - B1_true_aug_scale).^2, "all")];
        MSE_B2_CSP = [MSE_B2_CSP, mean((B2_i_CSP - B2_true_aug).^2, "all")];
        acc_G1_CSP = [acc_G1_CSP, mean(G1_true_aug == G1_i_CSP, "all")];
        acc_G2_CSP = [acc_G2_CSP, mean(G2_true_aug == G2_i_CSP, "all")];

        % -----------------------------
        % Gibbs
        % -----------------------------
        B1_i_Gibbs = B1_final_Gibbs(:,:,aa,c);
        G1_i_Gibbs = B1_i_Gibbs ~= 0;
        B2_i_Gibbs = B2_final_Gibbs(:,:,aa,c);
        G2_i_Gibbs = B2_i_Gibbs ~= 0;

        MSE_B1_Gibbs = [MSE_B1_Gibbs, mean((B1_i_Gibbs - B1_true_aug_scale).^2, "all")];
        MSE_B2_Gibbs = [MSE_B2_Gibbs, mean((B2_i_Gibbs - B2_true_aug).^2, "all")];
        acc_G1_Gibbs = [acc_G1_Gibbs, mean(G1_true_aug == G1_i_Gibbs, "all")];
        acc_G2_Gibbs = [acc_G2_Gibbs, mean(G2_true_aug == G2_i_Gibbs, "all")];

        % -----------------------------
        % noCSP
        % -----------------------------
        B1_i_noCSP = B1_final_noCSP(:,:,aa,c);
        G1_i_noCSP = B1_i_noCSP ~= 0;
        B2_i_noCSP = B2_final_noCSP(:,:,aa,c);
        G2_i_noCSP = B2_i_noCSP ~= 0;

        MSE_B1_noCSP = [MSE_B1_noCSP, mean((B1_i_noCSP - B1_true_aug_scale).^2, "all")];
        MSE_B2_noCSP = [MSE_B2_noCSP, mean((B2_i_noCSP - B2_true_aug).^2, "all")];
        acc_G1_noCSP = [acc_G1_noCSP, mean(G1_true_aug == G1_i_noCSP, "all")];
        acc_G2_noCSP = [acc_G2_noCSP, mean(G2_true_aug == G2_i_noCSP, "all")];

        % -----------------------------
        % Poisson
        % -----------------------------
        B1_i_pois = B1_final_pois(:,:,aa,c);
        G1_i_pois = B1_i_pois ~= 0;
        B2_i_pois = B2_final_pois(:,:,aa,c);
        G2_i_pois = B2_i_pois ~= 0;

        MSE_B1_pois = [MSE_B1_pois, mean((B1_i_pois - B1_true_aug_scale).^2, "all")];
        MSE_B2_pois = [MSE_B2_pois, mean((B2_i_pois - B2_true_aug).^2, "all")];
        acc_G1_pois = [acc_G1_pois, mean(G1_true_aug == G1_i_pois, "all")];
        acc_G2_pois = [acc_G2_pois, mean(G2_true_aug == G2_i_pois, "all")];

    end

    MSE_B1s_CSP   = [MSE_B1s_CSP;   MSE_B1_CSP];
    MSE_B1s_Gibbs = [MSE_B1s_Gibbs; MSE_B1_Gibbs];
    MSE_B1s_noCSP = [MSE_B1s_noCSP; MSE_B1_noCSP];
    MSE_B1s_pois  = [MSE_B1s_pois;  MSE_B1_pois];

    MSE_B2s_CSP   = [MSE_B2s_CSP;   MSE_B2_CSP];
    MSE_B2s_Gibbs = [MSE_B2s_Gibbs; MSE_B2_Gibbs];
    MSE_B2s_noCSP = [MSE_B2s_noCSP; MSE_B2_noCSP];
    MSE_B2s_pois  = [MSE_B2s_pois;  MSE_B2_pois];

    acc_G1s_CSP   = [acc_G1s_CSP;   acc_G1_CSP];
    acc_G1s_Gibbs = [acc_G1s_Gibbs; acc_G1_Gibbs];
    acc_G1s_noCSP = [acc_G1s_noCSP; acc_G1_noCSP];
    acc_G1s_pois  = [acc_G1s_pois;  acc_G1_pois];

    acc_G2s_CSP   = [acc_G2s_CSP;   acc_G2_CSP];
    acc_G2s_Gibbs = [acc_G2s_Gibbs; acc_G2_Gibbs];
    acc_G2s_noCSP = [acc_G2s_noCSP; acc_G2_noCSP];
    acc_G2s_pois  = [acc_G2s_pois;  acc_G2_pois];

end


%------------------------------------------------------
% Error Evaluations on active columns


B1_true_sub = B1_true_aug_scale(:, 1:K1_true + 1);
B2_true_sub = B2_true_aug(1:K1_true, 1:K2_true + 1);

G1_true_sub = B1_true_sub ~= 0;
G2_true_sub = B2_true_sub ~= 0;

MSE_B1s_CSP    = [];
MSE_B1s_Gibbs  = [];
MSE_B1s_noCSP  = [];
MSE_B1s_pois   = [];

MSE_B2s_CSP    = [];
MSE_B2s_Gibbs  = [];
MSE_B2s_noCSP  = [];
MSE_B2s_pois   = [];

acc_G1s_CSP    = [];
acc_G1s_Gibbs  = [];
acc_G1s_noCSP  = [];
acc_G1s_pois   = [];

acc_G2s_CSP    = [];
acc_G2s_Gibbs  = [];
acc_G2s_noCSP  = [];
acc_G2s_pois   = [];

n_aa  = size(B1_final_CSP, 3);
n_rep = size(B1_final_CSP, 4);

for aa = 1:n_aa

    MSE_B1_CSP    = [];
    MSE_B1_Gibbs  = [];
    MSE_B1_noCSP  = [];
    MSE_B1_pois   = [];

    MSE_B2_CSP    = [];
    MSE_B2_Gibbs  = [];
    MSE_B2_noCSP  = [];
    MSE_B2_pois   = [];

    acc_G1_CSP    = [];
    acc_G1_Gibbs  = [];
    acc_G1_noCSP  = [];
    acc_G1_pois   = [];

    acc_G2_CSP    = [];
    acc_G2_Gibbs  = [];
    acc_G2_noCSP  = [];
    acc_G2_pois   = [];

    for c = 1:n_rep

        % -----------------------------
        % CSP
        % -----------------------------
        B1_i_CSP = B1_final_CSP(:, 1:K1_true+1, aa, c);
        B2_i_CSP = B2_final_CSP(1:K1_true, 1:K2_true+1, aa, c);

        G1_i_CSP = B1_i_CSP ~= 0;
        G2_i_CSP = B2_i_CSP ~= 0;

        MSE_B1_CSP = [MSE_B1_CSP, mean((B1_i_CSP - B1_true_sub).^2, "all")];
        MSE_B2_CSP = [MSE_B2_CSP, mean((B2_i_CSP - B2_true_sub).^2, "all")];
        acc_G1_CSP = [acc_G1_CSP, mean(G1_true_sub == G1_i_CSP, "all")];
        acc_G2_CSP = [acc_G2_CSP, mean(G2_true_sub == G2_i_CSP, "all")];

        % -----------------------------
        % Gibbs
        % -----------------------------
        B1_i_Gibbs = B1_final_Gibbs(:, 1:K1_true+1, aa, c);
        B2_i_Gibbs = B2_final_Gibbs(1:K1_true, 1:K2_true+1, aa, c);

        G1_i_Gibbs = B1_i_Gibbs ~= 0;
        G2_i_Gibbs = B2_i_Gibbs ~= 0;

        MSE_B1_Gibbs = [MSE_B1_Gibbs, mean((B1_i_Gibbs - B1_true_sub).^2, "all")];
        MSE_B2_Gibbs = [MSE_B2_Gibbs, mean((B2_i_Gibbs - B2_true_sub).^2, "all")];
        acc_G1_Gibbs = [acc_G1_Gibbs, mean(G1_true_sub == G1_i_Gibbs, "all")];
        acc_G2_Gibbs = [acc_G2_Gibbs, mean(G2_true_sub == G2_i_Gibbs, "all")];

        % -----------------------------
        % noCSP
        % -----------------------------
        B1_i_noCSP = B1_final_noCSP(:, 1:K1_true+1, aa, c);
        B2_i_noCSP = B2_final_noCSP(1:K1_true, 1:K2_true+1, aa, c);

        G1_i_noCSP = B1_i_noCSP ~= 0;
        G2_i_noCSP = B2_i_noCSP ~= 0;

        MSE_B1_noCSP = [MSE_B1_noCSP, mean((B1_i_noCSP - B1_true_sub).^2, "all")];
        MSE_B2_noCSP = [MSE_B2_noCSP, mean((B2_i_noCSP - B2_true_sub).^2, "all")];
        acc_G1_noCSP = [acc_G1_noCSP, mean(G1_true_sub == G1_i_noCSP, "all")];
        acc_G2_noCSP = [acc_G2_noCSP, mean(G2_true_sub == G2_i_noCSP, "all")];

        % -----------------------------
        % Poisson
        % -----------------------------
        B1_i_pois = B1_final_pois(:, 1:K1_true+1, aa, c);
        B2_i_pois = B2_final_pois(1:K1_true, 1:K2_true+1, aa, c);

        G1_i_pois = B1_i_pois ~= 0;
        G2_i_pois = B2_i_pois ~= 0;

        MSE_B1_pois = [MSE_B1_pois, mean((B1_i_pois - B1_true_sub).^2, "all")];
        MSE_B2_pois = [MSE_B2_pois, mean((B2_i_pois - B2_true_sub).^2, "all")];
        acc_G1_pois = [acc_G1_pois, mean(G1_true_sub == G1_i_pois, "all")];
        acc_G2_pois = [acc_G2_pois, mean(G2_true_sub == G2_i_pois, "all")];

    end

    MSE_B1s_CSP   = [MSE_B1s_CSP;   MSE_B1_CSP];
    MSE_B1s_Gibbs = [MSE_B1s_Gibbs; MSE_B1_Gibbs];
    MSE_B1s_noCSP = [MSE_B1s_noCSP; MSE_B1_noCSP];
    MSE_B1s_pois  = [MSE_B1s_pois;  MSE_B1_pois];

    MSE_B2s_CSP   = [MSE_B2s_CSP;   MSE_B2_CSP];
    MSE_B2s_Gibbs = [MSE_B2s_Gibbs; MSE_B2_Gibbs];
    MSE_B2s_noCSP = [MSE_B2s_noCSP; MSE_B2_noCSP];
    MSE_B2s_pois  = [MSE_B2s_pois;  MSE_B2_pois];

    acc_G1s_CSP   = [acc_G1s_CSP;   acc_G1_CSP];
    acc_G1s_Gibbs = [acc_G1s_Gibbs; acc_G1_Gibbs];
    acc_G1s_noCSP = [acc_G1s_noCSP; acc_G1_noCSP];
    acc_G1s_pois  = [acc_G1s_pois;  acc_G1_pois];

    acc_G2s_CSP   = [acc_G2s_CSP;   acc_G2_CSP];
    acc_G2s_Gibbs = [acc_G2s_Gibbs; acc_G2_Gibbs];
    acc_G2s_noCSP = [acc_G2s_noCSP; acc_G2_noCSP];
    acc_G2s_pois  = [acc_G2s_pois;  acc_G2_pois];

end

%----------------------------------------------
%plot

methods = {'RL CSP','RL Gibbs','RL SAEM','Poisson DDE'};
n_methods = length(methods);
n_aa = size(B1_final_CSP,3);

B1_true_full = B1_true_aug_scale;

B1_avg = cell(n_methods, n_aa);
for aa = 1:n_aa
    B1_avg{1,aa} = mean(B1_final_CSP(:,:,aa,:), 4);
    B1_avg{2,aa} = mean(B1_final_Gibbs(:,:,aa,:), 4);
    B1_avg{3,aa} = mean(B1_final_noCSP(:,:,aa,:), 4);
    B1_avg{4,aa} = mean(B1_final_pois(:,:,aa,:), 4);
end

% color scale
all_vals = B1_true_full(:);
for m = 1:n_methods
    for aa = 1:n_aa
        all_vals = [all_vals; B1_avg{m,aa}(:)];
    end
end
clim = max(abs(all_vals));

figure;
tiledlayout(n_methods, n_aa+1, 'Padding','compact','TileSpacing','compact');

for m = 1:n_methods
    for aa = 1:n_aa
        nexttile
        imagesc(B1_avg{m,aa});
        caxis([-clim clim]);
        axis tight

        title(sprintf('n=%d', n_vec(aa)), 'FontSize', 20)

        if aa==1
            ylabel(methods{m}, 'FontSize', 20)
        end

        % axis labels (only bottom row to avoid clutter)
        if m == n_methods
            xlabel('K_1', 'FontSize', 20)
        end
        if aa == 1
            ylabel({methods{m}; 'J'}, 'FontSize', 20)
        end
    end

    % truth column
    nexttile
    imagesc(B1_true_full);
    caxis([-clim clim]);
    axis tight
    title('Truth', 'FontSize', 20)

    if m == n_methods
        xlabel('K_1', 'FontSize', 20)
    end
    if m == 1
        ylabel('J', 'FontSize', 20)
    end
end


cb = colorbar;
cb.FontSize = 20;



B2_true_full = B2_true_aug;

B2_avg = cell(n_methods, n_aa);
for aa = 1:n_aa
    B2_avg{1,aa} = mean(B2_final_CSP(:,:,aa,:), 4);
    B2_avg{2,aa} = mean(B2_final_Gibbs(:,:,aa,:), 4);
    B2_avg{3,aa} = mean(B2_final_noCSP(:,:,aa,:), 4);
    B2_avg{4,aa} = mean(B2_final_pois(:,:,aa,:), 4);
end

% color scale
all_vals = B2_true_full(:);
for m = 1:n_methods
    for aa = 1:n_aa
        all_vals = [all_vals; B2_avg{m,aa}(:)];
    end
end
clim = max(abs(all_vals));

figure;
tiledlayout(n_methods, n_aa+1, 'Padding','compact','TileSpacing','compact');

for m = 1:n_methods
    for aa = 1:n_aa
        nexttile
        imagesc(B2_avg{m,aa});
        caxis([-clim clim]);
        axis tight

        title(sprintf('n=%d', n_vec(aa)), 'FontSize', 20)

        if aa==1
            ylabel(methods{m}, 'FontSize', 20)
        end

        % axis labels (bottom row only)
        if m == n_methods
            xlabel('K_2', 'FontSize', 20)
        end
        if aa == 1
            ylabel({methods{m}; 'K_1'}, 'FontSize', 20)
        end
    end

    % truth column
    nexttile
    imagesc(B2_true_full);
    caxis([-clim clim]);
    axis tight
    title('Truth', 'FontSize', 20)

    if m == n_methods
        xlabel('K_2', 'FontSize', 20)
    end
    if m == 1
        ylabel('K_1', 'FontSize', 20)
    end
end


cb = colorbar;
cb.FontSize = 20;



% ================================
% Entry-wise MSE line plot (with jitter)
% ================================

% Sample sizes
n = [500 1000 2000 4000 8000 16000];

% MSE values
MSE_B1 = [mean(MSE_B1s_CSP,2), mean(MSE_B1s_Gibbs,2), ...
          mean(MSE_B1s_noCSP,2), mean(MSE_B1s_pois,2)];

% Extract methods
CSP   = MSE_B1(:,1);
Gibbs = MSE_B1(:,2);
SAEM  = MSE_B1(:,3);
Poiss = MSE_B1(:,4);

% --- JITTER (multiplicative for log scale) ---
jitter = [0.97, 0.99, 1.01, 1.03];  % small offsets

n_CSP   = n * jitter(1);
n_Gibbs = n * jitter(2);
n_SAEM  = n * jitter(3);
n_Poiss = n * jitter(4);

% Plot
figure;
hold on;

plot(n_CSP,   CSP,   '-o', 'LineWidth', 2.5, 'MarkerSize', 15);
plot(n_Gibbs, Gibbs, '-s', 'LineWidth', 2.5, 'MarkerSize', 15);
plot(n_SAEM,  SAEM,  '-d', 'LineWidth', 2.5, 'MarkerSize', 15);
plot(n_Poiss, Poiss, '-^', 'LineWidth', 2.5, 'MarkerSize', 15);

xlabel('Sample Size (n)', 'FontSize', 40);
ylabel('Average Entry-wise MSE', 'FontSize',40);
title('J = 100: B^{(1)} Entry-wise MSE vs Sample Size', 'FontSize',40);

% Log scale with clean ticks
set(gca, 'XScale', 'log');
xticks(n);
xticklabels(string(n));

set(gca, 'FontSize', 40);

legend({'RL CSP','RL Gibbs','RL SAEM','Poisson DDE'}, ...
       'Location','east', 'FontSize', 20);

grid on;
box on;
axis tight;

%%%% B2
% Sample sizes

% MSE values (rows = n, cols = methods)
MSE_B2 = [mean(MSE_B2s_CSP,2), mean(MSE_B2s_Gibbs,2),mean(MSE_B2s_noCSP,2), mean(MSE_B2s_pois,2)];

% Extract methods
CSP   = MSE(:,1);
Gibbs = MSE(:,2);
SAEM  = MSE(:,3);   % noCSP
Poiss = MSE(:,4);

% Plot
figure;
hold on;

plot(n, CSP,   '-o', 'LineWidth', 2.5, 'MarkerSize', 15);
plot(n, Gibbs, '-s', 'LineWidth', 2.5, 'MarkerSize', 15);
plot(n, SAEM,  '-d', 'LineWidth', 2.5, 'MarkerSize', 15);
plot(n, Poiss, '-^', 'LineWidth', 2.5, 'MarkerSize', 15);

xlabel('Sample Size (n)', 'FontSize',40);
ylabel('Average Entry-wise MSE', 'FontSize', 40);
title('J = 100: B^{(2)} Entry-wise MSE vs Sample Size', 'FontSize',40);

% Log scale with explicit ticks
set(gca, 'XScale', 'log');
xticks(n);
xticklabels(string(n));

% Increase font size
set(gca, 'FontSize', 40);

% Move legend lower
legend({'RL CSP','RL Gibbs','RL SAEM','Poisson DDE'}, ...
       'Location','east', 'FontSize', 20);

grid on;
box on;
axis tight;