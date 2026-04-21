%% run_Dlayer_DDE_depCUSP.m
% Driver script for:
%   D-layer Copula DDE + Rank Likelihood + SAEM + dependent CUSP
%
% Requires these functions on your MATLAB path:
%   - fit_Dlayer_DDE_SAEM_RL_depCUSP.m
%   - generate_X_cop_D.m
%   - sim_block_loadings.m
%   - rescale_B1.m
%   - simulate_gaussian_mixture.m
%   - Normal_init_D.m
%   - logistic.m
%   - F_1_SAEM.m
%   - F_2_SAEM.m
%   - thres.m
%
% Notes:
%   * B_cell_true{d} includes intercept in column 1
%   * A_cell_init{d} should NOT include an intercept column
%   * This script stores one struct per simulation replicate in RESULTS

clear; clc;

%% ------------------------------------------------------------
% Add paths
% -------------------------------------------------------------
addpath('/Users/jfeldm01/Library/CloudStorage/OneDrive-Kearney/Documents/Deep-Discrete-Encoders-main/CopDDE/Utilities/')
addpath('/Users/jfeldm01/Library/CloudStorage/OneDrive-Kearney/Documents/Deep-Discrete-Encoders-main/CopDDE//Algorithms/')

%% ------------------------------------------------------------
% True model dimensions
% -------------------------------------------------------------
D = 3;

K3_true = 2;
K2_true = 6;
K1_true = 18;

J = 108;


B3_sub = zeros(K2_true, K3_true);

max_val = 4;
idx = floor(linspace(1, K2_true, K3_true + 1));
for i = 1:(length(idx) - 1)
    if mod(i,2) == 0
        B3_sub(idx(i):idx(i+1), i)   = max_val;
        B3_sub(idx(i):idx(i+1), i-1) = -max_val/3;
    else
        B3_sub(idx(i):idx(i+1), i) = max_val;
        if i < length(idx) - 2
            B3_sub(idx(i):idx(i+1), i+1) = -max_val/3;
        end
    end
end

B2_sub = sim_block_loadings(K1_true, K2_true,K1_true/K2_true,4,4/3,0,1); % first entry by J: J = 50 -> 5, J = 100 -> 10, J = 150 -> 15

B1_sub = sim_block_loadings(J, K1_true,J/K1_true,10,5,0,1); % first entry by J: J = 50 -> 5, J = 100 -> 10, J = 150 -> 15


B3_true = [-2 * ones(K2_true,1), B3_sub];
B2_true = [-2 * ones(K1_true,1), B2_sub];

B1_true_unscale = [ ...
    [-2 * ones(floor(J/3),1); ...
     -2 * ones(floor(J/3),1); ...
     -2 * ones(J - 2*floor(J/3),1)], ...
    B1_sub];



% Optional support indicators
G1 = [(B1_sub ~= 0), zeros(J, floor(J/3) - K1_true)];
G2 = [[(B2_sub ~= 0); zeros(floor(J/3) - K1_true, K2_true)], ...
      zeros(floor(J/3), floor(J/9) - K2_true)];
G3 = [[(B3_sub ~= 0); zeros(floor(J/9) - K2_true, K3_true)], ...
      zeros(floor(J/9), floor(J/27) - K3_true)];

prop_true = 0.5 * ones(1, K3_true);
gamma_true = ones(J,1);

B_cell_true = cell(1,D);
B_cell_true{1} = B1_true_unscale;
B_cell_true{2} = B2_true;
B_cell_true{3} = B3_true;

lambdas = randi([1, 10], 1, J);

%% ------------------------------------------------------------
% Simulation settings
% -------------------------------------------------------------
C_sims     = 100;
n_vec      = [500 1000 2000 4000 8000 16000];

% Initial layer sizes for fitting
K_cell_init = cell(1,D);
K_cell_init{1} = floor(J/3);
K_cell_init{2} = floor(J/9);
K_cell_init{3} = floor(J/27);

B2_true_aug = zeros(K_cell_init{1}, K_cell_init{2} + 1);
B2_true_aug(1:K1_true, 1:K2_true + 1) = B2_true;

B3_true_aug = zeros(K_cell_init{2}, K_cell_init{3} + 1);
B3_true_aug(1:K2_true, 1:K3_true + 1) = B3_true;


epsilon_init = 1e-4;

% Fitter controls

C        = 1;                 % MC draws inside SAEM step
it       = 50;                % max iterations
temp     = .9;               % initial temperature           
t_spike  = [0.01, 0.03, 0.005]; % layer-specific spike scales


%% ------------------------------------------------------------
% Preallocate output
% -------------------------------------------------------------
RESULTS = cell(numel(n_vec), C_sims);

%% ------------------------------------------------------------
% Main simulation loop
% -------------------------------------------------------------
for aa = numel(n_vec):-1:1

    Nsim = n_vec(aa);
    tau = max(.3,3*Nsim^-.3);
    for c = 1:C_sims

        rng(50 + c);

        %% ----------------------------------------------------
        % 1) Simulate data from true model
        % -----------------------------------------------------
        [X, Z_true, A_true] = generate_X_cop_D( ...
            Nsim, lambdas, prop_true, D, B_cell_true, gamma_true);

        % Optional rescaling diagnostic for layer 1
        [B1_true_scale, gamma_true_scale] = ...
            rescale_B1(prop_true, B2_true, B1_true_unscale, gamma_true,true, Z_true);

        B1_true_scale_aug = [B1_true_scale, zeros(J, floor(J/3) - K1_true)]; %#ok<NASGU>




        %% ----------------------------------------------------
        % 2) Build rank-likelihood object R
        %    R(i,j) = rank level / category index for X(i,j)
        % -----------------------------------------------------
        R = NaN(Nsim, J);

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

      
        %% ----------------------------------------------------
        % 5) Initialize latent Gaussian Z
        % -----------------------------------------------------
        % mixture-based init
        Z_init = simulate_gaussian_mixture(X, J);


        %% ----------------------------------------------------
        % 6) Initialize D-layer model parameters
        % -----------------------------------------------------
        [prop_init, B_cell_init, gamma_init, A_cell_init] = ...
            Normal_init_D(Z_init, D, K_cell_init, epsilon_init);

        %% ----------------------------------------------------
        % 7) Fit D-layer model
        % -----------------------------------------------------


        fit_out = get_SAEM_RL_CSP_D( ...
        X, Z_init, R, Rlevels, ...
        prop_init, gamma_init, B_cell_init, A_cell_init, ...
        1, 50, t_spike, temp, tau);

        fit_out.num_act
        fit_out.B{3}


        RESULTS{aa, c} = fit_out;
    end

    fprintf('Finished sample size %d\n', Nsim);
end


num_act1 = zeros(aa, C_sims); num_act2 = zeros(aa,C_sims); num_act3= zeros(aa,C_sims);

for aa = numel(n_vec):-1:1
    for c = 1:C_sims
        num_act1(aa,c) = RESULTS{aa,c}.num_act(1);
        num_act2(aa,c) = RESULTS{aa,c}.num_act(2);
        num_act3(aa,c) = RESULTS{aa,c}.num_act(3);
    end
end

B1_D = zeros(J,K_cell_init{1} + 1,aa,C_sims); B2_D = zeros(K_cell_init{1}, K_cell_init{2}+1, aa, C_sims); B3_D = zeros(K_cell_init{2}, K_cell_init{3}+1, aa, C_sims);

for aa = 1:6
    for c = 1:C_sims
        [B1_D(:,:,aa,c), B2_D(:,:,aa,c),p,q,assign1, assign2] = format_estimates(B1_true_scale_aug, B2_true_aug,...
            RESULTS{aa,c}.B{1},RESULTS{aa,c}.B{2},RESULTS{aa,c}.gamma,RESULTS{aa,c}.prop,true, RESULTS{aa,c}.Z, true);
        B3_i = RESULTS{aa,c}.B{3};
        B3_i = B3_i(assign2,:);
        costmat = zeros(K_cell_init{3}, K_cell_init{3});
        B3_est = B3_i(:,2:end);
        for k = 1:K3_true
            costmat(k,:) = - sum(B3_est(find(B3_true_aug(:,k+1) > 0), :), 1); 
        end 
        [assignment3, ~] = munkres(costmat);
        B3_est = [B3_i(:,1),B3_est(:, assignment3)];
   
        B3_D(:,:,aa,c) = B3_est;
    end
end

        

%----------------------------------------------

%% Error Evaluations

G1_true_aug = B1_true_scale_aug ~= 0;
G2_true_aug = B2_true_aug ~= 0;
G3_true_aug = B3_true_aug ~= 0;

MSE_B1s_CSP    = [];
MSE_B2s_CSP    = [];
MSE_B3s_CSP    = [];


acc_G1s_CSP    = [];
acc_G2s_CSP    = [];
acc_G3s_CSP    = [];



n_aa = size(B1_D, 3);
n_rep = size(B1_D, 4);

for aa = 1:n_aa

    MSE_B1_CSP    = [];
    MSE_B2_CSP    = [];
    MSE_B3_CSP    = [];
 
    acc_G1_CSP    = [];
    acc_G2_CSP    = [];
    acc_G3_CSP    = [];

    for c = 1:n_rep

        % -----------------------------
        % CSP
        % -----------------------------
        B1_i_CSP = B1_D(:,:,aa,c);
        G1_i_CSP = B1_i_CSP ~= 0;
        B2_i_CSP = B2_D(:,:,aa,c);
        G2_i_CSP = B2_i_CSP ~= 0;
        B3_i_CSP = B3_D(:,:,aa,c);
        G3_i_CSP = B3_i_CSP ~= 0;
        


        MSE_B1_CSP = [MSE_B1_CSP, mean((B1_i_CSP - B1_true_scale_aug).^2, "all")];
        MSE_B2_CSP = [MSE_B2_CSP, mean((B2_i_CSP - B2_true_aug).^2, "all")];
        MSE_B3_CSP = [MSE_B3_CSP, mean((B3_i_CSP - B3_true_aug).^2, "all")];
        acc_G1_CSP = [acc_G1_CSP, mean(G1_true_aug == G1_i_CSP, "all")];
        acc_G2_CSP = [acc_G2_CSP, mean(G2_true_aug == G2_i_CSP, "all")];
        acc_G3_CSP = [acc_G3_CSP, mean(G3_true_aug == G3_i_CSP, "all")];

       
    end

    MSE_B1s_CSP   = [MSE_B1s_CSP;   MSE_B1_CSP];
    MSE_B2s_CSP   = [MSE_B2s_CSP;   MSE_B2_CSP];
    MSE_B3s_CSP   = [MSE_B3s_CSP;   MSE_B3_CSP];

    acc_G1s_CSP   = [acc_G1s_CSP;   acc_G1_CSP];
    acc_G2s_CSP   = [acc_G2s_CSP;   acc_G2_CSP];
    acc_G3s_CSP   = [acc_G3s_CSP;   acc_G3_CSP];

end
mean(num_act1,2)
mean(num_act2,2)
mean(num_act3,2)
mean(acc_G1s_CSP,2)
mean(acc_G2s_CSP,2)
mean(acc_G3s_CSP,2)


%%% Plots

% Sample sizes
n = [500 1000 2000 4000 8000 16000];

% MSE values
MSE_B1 = [mean(MSE_B1s_CSP,2)];

% Extract methods
CSP   = MSE_B1(:,1);

% --- JITTER (multiplicative for log scale) ---
jitter = [0.97, 0.99, 1.01, 1.03];  % small offsets

n_CSP   = n * jitter(1);

% Plot
figure;
hold on;

plot(n_CSP, CSP, '-o', ...
    'LineWidth', 2.5, ...
    'MarkerSize', 15, ...
    'MarkerFaceColor','auto');

xlabel('Sample Size (n)', 'FontSize', 40);
ylabel('Average Entry-wise MSE', 'FontSize',40);
title(' B^{(1)} Entry-wise MSE vs Sample Size', 'FontSize',40);

% Log scale with clean ticks
set(gca, 'XScale', 'log');
xticks(n);
xticklabels(string(n));

set(gca, 'FontSize', 40);

legend({'RL CSP','RL Gibbs','RL SAEM','Poisson DDE'}, ...
       'Location','east', 'FontSize', 20);
xlim([400 20000]);

grid on;
box on;
axis tight;

%%%% B2
% Sample sizes

% MSE values (rows = n, cols = methods)
MSE_B2 = [mean(MSE_B2s_CSP,2)];

% Extract methods
CSP   = MSE_B2(:,1);
% Plot
figure;
hold on;

plot(n, CSP,   '-o', 'LineWidth', 2.5, 'MarkerSize', 15);

xlabel('Sample Size (n)', 'FontSize',40);
ylabel('Average Entry-wise MSE', 'FontSize', 40);
title('B^{(2)} Entry-wise MSE vs Sample Size', 'FontSize',40);

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

% MSE values (rows = n, cols = methods)
MSE_B3 = [mean(MSE_B3s_CSP,2)];

% Extract methods
CSP   = MSE_B3(:,1);
% Plot
figure;
hold on;

plot(n, CSP,   '-o', 'LineWidth', 2.5, 'MarkerSize', 15);

xlabel('Sample Size (n)', 'FontSize',40);
ylabel('Average Entry-wise MSE', 'FontSize', 40);
title('B^{(3)} Entry-wise MSE vs Sample Size', 'FontSize',40);

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