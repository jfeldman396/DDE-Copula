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

K3_true = 3;
K2_true = 10;
K1_true = 30;

J = 150;

%% ------------------------------------------------------------
% Construct true loading matrices
% -------------------------------------------------------------
B3_sub = zeros(K2_true, K3_true);

max_val = 4;
idx = floor(linspace(1, K2_true, 4));
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

B2_sub = sim_block_loadings_test(K1_true, K2_true,K2_true, 3, 4, 4/3, 0,1);
B1_sub = sim_block_loadings_test(J, K1_true,K1_true,5,10,5,0,1);

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
n_vec      = [4000, 8000,12000,16000];

% Initial layer sizes for fitting
K_cell_init = cell(1,D);
K_cell_init{1} = 50;
K_cell_init{2} = 16;
K_cell_init{3} = 5;

epsilon_init = 1e-4;

% Fitter controls

C        = 1;                 % MC draws inside SAEM step
it       = 50;                % max iterations
temp     = .7;               % initial temperature
tau      = .3;               % threshold level
t_spike  = [0.01, 0.02, 0.01]; % layer-specific spike scales


%% ------------------------------------------------------------
% Preallocate output
% -------------------------------------------------------------
RESULTS = cell(numel(n_vec), C_sims);

%% ------------------------------------------------------------
% Main simulation loop
% -------------------------------------------------------------
for aa = 1:numel(n_vec)

    Nsim = n_vec(aa);

    parfor (c = 1:C_sims, n_parallel)

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
        % Option A: mixture-based init
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
    1, 50, t_spike, temp, tau)

        %% ----------------------------------------------------
        % 8) Store results
        % -----------------------------------------------------
        tmp = struct();

        % truth
        tmp.N = Nsim;
        tmp.X = X;
        tmp.Z_true = Z_true;
        tmp.A_true = A_true;
        tmp.prop_true = prop_true;
        tmp.gamma_true = gamma_true;
        tmp.B_cell_true = B_cell_true;
        tmp.B1_true_scale = B1_true_scale;
        tmp.gamma_true_scale = gamma_true_scale;

        % rank-likelihood objects
        tmp.R = R;
        tmp.Rlevels = Rlevels;
        tmp.Ranks = Ranks;

        % initial values
        tmp.Z_init = Z_init;
        tmp.prop_init = prop_init;
        tmp.gamma_init = gamma_init;
        tmp.B_cell_init = B_cell_init;
        tmp.A_cell_init = A_cell_init;

        % fit
        tmp.fit = fit_out;

        % supports
        tmp.G1 = G1;
        tmp.G2 = G2;
        tmp.G3 = G3;

        RESULTS{aa, c} = tmp;
    end

    fprintf('Finished sample size %d\n', Nsim);
end

%% ------------------------------------------------------------
% Save
% -------------------------------------------------------------
save('RESULTS_Dlayer_DDE_depCUSP.mat', 'RESULTS', 'n_vec', ...
     'C_sims', 'model', 'D', 'K1_true', 'K2_true', 'K3_true', ...
     'J', 'prop_true', 'gamma_true', 'B_cell_true', ...
     '-v7.3');