clear;
addpath('/Users/jfeldm01/Library/CloudStorage/OneDrive-Kearney/Documents/Deep-Discrete-Encoders-main/CopDDE/Utilities/')
addpath('/Users/jfeldm01/Library/CloudStorage/OneDrive-Kearney/Documents/Deep-Discrete-Encoders-main/CopDDE//Algorithms/')

B5 = readtable('data.csv');

% Keep variables 3:102
X = B5(:, setdiff(3:102, [53,46]));
var_names = X.Properties.VariableNames;

[N, J] = size(X);
R = NaN(N, J);

K1_max = floor(J/3); K2_max = floor(K1_max/3); epsilon = .00001;

%% ----------------------------------------------------
% 1) Convert each column to integer ranks
%    Missing values stay as NaN
% ----------------------------------------------------
for j = 1:J
    yj = X{:, j};   % extract raw column data

    % Convert nonnumeric columns to numeric codes
    if iscell(yj) || isstring(yj) || ischar(yj) || iscategorical(yj)
        yj = double(categorical(yj));
    end

    % Force to column vector
    yj = yj(:);

    % Check that we now have numeric data
    if ~isnumeric(yj)
        error('Column %d (%s) could not be converted to numeric.', j, var_names{j});
    end

    % Identify missing values
    isn = isnan(yj);

    % Rank distinct observed values in sorted order
    yj_nonan = yj(~isn);
    [~, ~, ic] = unique(yj_nonan, 'sorted');

    % Store ranks
    R(~isn, j) = ic;
end

%% ----------------------------------------------------
% 2) Number of rank levels per variable
% ----------------------------------------------------
Rlevels = zeros(1, J);
for j = 1:J
    if all(isnan(R(:, j)))
        Rlevels(j) = 0;
    else
        Rlevels(j) = max(R(:, j));
    end
end



X = X{:,:};
X(isnan(X)) = 3;

Z_init = simulate_gaussian_mixture(X,J);
[prop_in, B1_in, B2_in, gamma_in, A1_in, A2_in] = Normal_init_v2(Z_init, K1_max, K1_max, K2_max,epsilon);

taus = [.1,.2,.3,.4];
t1s = [.02,.04,.06,.08];
t2s = [.02,.04,.06,.08];
temps = [.7,.8,.9];

% Store all results
results = [];
res_idx = 1;

% Initialize best-so-far trackers
best_avg_pMSE = Inf;
best_B1_CSP = [];
best_B2_CSP = [];
best_prop_CSP = [];
best_gamma_CSP = [];
best_settings = struct('tau', [], 't1', [], 't2', [], 'temp', []);

for t = 1:length(taus)
    for l_1 = 1:length(t1s)
        for l_2 = 1:length(t2s)
            for tt = 1:length(temps)

                % Current tuning parameters
                tau  = taus(t);
                t1   = t1s(l_1);
                t2   = t2s(l_2);
                temp = temps(tt);

                % Fit model
                [num_act1, num_act2, theta, theta2, gamma_CSP, p, q, ...
                    A1_sample_long, A2_sample_long, Z, prop_CSP, ...
                    B1_CSP, B2_CSP, A1_new, A2_new, it_num] = ...
                    get_SAEM_RL_CSP(X, Z_init, R, Rlevels, prop_in, ...
                    gamma_in, B1_in, B2_in, A1_in, A2_in, 1, 50, t1, t2, temp, tau);

                % Compute pMSE over repeated synthetic datasets
                nRep = 50;
                Kfold = 5;
                pMSE = zeros(nRep,1);

                parfor s = 1:nRep

                    % Generate synthetic data
                    [X_sim, Z_sim] = generate_X_Cop_pred(N, prop_CSP, B1_CSP, B2_CSP, gamma_CSP, X);

                    % Combine real and synthetic data
                    X_all = [X; X_sim];
                    y_all = [ones(N,1); zeros(N,1)];   % 1 = real, 0 = synthetic
                    y_cat = categorical(y_all);

                    % Cross-validation split
                    cv = cvpartition(y_all, 'KFold', Kfold);

                    % Out-of-fold predicted probabilities
                    p_hat = zeros(size(y_all));

                    for k = 1:Kfold
                        idxTrain = training(cv, k);
                        idxTest  = test(cv, k);

                        X_train = X_all(idxTrain, :);
                        y_train = y_cat(idxTrain);
                        X_test  = X_all(idxTest, :);

                        % Random forest
                        rf = TreeBagger(100, X_train, y_train, ...
                            'Method', 'classification', ...
                            'OOBPrediction', 'Off', ...
                            'MinLeafSize', 5);

                        % Predict held-out fold
                        [~, scores] = predict(rf, X_test);

                        % Probability of class "1" = real
                        idx_real = find(rf.ClassNames == categorical(1));
                        p_hat(idxTest) = scores(:, idx_real);
                    end

                    % pMSE for this repetition
                    c = mean(y_all);   % usually 0.5
                    pMSE(s) = mean((p_hat - c).^2);
                end

                % Average pMSE for this parameter combination
                avg_pMSE = mean(pMSE);

                % Store all results
                results(res_idx).tau = tau;
                results(res_idx).t1 = t1;
                results(res_idx).t2 = t2;
                results(res_idx).temp = temp;
                results(res_idx).avg_pMSE = avg_pMSE;
                results(res_idx).all_pMSE = pMSE;
                results(res_idx).B1_CSP = B1_CSP;
                results(res_idx).B2_CSP = B2_CSP;
                results(res_idx).prop_CSP = prop_CSP;
                results(res_idx).gamma_CSP = gamma_CSP;
                results(res_idx).num_act1 = num_act1;
                results(res_idx).num_act2 = num_act2;
                results(res_idx).it_num = it_num;

                res_idx = res_idx + 1;

                % Update best model/settings if improved
                if avg_pMSE < best_avg_pMSE
                    best_avg_pMSE = avg_pMSE;
                    best_B1_CSP = B1_CSP;
                    best_B2_CSP = B2_CSP;
                    best_prop_CSP = prop_CSP;
                    best_gamma_CSP = gamma_CSP;
                    best_A1 = A1_sample_long;
                    best_A2 = A2_sample_long;

                    best_settings.tau = tau;
                    best_settings.t1 = t1;
                    best_settings.t2 = t2;
                    best_settings.temp = temp;
                end

                fprintf('tau=%.4f, t1=%.4f, t2=%.4f, temp=%.4f, avg pMSE=%.6f, best=%.6f\n', ...
                    tau, t1, t2, temp, avg_pMSE, best_avg_pMSE);

            end
        end
    end
end

% Display best result
disp('Best settings found:')
disp(best_settings)
disp(['Best average pMSE: ', num2str(best_avg_pMSE)])

%% Find anchor items for B1 among active nodes only
top_N = 5;

J = size(B1_CSP, 1);
K1_full = size(B1_CSP, 2) - 1;   % total non-intercept columns

% Active non-intercept columns of B1_CSP
active_cols_B1 = find(any(B1_CSP(:, 2:end) ~= 0, 1)) + 1;   % shift by 1 for intercept
K1_active = length(active_cols_B1);

top_items_B1   = strings(K1_active, top_N);
top_index_B1   = zeros(K1_active, top_N);
top_tmp_values = zeros(K1_active, top_N);

for kk = 1:K1_active
    k_col = active_cols_B1(kk);   % actual column index in B1_CSP
    tmp = zeros(J,1);

    other_cols = setdiff(active_cols_B1, k_col);

    for j = 1:J
        if isempty(other_cols)
            tmp(j) = max(0, B1_CSP(j, k_col));
        else
            tmp(j) = max(0, min(B1_CSP(j, k_col) - B1_CSP(j, other_cols)));
        end
    end

    [sorted_tmp, sort_idx] = sort(tmp, 'descend');

    n_keep = min(top_N, J);
    top_index_B1(kk, 1:n_keep)   = sort_idx(1:n_keep);
    top_tmp_values(kk, 1:n_keep) = sorted_tmp(1:n_keep);
    top_items_B1(kk, 1:n_keep)   = string(var_names(sort_idx(1:n_keep)));
end

for kk = 1:K1_active
    fprintf('\nB1 Active Column %d:\n', active_cols_B1(kk) - 1);
    for m = 1:min(top_N, J)
        fprintf('  %d. %s (tmp = %.4f)\n', ...
            m, top_items_B1(kk,m), top_tmp_values(kk,m));
    end
end


%% Find anchor items for B2 among active nodes only
top_N = 3;

K1_rows = size(B2_CSP, 1);
K2_full = size(B2_CSP, 2) - 1;   % total non-intercept columns

% Active non-intercept columns of B2_CSP
active_cols_B2 = find(any(B2_CSP(:, 2:end) ~= 0, 1)) + 1;   % shift by 1 for intercept
K2_active = length(active_cols_B2);

top_items_B2      = strings(K2_active, top_N);
top_index_B2      = zeros(K2_active, top_N);
top_tmp_values_B2 = zeros(K2_active, top_N);

for kk = 1:K2_active
    k_col = active_cols_B2(kk);   % actual column index in B2_CSP
    tmp = zeros(K1_rows,1);

    other_cols = setdiff(active_cols_B2, k_col);

    for j = 1:K1_rows
        if isempty(other_cols)
            tmp(j) = max(0, B2_CSP(j, k_col));
        else
            tmp(j) = max(0, min(B2_CSP(j, k_col) - B2_CSP(j, other_cols)));
        end
    end

    [sorted_tmp, sort_idx] = sort(tmp, 'descend');

    n_keep = min(top_N, K1_rows);
    top_index_B2(kk, 1:n_keep)      = sort_idx(1:n_keep);
    top_tmp_values_B2(kk, 1:n_keep) = sorted_tmp(1:n_keep);
    top_items_B2(kk, 1:n_keep)      = "A1_" + string(sort_idx(1:n_keep));
end

for kk = 1:K2_active
    fprintf('\nB2 Active Column %d:\n', active_cols_B2(kk) - 1);
    for m = 1:min(top_N, K1_rows)
        fprintf('  %d. %s (tmp = %.4f)\n', ...
            m, top_items_B2(kk,m), top_tmp_values_B2(kk,m));
    end
end
%%% Predictive validity using only active nodes

% -----------------------------------------
% Build predictors
% -----------------------------------------
A1_mean = mean(A1_sample_long, 3);
A2_mean = mean(A2_sample_long, 3);

% Active nodes based on nonzero columns in B1_CSP and B2_CSP
active_A1 = find(any(B1_CSP(:, 2:end) ~= 0, 1));   % indices 1,...,K1
active_A2 = find(any(B2_CSP(:, 2:end) ~= 0, 1));   % indices 1,...,K2

% Keep only active columns
A1_mean_active = A1_mean(:, active_A1);
A2_mean_active = A2_mean(:, active_A2);

% Build tables
A1_tbl = array2table(A1_mean_active, ...
    'VariableNames', strcat('A1_', string(active_A1)));

A2_tbl = array2table(A2_mean_active, ...
    'VariableNames', strcat('A2_', string(active_A2)));

vars_to_add = {'gender','race','educ','marstat','employ','faminc_new'};
X_extra = B5(:, vars_to_add);

X_tree = [A1_tbl, A2_tbl, X_extra];

% Response (EDIT THIS)
y = B5.pid3;   % e.g., categorical(B5.outcome)

% -----------------------------------------
% Train / test split
% -----------------------------------------
cv = cvpartition(height(X_tree), 'HoldOut', 0.3);

idxTrain = training(cv);
idxTest  = test(cv);

X_train = X_tree(idxTrain, :);
X_test  = X_tree(idxTest, :);

y_train = y(idxTrain);
y_test  = y(idxTest);

% -----------------------------------------
% Fit tree
% -----------------------------------------
tree_model = fitctree(X_train, y_train,"MaxNumSplits",10);
view(tree_model, 'Mode', 'graph');
% -----------------------------------------
% Predict + evaluate
% -----------------------------------------
y_pred = predict(tree_model, X_test);

% Classification accuracy
accuracy = mean(y_pred == y_test);
fprintf('Test Accuracy: %.4f\n', accuracy);

% Confusion matrix
confusionchart(y_test, y_pred);