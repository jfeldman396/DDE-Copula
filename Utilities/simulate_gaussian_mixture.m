function Sim_all = simulate_gaussian_mixture(X, J)
% SIMULATE_GAUSSIAN_MIXTURE
%   Simulate Gaussian mixture–based columns using truncated SVD 
%   and k-means clustering with elbow detection.
%
% INPUTS:
%   X : N x P data matrix
%   J : number of columns to simulate (e.g., first J columns)
%
% OUTPUT:
%   Sim_all : N x J matrix of simulated columns (quantile-matched)
%
% Steps:
%   1. Compute SVD of normalized X
%   2. Select number of components (k) capturing 95% variance
%   3. Perform rank-k reconstruction
%   4. For each column j=1:J:
%         - find nearest column in latent space
%         - estimate number of clusters using elbow method
%         - fit Gaussian mixture via k-means
%         - simulate and quantile-match new data

    % --- Step 1: Normalize and compute SVD ---
    [U, S, V] = svd(normalize(X), 'econ');
    sing_vals = diag(S);
    total_var = sum(sing_vals.^2);
    cum_var = cumsum(sing_vals.^2) / total_var;

    % --- Step 2: Select number of components covering 95% of variance ---
    K1 = find(cum_var >= 0.8, 1);
    % fprintf('Retaining %d singular components (80%% variance)\n', K1);

    % --- Step 3: Truncated rank-K1 reconstruction ---
    U_k = U(:,1:K1);
    S_k = S(1:K1,1:K1);
    V_k = V(:,1:K1);
    A_k = U_k * S_k * V_k';

    % --- Step 4: Distance matrix among right singular vectors ---
    D = pdist2(V_k, V_k);
    N = size(X,1);
    maxK = 10;
    wcss = zeros(maxK,1);

    % --- Step 5: Initialize output matrix ---
    Sim_all = zeros(N, J);
     
    % --- Step 6: Loop over columns ---
    for j = 1:J
        D(j,j) = inf;  % ignore self-distance

        % Find index of closest column to j
        [~, closest_col] = min(D(j,:));

        % Subset rank-K1 approximation
        X_subset = A_k(:, [j, closest_col]);
        noisy_X = X_subset;
        % Step 7: Find optimal k via elbow method
        for k = 1:maxK
            warning('off', 'stats:kmeans:FailedToConverge');
            [~, ~, sumd] = kmeans(noisy_X, k, 'Replicates', 1, 'Display', 'off');
            wcss(k) = sum(sumd);
        end
        x = (1:maxK)'; y = wcss;
        x1 = x(1); y1 = y(1);
        x2 = x(end); y2 = y(end);
        dist = abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1) ./ sqrt((y2-y1)^2 + (x2-x1)^2);
        [~, elbow_k] = max(dist);
 

        % Step 8: Fit k-means with optimal k
        [idx, C] = kmeans(noisy_X, elbow_k, 'Replicates', 1, 'Display', 'off');

        % Step 9: Cluster proportions
        proportions = accumarray(idx, 1) / length(idx);

        % Step 10: Covariance per cluster
        covariances = cell(elbow_k,1);
        for k = 1:elbow_k
            cluster_points = noisy_X(idx==k, :);
            if size(cluster_points,1) > 1
                covariances{k} = cov(cluster_points);
            else
                covariances{k} = eye(size(noisy_X,2));
            end
        end

        % Step 11: Simulate Gaussian mixture
        sim = zeros(N,2);
        for i = 1:N
            cluster = find(mnrnd(1, proportions));
            sim(i,:) = mvnrnd(C(cluster,:), covariances{cluster});
        end

        % scatter(sim(:,1), sim(:,2))
        % drawnow;
        % Step 12: Quantile matching (first column)
        [sorted_X, orderX] = sort(X(:,j));
        [sorted_sim, orderSim] = sort(sim(:,1));

        matched_sim1 = zeros(N,1);
        matched_sim1(orderX) = sorted_sim;

        % Step 13: Store result
        Sim_all(:, j) = matched_sim1;
    end
end