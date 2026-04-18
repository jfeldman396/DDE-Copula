function out = get_SAEM_RL_CSP_D( ...
    X, Z, R, Rlevels, ...
    prop_in, gamma_in, B_in, A_in, ...
    C, it, t_spike, temp, tau)

% ============================================================
% General D-layer version of get_SAEM_RL_adapt_indepCUSP
%
% Layer convention:
%   d = 1,...,D
%   A{d}      : N x K(d) binary latent layer
%   B{1}      : J x (K(1)+1), Gaussian layer for Z | A{1}
%   B{d}      : K(d-1) x (K(d)+1), logistic layer for A{d-1} | A{d}, d>=2
%   prop      : 1 x K(D), top-layer Bernoulli probabilities
%
% Inputs
%   prop_in   : 1 x K(D)
%   gamma_in  : J x 1 or 1 x J
%   B_in      : cell(1,D), B_in{1},...,B_in{D}
%   A_in      : cell(1,D), A_in{1},...,A_in{D}
%   t_spike   : scalar or 1 x D, spike scales for CUSP
%
% Output
%   out is a struct containing updated parameters
% ============================================================

%% -----------------------------
% Dimensions
% ------------------------------
D = numel(A_in);
N = size(X,1);
J = size(Z,2);

K = zeros(1,D);
for d = 1:D
    K(d) = size(A_in{d},2);
end

if isscalar(t_spike)
    t_spike = repmat(t_spike, 1, D);
end

%% -----------------------------
% Initialize parameters
% ------------------------------
A = A_in;
B = B_in;
gamma = gamma_in(:);
prop = prop_in;

% CUSP state by layer
theta_slab = cell(1,D);
theta      = cell(1,D);
z_cusp     = cell(1,D);
v          = cell(1,D);
omega      = cell(1,D);

for d = 1:D
    theta_slab{d} = ones(K(d),1);

    % truncation K(d)+1
    v{d} = rand(1, K(d)+1);
    omega{d} = (1/(K(d)+1))*ones(1,K(d)+1);

    % initialize CUSP labels near middle
    z_cusp{d} = floor(K(d)/2)*ones(1, K(d));

    % penalty matrix for non-intercepts
    if d == 1
        theta{d} = t_spike(d) * ones(J, K(d));
    else
        theta{d} = t_spike(d) * ones(K(d-1), K(d));
    end
end

window = 7;
burn   = 5;
rel_hist = cell(1,D);
for d = 1:D
    rel_hist{d} = [];
end

%% -----------------------------
% Optimization settings
% ------------------------------
epsilon = 1e-2; %#ok<NASGU>
bb = [];
Aeq = [];
beq = [];
AA = [];
options = optimset('Display','off','MaxIter',15);

lb = cell(1,D);
ub = cell(1,D);

% B{1}: Gaussian layer
lb{1} = [-10, -10*ones(1,K(1))];
ub{1} =  10*ones(1,K(1)+1);

% B{d}, d>=2: logistic layers
for d = 2:D
    lb{d} = [-5, -10*ones(1,K(d))];
    ub{d} =  10*ones(1,K(d)+1);
end

%% -----------------------------
% Main iteration
% ------------------------------
iter_indicator = true;
it_num = 0;

num_act = zeros(D,1);
for d = 1:D
    num_act(d) = K(d);
end
 
while iter_indicator
    
    A_old = A;
    
    ber = cell(1,D);
    p_store = cell(1,D);
    for d = 1:D
        ber{d} = zeros(N, K(d));
        p_store{d} = zeros(N, K(d));
    end
    
    ber_rows = cell(N,1);   % each entry will be a 1xD cell
    
    parfor i = 1:N
    
        A_i_old = cell(1,D);
        for d = 1:D
            A_i_old{d} = A_old{d}(i,:);
        end
    
        ber_i = cell(1,D);
        for d = 1:D
            ber_i{d} = zeros(1,K(d));
        end
    
        % top layer
        for l = 1:K(D)
    
            A_top = A_i_old{D};
            A_top1 = A_top; A_top1(l) = 1;
            A_top0 = A_top; A_top0(l) = 0;
    
            eta1 = B{D} * [1, A_top1]';
            eta0 = B{D} * [1, A_top0]';
    
            logit_top = log(prop(l)/(1-prop(l))) + ...
                sum(A_i_old{D-1}' .* B{D}(:,l+1)) - ...
                sum(log((1 + exp(eta1)) ./ (1 + exp(eta0))));
    
            ber_i{D}(l) = logistic(temp * logit_top);
        end
    
        % intermediate layers
        for d = D-1:-1:2
            for k = 1:K(d)
    
                A_curr = A_i_old{d};
    
                logit_up = [1, A_i_old{d+1}] * B{d+1}(k,:)';
    
                A1 = A_curr; A1(k) = 1;
                A0 = A_curr; A0(k) = 0;
    
                eta1 = B{d} * [1, A1]';
                eta0 = B{d} * [1, A0]';
    
                child_term = sum(A_i_old{d-1}' .* B{d}(:,k+1)) - ...
                    sum(log((1 + exp(eta1)) ./ (1 + exp(eta0))));
    
                ber_i{d}(k) = logistic(temp * (logit_up + child_term));
            end
        end
    
        % bottom layer
        for k = 1:K(1)
    
            A1_i = A_i_old{1};
            A1_minus_k = A1_i(1:end ~= k);
    
            mu_minus_k = B{1}(:, 1:end~=(k+1)) * [1; A1_minus_k(:)];
    
            logit_bottom = [1, A_i_old{2}] * B{2}(k,:)' ...
                - 0.5 * sum((B{1}(:,k+1).^2 ...
                - 2*(Z(i,:)' - mu_minus_k).*B{1}(:,k+1)) ./ gamma);
    
            ber_i{1}(k) = logistic(temp * logit_bottom);
        end
    
        ber_rows{i} = ber_i;
    end
    
    for i = 1:N
        for d = 1:D
            ber{d}(i,:) = ber_rows{i}{d};
        end
    end

    
    % sample binary layers C times
    A_sample_long = cell(1,D);
    for d = 1:D
        A_sample_long{d} = zeros(N, K(d)+1, C);
        for c = 1:C
            A_sample_long{d}(:,:,c) = [ones(N,1), double(rand(N,K(d)) < ber{d})];
        end
        p_store{d} = ber{d};
    end

  
    %% =======================================================
    % 2) Rank-likelihood simulation for Z using A{1}
    % ========================================================
    mu = mean(A_sample_long{1}, 3) * B{1}.';

    parfor j = 1:J

        mu_j = mu(:,j);
        sd_j = sqrt(gamma(j)/temp);

        Zj = Z(:,j);

        ir_na = find(isnan(R(:,j)));
        Zj(ir_na) = mu_j(ir_na) + sd_j * randn(numel(ir_na),1);

        for r = 1:Rlevels(j)

            ir = find(R(:,j)==r & ~isnan(R(:,j)));
            if isempty(ir), continue; end

            lbCand = Zj(R(:,j)==(r-1));
            lbCand = lbCand(~isnan(lbCand));
            if isempty(lbCand)
                lower_b = -inf;
            else
                lower_b = max(lbCand);
            end

            ubCand = Zj(R(:,j)==(r+1));
            ubCand = ubCand(~isnan(ubCand));
            if isempty(ubCand)
                upper_b = inf;
            else
                upper_b = min(ubCand);
            end

            pL = normcdf(lower_b, mu_j(ir), sd_j);
            pU = normcdf(upper_b, mu_j(ir), sd_j);

            eps0 = 1e-12;
            pL = max(min(pL,1-eps0),eps0);
            pU = max(min(pU,1-eps0),eps0);

            if pU <= pL
                mid = (pL+pU)/2;
                pL = mid;
                pU = mid;
            end

            u = pL + (pU-pL).*rand(numel(ir),1);
            Zj(ir) = norminv(u, mu_j(ir), sd_j);
        end

        Z(:,j) = Zj;
    end

    
    %% =======================================================
    % 3) CUSP updates for each layer
    % ========================================================
    % layer 1 uses columns of B{1} (size J x K1)
    % layers d>=2 use columns of B{d} (size K(d-1) x K(d))

    for d = D:-1:1

        if d == 1
            num_rows = J;
        else
            num_rows = num_act(d-1);
        end

        L = K(d) + 1;

        B_pad = [B{d}, zeros(size(B{d},1),1)];
        prob_mat = zeros(K(d), L);

        parfor h = 1:K(d)

            log_probs = -inf(1, L);
            xh = B_pad(:, h+1);

            log_dens_spike = -num_rows*log(2*t_spike(d)) - sum(abs(xh))/t_spike(d);

            sum_abs_x = sum(abs(xh));
            log_dens_slab = log(5) - num_rows*log(2) - gammaln(1) + gammaln(1+num_rows) ...
                - (1 + num_rows)*log(5 + sum_abs_x);

            for ell = 1:L
                if ell <= h
                    log_col = log_dens_spike;
                else
                    log_col = log_dens_slab;
                end

                log_probs(ell) = log(omega{d}(ell)) + log_col;
            end

            mx = max(log_probs);
            probs = exp(log_probs - mx);
            probs = probs / sum(probs);
            prob_mat(h,:) = probs;
        end

        [~, z_cusp_d] = max(prob_mat, [], 2);
        z_cusp{d} = z_cusp_d';

        % stick-breaking update
        alpha_d = K(d);
        v_d = zeros(1, L);
        omega_d = zeros(1, L);

        for ell = 1:K(d)
            n_l   = sum(z_cusp{d}(1:K(d)) == ell);
            n_gtl = sum(z_cusp{d}(1:K(d)) >  ell);

            v_d(ell) = betarnd(1 + n_l, alpha_d + n_gtl);
        end

        v_d(L) = 1;

        omega_d(1) = v_d(1);
        for ell = 2:L
            omega_d(ell) = v_d(ell) * prod(1 - v_d(1:ell-1));
        end

        v{d} = v_d;
        omega{d} = omega_d;

        % slab scale updates
        for h = 1:K(d)
            if z_cusp{d}(h) > h
                shape_post = 1 + num_rows;
                rate_post  = 5 + sum(abs(B{d}(:,h+1)));
                theta_slab{d}(h) = 1 / gamrnd(shape_post, 1 ./ rate_post);
            end
        end

        % penalty matrix
        if d == 1
            theta{d} = zeros(J, K(d));
        else
            theta{d} = zeros(K(d-1), K(d));
        end

        for h = 1:K(d)
            if z_cusp{d}(h) <= h
                theta{d}(:,h) = t_spike(d);
            else
                theta{d}(:,h) = theta_slab{d}(h);
            end
        end
    end

  
    %% =======================================================
    % 4) Top-layer Bernoulli MAP
    % ========================================================
    prop_update = sum(p_store{D}, 1) / N;

    %% =======================================================
    % 5) Update B{1} and gamma
    % ========================================================
    B_update = B;
    gamma_update = zeros(J,1);
    
    %% =======================================================
    % 5) Update B{1} and gamma
    % ========================================================
    B1_update = zeros(size(B{1}));
    
    parfor j = 1:J
        
        Zj = Z(:,j);
    
        f_loglik = F_1_SAEM_cop(Zj, A_sample_long{1}, N, K(1), C, temp, gamma(j));
    
        W1 = [0, 1 ./ theta{1}(j,:)];
    
        f_pen = @(x) sum(W1(2:end) .* abs(x(2:end)));
        f_j = @(x) f_loglik(x) + f_pen(x);
    
        opt = fmincon(f_j, B{1}(j,:), AA, [], Aeq, beq, lb{1}, ub{1}, [], options);
        B1_update(j,:) = opt;
    
        eta = reshape( ...
            sum(bsxfun(@times, A_sample_long{1}, reshape(opt, [1, K(1)+1, 1])), 2), ...
            [N, C] ...
        );
    
        SSE = sum((Zj - eta).^2, 'all');
        gamma_update(j) = (1 + 0.5*SSE) / (1 + 0.5*N*C + 1);
    end
    
    B_update{1} = B1_update;
    
    %% =======================================================
    % 6) Update B{d}, d >= 2
    % ========================================================
    for d = 2:D
    
        Bd_update = zeros(size(B{d}));
    
        parfor k = 1:K(d-1)
    
            A_child_k = reshape(A_sample_long{d-1}(:,k+1,:), [N, C]);
    
            f_loglik = F_2_SAEM_cop(A_child_k, A_sample_long{d}, N, K(d), C, temp);
    
            Wd = [0, 1 ./ theta{d}(k,:)];
    
            f_pen = @(x) sum(Wd(2:end) .* abs(x(2:end)));
            f_k = @(x) f_loglik(x) + f_pen(x);
    
            opt = fmincon(f_k, B{d}(k,:), AA, [], Aeq, beq, lb{d}, ub{d}, [], options);
            Bd_update(k,:) = opt;
        end
    
        B_update{d} = Bd_update;
    end

    it_num = it_num + 1;
    
    %% =======================================================
    % 7) Threshold + hierarchical gating
    % ========================================================
    num_act = zeros(1,D);
    ind_inact = cell(1,D);

    for d = 1:D
        num_act(d) = sum(z_cusp{d} > 1:K(d));
        ind_inact{d} = find(z_cusp{d} <= 1:K(d));
    end

    
    for d = 1:D
        B_update{d}(:,2:end) = thres(B_update{d}(:,2:end), tau);
    end

    % If a node in layer d is inactive, zero out rows of B{d+1}
    for d = 1:D-1
        B_update{d+1}(ind_inact{d}, :) = 0;
    end

    %% =======================================================
    % 8) Temperature schedule
    % ========================================================
    if it_num > burn
        temp = min(1, temp + 0.01*((it_num - burn)-1));
    end

    %% =======================================================
    % 9) Convergence check
    % ========================================================
    stop_now = true;
    for d = 1:D
        rel_err_d = norm(B_update{d}(:,2:end) - B{d}(:,2:end), 'fro') / ...
                    (norm(B{d}(:,2:end), 'fro') + 1e-6);
        rel_hist{d} = [rel_hist{d}, rel_err_d]; %#ok<AGROW>

        if numel(rel_hist{d}) < window
            stop_now = false;
        else
            recent_d = rel_hist{d}(end-window+1:end);
            if std(recent_d) >= 0.005
                stop_now = false;
            end
        end
    end

    %% =======================================================
    % 10) Update latent binary point estimates
    % ========================================================
    A_new = cell(1,D);
    for d = 1:D
        A_new{d} = double(mean(A_sample_long{d}(:,2:end,:), 3) > 0.5);
    end

    A = A_new;
    B = B_update;
    gamma = gamma_update;
    prop = prop_update;

  

    if stop_now
        for d = 1:D
            B{d}(:, ind_inact{d}+1) = 0;
        end
        disp('ending algorithm')
        iter_indicator = false;
        break
    end

    if iter_indicator
        iter_indicator = it_num < it;
        if ~iter_indicator
            for d = 1:D
                B{d}(:, ind_inact{d}+1) = 0;
            end
            disp('Ending algorithm for iteration budget')
        end
    end
end

%% -----------------------------
% Pack output
% ------------------------------
out = struct();
out.A = A;
out.B = B;
out.Z = Z;
out.gamma = gamma;
out.prop = prop;
out.theta = theta;
out.theta_slab = theta_slab;
out.z_cusp = z_cusp;
out.v = v;
out.omega = omega;
out.num_act = num_act;
out.it_num = it_num;
out.A_sample_long = A_sample_long;
out.p = p_store;

end