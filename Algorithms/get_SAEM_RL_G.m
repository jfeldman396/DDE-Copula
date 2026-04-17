function [gamma,p,q,A1_sample_long, A2_sample_long,Z,prop, B1, B2, A1_new, A2_new, t] = get_SAEM_RL_G(X, Z, R, Rlevels, prop_in, B1_in, B2_in, gamma_in, A1_in, A2_in, C, tol, it, temp, G1, G2)

% Step 3: penalized SAEM / PEM
% below is a one-line code for a fast output
% note that the latent variables are subject to label permutation
K2 = size(prop_in,2);
[J, K1] = size(B1_in);
K1 = K1-1;

N = size(X, 1);
prop = prop_in;
B1 = B1_in;
B2 = B2_in;
gamma = gamma_in.';
% iteration settings
t = 1;                  % iteration index
iter_indicator = true;

prop_update = prop_in;
B1_update = zeros(J, K1+1);
B2_update = zeros(K1, K2+1);
gamma_update = ones(J,1);


% optimization settings
bb = []; Aeq = []; beq = [];
AA = [];

%% general coefs
lb_1 = [-10*ones(1,1), -10*ones(1, K1)]; % zeros(J, K1+1);
ub_1 = 10*ones(1, K1+1);

lb_2 = [-5*ones(1,1), -10*ones(1, K2)]; % zeros(J, K1+1);
ub_2 = 10*ones(1, K2+1);
options = optimset('Display', 'off', 'MaxIter', 10);

% partition_ftn = sum(log(factorial(X)), 'all');
A1_new = A1_in;
A2_new = A2_in;

A1_sample_long = zeros(N,K1+1,C);
A2_sample_long = zeros(N,K2+1,C);



ber_2 = zeros(N, K2);
ber_1 = zeros(N, K1);
q = zeros(N, K2);
p = zeros(N, K1);


burn = 5;
while iter_indicator
 %% Gibbs step for binary latent variables
    A1_old= A1_new;
    A2_old = A2_new;


    parfor i = 1:N

        A1_i = A1_old(i,:);
        A2_i = A2_old(i,:);

        % -------------------------
        % Update A2_i(l) | rest
        % -------------------------
        for l = 1:K2

            A2_tmp1 = A2_i;
            A2_tmp0 = A2_i;

            A2_tmp1(l) = 1;
            A2_tmp0(l) = 0;

            eta_1 = B2 * [1, A2_tmp1]';   % K1 x 1
            eta_0 = B2 * [1, A2_tmp0]';   % K1 x 1

            logit_p2 = temp*(log(prop(l)/(1-prop(l))) + ...
                sum(A1_i' .* B2(:,l+1)) - ...
                sum(log((1 + exp(eta_1)) ./ (1 + exp(eta_0)))));


            p2 = 1 / (1 + exp(-logit_p2));
            q(i,l) = p2;

            % Gibbs draw
            A2_i(l) = rand < p2;
        end

        % -------------------------
        % Update A1_i(k) | rest
        % -------------------------
        for k = 1:K1

            A1_minus_k = A1_i(1:end ~= k);

            mu_minus_k = B1(:, 1:end~=(k+1)) * [1; A1_minus_k(:)];

            logit_p1 = temp*([1, A2_i] * B2(k,:)' ...
                - 0.5 * sum((B1(:,k+1).^2 - 2*(Z(i,:)' - mu_minus_k).*B1(:,k+1)) ./ gamma));

            p1 = 1 / (1 + exp(-logit_p1));
            p(i,k) = p1;

            % Gibbs draw
            A1_i(k) = rand < p1;
        end

        A1_old(i,:) = A1_i;
        A2_old(i,:) = A2_i;
    end


    for c = 1:C
        A2_sample_long(:,:,c) = [ones(N,1), A2_old];
        A1_sample_long(:,:,c) = [ones(N,1), A1_old];
    end

     mu = mean(A1_sample_long,3) * B1.';

    parfor j = 1:J
    
        mu_j = mu(:,j);
        sd_j = sqrt(gamma(j)/temp);
    
        Zj = Z(:,j);   % local copy
    
        % missing values
        ir_na = find(isnan(R(:,j)));
        Zj(ir_na) = mu_j(ir_na) + sd_j * randn(numel(ir_na),1);
    
        for r = 1:Rlevels(j)
    
            ir = find(R(:,j)==r & ~isnan(R(:,j)));
            if isempty(ir), continue; end
    
            % lower bound
            lbCand = Zj(R(:,j)==(r-1));
            lbCand = lbCand(~isnan(lbCand));
            if isempty(lbCand)
                lb = -inf;
            else
                lb = max(lbCand);
            end
    
            % upper bound
            ubCand = Zj(R(:,j)==(r+1));
            ubCand = ubCand(~isnan(ubCand));
            if isempty(ubCand)
                ub = inf;
            else
                ub = min(ubCand);
            end
    
            % probabilities
            pL = normcdf(lb, mu_j(ir), sd_j);
            pU = normcdf(ub, mu_j(ir), sd_j);
    
            eps0 = 1e-12;
            pL = max(min(pL,1-eps0),eps0);
            pU = max(min(pU,1-eps0),eps0);
    
            if pU <= pL
                mid = (pL+pU)/2;
                pL = mid; 
                pU = mid;
            end
    
            % sample truncated normal
            u = pL + (pU-pL).*rand(numel(ir),1);
            Zj(ir) = norminv(u, mu_j(ir), sd_j);
    
        end
    
        Z(:,j) = Zj;   % write back
    end



    %% Stochastic approximation M-step
    % prop
    prop_update = sum(q, 1) /N;


    parfor j = 1:J
        Zj = Z(:,j);
        f_loglik = F_1_SAEM_cop(Zj, A1_sample_long, N, K1, C,temp, gamma(j));

        f_j = @(x) f_loglik(x)
        opt = fmincon(f_j, B1(j,:), AA, bb, Aeq, beq, lb_1, ub_1, nonlcon, options);
        B1_update(j,:) = opt;
        eta = reshape(sum(bsxfun(@times, A1_sample_long, reshape(opt, [1 K1 + 1 1])), 2), [N C]);
        gamma_update(j) = sum((Zj - eta).^2, 'all')/(N*C);

    end
    % B2 
    parfor k = 1:K1
        A1_sample_k = reshape(A1_sample_long(:,k+1,:), [N C]);
        f_loglik = F_2_SAEM_cop(A1_sample_k, A2_sample_long, N, K2, C, temp);

        f_k = @(x) f_loglik(x) 
        B2_update(k,:) = fmincon(f_k, B2(k,:), AA, bb, Aeq, beq, lb_2, ub_2, nonlcon, options);
    end
    

    %% compute error
    err = sqrt(norm(prop - prop_update, "fro")^2 + norm(B1 - B1_update, "fro")^2 + norm(B2 - B2_update, "fro")^2);

       

    A1_new = double(mean(A1_sample_long(:,2:end,:), 3) > 0.5); 
    A2_new = double(mean(A2_sample_long(:,2:end,:), 3) > 0.5);

    prop = prop_update;

    gamma = gamma_update;
    
    %% set all zero coefficeints to 0
    B1_update(G1 == 0) = 0;
    B2_update(G2 == 0) = 0;

    B1 = B1_update;
    B2 = B2_update;

    t = t + 1;

    if t > burn
        temp = min(1, temp + 0.01*((t - burn)-1)); %% raise the temperature
    end
    if mod(t,5) == 0
        fprintf('EM Iteration %d,\t', t);
 
    end

    iter_indicator = ( abs(err) > tol & t < it);
end