function [gamma,p,q,A1_sample_long, A2_sample_long,Z,prop, B1, B2, A1_new, A2_new, t] = get_SAEM_RL(X, Z, R, Rlevels, prop_in, B1_in, B2_in, gamma_in, pen_1, pen_2, tau, A1_in, A2_in, C, tol, it, temp)

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
% optimization functions
ftn_pen_1 = @(x) pen_1 * TLP(x(2:end), tau);
ftn_pen_2 = @(x) pen_2 * TLP(x(2:end), tau);

burn = 5;
while iter_indicator
    A1_old = A1_new;
    A2_old = A2_new;

    %% Simulation step for Binary latent variables
    parfor i = 1:N            
        % variational step for A^{(d)}
        for l = 1:K2

            A2_i = A2_old(i,:);
            A2_i(l) = 1; eta_1 = B2 * [1, A2_i]'; % K1 x 1
            A2_i(l) = 0; eta_0 = B2 * [1, A2_i]';

            ber_2(i,l) = logistic(temp*(log(prop(l) / (1 - prop(l))) + ...
                (sum(A1_old(i,:)' .* B2(:,l+1)) - sum(log( (1 + exp(eta_1)) ./ (1 + exp(eta_0)))))));
        end


        for k = 1:K1                              
            A1_i = A1_old(i,:);
            ber_1(i,k) = logistic(temp*(([1, A2_old(i,:)] * B2(k,:)') - sum(( (B1(:,k+1).^2 -2*(Z(i,:)' - B1(:, 1:end~=(k+1)) * [1; A1_i(1:end~=k)']) ...
                .*B1(:,k+1) ) )./gamma) /2 ));
        end
    end


    % sample each latent variable

    A1_sample_long = zeros(N,K1+1,C);
    A2_sample_long = zeros(N,K2+1,C);
    for c = 1:C
        A2_sample_long(:,:,c) = [ones(N,1), double(rand(N,K2) < ber_2)];
        A1_sample_long(:,:,c) = [ones(N,1), double(rand(N,K1) < ber_1)];
    end

 
    for k = 1:K1
        p(:,k) = ber_1(:,k);
    end
    for l = 1:K2
        q(:,l) =  ber_2(:,l);
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
        % f_old_1{j} = @(x) (1-1/t) * f_old_1{j}(x) + 1/t * f_loglik(x);

        f_j = @(x) f_loglik(x)+ ftn_pen_1(x);
        opt = fmincon(f_j, B1(j,:), AA, bb, Aeq, beq, lb_1, ub_1, nonlcon, options);
        B1_update(j,:) = opt;
        eta = reshape(sum(bsxfun(@times, A1_sample_long, reshape(opt, [1 K1 + 1 1])), 2), [N C]);
        gamma_update(j) = sum((Zj - eta).^2, 'all')/(N*C);

    end
    % B2 
    parfor k = 1:K1
        A1_sample_k = reshape(A1_sample_long(:,k+1,:), [N C]);
        f_loglik = F_2_SAEM_cop(A1_sample_k, A2_sample_long, N, K2, C, temp);
        % f_old_2{k} = @(x) (1-1/t) * f_old_2{k}(x) + 1/t * f_loglik(x);
        f_k = @(x) f_loglik(x)  + ftn_pen_2(x);
        B2_update(k,:) = fmincon(f_k, B2(k,:), AA, bb, Aeq, beq, lb_2, ub_2, nonlcon, options);
    end
    

    %% compute error
    err = sqrt(norm(prop - prop_update, "fro")^2 + norm(B1 - B1_update, "fro")^2 + norm(B2 - B2_update, "fro")^2);

       

    A1_new = double(mean(A1_sample_long(:,2:end,:), 3) > 0.5); 
    A2_new = double(mean(A2_sample_long(:,2:end,:), 3) > 0.5);

    prop = prop_update;

    gamma = gamma_update;

    B1 = [B1_update(:, 1), thres(B1_update(:,2:end), tau)];
    B2 = [B2_update(:, 1), thres(B2_update(:,2:end), tau)];


    t = t + 1;

    if t > burn
        temp = min(1, temp + 0.01*((t - burn)-1)); %% raise the temperature
    end
    if mod(t,5) == 0
        fprintf('EM Iteration %d,\t', t);
 
    end

    iter_indicator = ( abs(err) > tol & t < it);
end