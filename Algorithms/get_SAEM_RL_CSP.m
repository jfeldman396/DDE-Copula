
function [num_act1,num_act2, theta, theta2,gamma,p,q,A1_sample_long, A2_sample_long,Z,prop, B1, B2, A1_new, A2_new, it_num] = get_SAEM_RL_CSP(X, Z, R, Rlevels, prop_in, gamma_in,B1_in, B2_in, A1_in, A2_in, C, it,t1,t2,temp,tau)

%% set latent dimensions
K2 = size(prop_in,2);
[J, K1] = size(B1_in);
K1 = K1-1;
H = K1;
N = size(X, 1);

%% Initialize DDE parameters
prop = prop_in;
B1 = B1_in;
B2 = B2_in;
gamma = gamma_in.';

%% Initialize CUSP parameters
theta_inf = t1;
theta_inf2 = t2;
theta_slab = ones(K1,1);
theta_slab2 = ones(K2,1);
v = rand(K1+1,1);
v2 = rand(K2+1,1);
omega = (1/(K1+1))*ones(K1+1,1);
omega2 = (1/(K2 + 1))*ones(K2+1,1);
z_layer2 = floor(K2/2)*ones(1, K2);   
num_act1 = K1;

% iteration settings
t = 1;                 
iter_indicator = true;
it_num = 0;

window = 7; % for ending algorithm
burn = 5;
rel_hist1 = []; rel_hist2= [];
 
%updates
prop_update = prop_in;

B1_update = zeros(J, K1+1);
B2_update = zeros(K1, K2+1);

A1_new = A1_in;
A2_new = A2_in;

A1_sample_long = zeros(N,K1+1,C);
A2_sample_long = zeros(N,K2+1,C);


ber_2 = zeros(N, K2);
ber_1 = zeros(N, K1);

q = zeros(N, K2);
p = zeros(N, K1);




epsilon = .01;
% optimization settings
bb = []; Aeq = []; beq = [];
AA = [];
lb_1 = [-10*ones(1,1), -10*ones(1, K1)]; % zeros(J, K1+1);
ub_1 = 10*ones(1, K1+1);
lb_2 = [-5*ones(1,1), -10*ones(1, K2)]; % zeros(J, K1+1);
ub_2 = 10*ones(1, K2+1);
options = optimset('Display', 'off', 'MaxIter', 50);



% optimization functions
f_old_1 = cell(J,1);
for j = 1:J
    f_old_1{j} = @(x) 0;
end
f_old_2 = cell(K1,1);
for k = 1:K1
    f_old_2{k} = @(x) 0;
end






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


  
    %% Rank likelihood simulation step
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



    %% Column activity MAP under CUSP: layer2

   
    L = K2+1;     % truncation level including remainder

    B2_pad = [B2, zeros(K1,1)]; % pad for error avoidance

    prob_mat2 = zeros(K2,K2+1);


  
    parfor h = 1:K2
    
        log_probs = -inf(1, L);
    
        % Column h of B2 (K1 x 1)
        xh = B2_pad(:, h+1);
        log_dens_spike = -num_act1 * log(2 * theta_inf2)  - sum(abs(xh)) / theta_inf2; % density at spike
        sum_abs_x = sum(abs(xh));
        log_dens_slab = 1*log(5) - num_act1*log(2) - gammaln(1) + gammaln(1+ num_act1) ...
               - (1 +num_act1) * log(5 + sum_abs_x); % marginal density over Gamma(1,5) prior
    
    
        for l = 1:L
    
            % spike for l<=h, slab for l>h 
            if l <= h
                log_col = log_dens_spike;
            else
                log_col = log_dens_slab;
            end
     
            % --- full unnormalized log-probability for z_h = l
            log_probs(l) = log(omega2(l)) + log_col ;
        end
    
        % Stable normalize and sample
        mx    = max(log_probs);
        probs = exp(log_probs - mx);
        probs = probs / sum(probs);
        prob_mat2(h,:) = probs;
    end

    [idx,z_layer2] = max(prob_mat2,[],2); %% MAP estimate of activation

    z_layer2 = z_layer2.';


    %% Gibbs sampling for CUSP layer 2 parameters

    % 5) Update sticks and omega for Layer 2 
    
    alpha2 = K2;                 
    L = K2 + 1;
    
    v2 = zeros(1, L);
    omega2 = zeros(1, L);
    

    for l = 1:K2
        n_l   = sum(z_layer2(1:K2) == l);
        n_gtl = sum(z_layer2(1:K2) >  l);
        
        v2(l) = betarnd(1 + n_l, alpha2 + n_gtl);

    end
    
    % remainder stick
    v2(L) = 1;
    
    % compute omega2
    omega2(1) = v2(1);
    for l = 2:L
        omega2(l) = v2(l) * prod(1 - v2(1:l-1));
    end

    % sample slabs for active columns
    for h = 1:K2
        if z_layer2(h)>h
            shape_post = 1 + num_act1;
            rate_post  = 5+ sum(abs(B2(:,h+1)));   % Jx1
            theta_slab2(h) = 1/gamrnd(shape_post, 1 ./ rate_post);  % MATLAB: shape, scale

        end
    end


    % 6) Update variances theta2 based on z_layer2 -- to be used 
    theta2 = zeros(K1,K2);
    for h = 1:K2
        % 
        if z_layer2(h) <= h
            theta2(:,h) = theta_inf2;
        else
            theta2(:,h) = theta_slab2(h);
        end
    end


    %% Column activity MAP under CUSP: layer1

    prob_mat1 = zeros(K1,K1+1);
    parfor h = 1:K1
        log_probs = -inf(1, H+1);
        x = B1(:, h+1);      % J x 1, the h-th loading column
    
        for l = 1:K1+1
            if l <= h
                % spike 
                log_dens = -J*log(2*theta_inf) - sum(abs(x))/theta_inf;
            else
                % slab
                sum_abs_x = sum(abs(x));
                log_dens = 1*log(5) - J*log(2) - gammaln(1) + gammaln(1+ J) ...
           - (1+ J) * log(5 + sum_abs_x);
            end
        
            log_probs(l) = log(omega(l)) + log_dens;
        end
    
        probs = exp(log_probs - max(log_probs));

        probs = probs / sum(probs);
        prob_mat1(h,:) = probs;
        
      
    end




    [idx,z] = max(prob_mat1,[],2); %% MAP
        
    %% Gibbs sampling for CUSP layer 2 parameters
    alpha = K1;
    v = zeros(1, H+1);
    for l = 1:H
        n_l   = sum(z == l);
        n_gtl = sum(z >  l);
    
      
        v(l) = betarnd(1 + n_l, alpha + n_gtl);
    
    end
    v(H+1) = 1;


    omega = zeros(1, H+1);
    omega(1) = v(1);
    for l = 2:H+1
        omega(l) = v(l) * prod(1 - v(1:l-1));
    end

    
    %% sample slabs
      for h = 1:H
            if z(h) > h
                shape_post = 1+ J;
                rate_post  = 5 + sum(abs(B1(:,h+1)));   % Jx1
                theta_slab(h) = 1/gamrnd(shape_post, 1 ./ rate_post);  % MATLAB: shape, scale
    
            end
      end


    % store penalties
    theta = zeros(J, H);
    for h = 1:H
        % theta(:,h) = sum(prob_mat1(h,h+1:end))*theta_slab(h) + sum(prob_mat1(h,1:h))*theta_inf;
        if z(h) <= h
            % In the spike: precision is the inverse of the point mass
    
            theta(:,h) = theta_inf;
        else
    
            theta(:,h) = theta_slab(h);
        end
    end






    %% MAP estimate for highest level latent binaries (flat prior)
    
    prop_update = sum(q, 1) /N;


    
    %% MAP estimate for B1 and gamma
    B1_update = zeros(J, K1+1);
    gamma_update = zeros(J, 1);
    f_old_1_new = cell(J,1);   % if you need to retain these handles
    
    parfor j = 1:J
    
        Zj = Z(:,j);
    
        % local log-likelihood handle
        f_loglik = F_1_SAEM_cop(Zj, A1_sample_long, N, K1, C, temp, gamma(j));
        f_old_1_new{j} = @(x) f_loglik(x);
    
        W1 = [0, 1 ./ theta(j,:)];
    
        %%% Weighted lasso
        ftn_adaptive_group_pen_1 = @(x) sum(W1(2:end) .* abs(x(2:end)));
        f_j = @(x) f_loglik(x) + ftn_adaptive_group_pen_1(x);
    
        opt = fmincon(f_j, B1(j,:), AA, bb, Aeq, beq, lb_1, ub_1, nonlcon, options);
        B1_update(j,:) = opt;
    
        eta = reshape( ...
            sum(bsxfun(@times, A1_sample_long, reshape(opt, [1, K1+1, 1])), 2), ...
            [N, C] ...
        );
    
        SSE = sum((Zj - eta).^2, 'all');
        gamma_update(j) = (1 + 0.5*SSE) / (1 + 0.5*N*C + 1); %% MAP for gamma under a gamma(1,1) prior
    end
    

    
    
  
    %% MAP esitmate for B2
    B2_update = zeros(K1, K2+1);
    f_old_2_new = cell(K1,1);  % if you need to retain these handles
    
    lb_2 = [-5, -10*ones(1, K2)];
    ub_2 = 10*ones(1, K2+1);
    
    parfor k = 1:K1
    
        A1_sample_k = reshape(A1_sample_long(:,k+1,:), [N, C]);
    
    
        f_loglik = F_2_SAEM_cop(A1_sample_k, A2_sample_long, N, K2, C, temp);
        f_old_2_new{k} = @(x) f_loglik(x);
    
        W2 = [0, 1 ./ theta2(k,:)];
    
        ftn_adaptive_group_pen_2 = @(x) sum(W2(2:end) .* abs(x(2:end)));
        f_k = @(x) f_loglik(x) + ftn_adaptive_group_pen_2(x);
    
        opt = fmincon(f_k, B2(k,:), AA, bb, Aeq, beq, lb_2, ub_2, nonlcon, options);
    
        B2_update(k,:) = opt;
    end
    



        
    it_num = it_num + 1;

    %% Compute MAP estimate of active nodes in each layer
    num_act1 = sum(z.' >1:K1);
    num_act2 = sum(z_layer2 >1:K2);
 
    %% Extract active indices in weight matrices
    ind_inact2 = find(z_layer2 <=1:K2);
    ind_inact1 = find(z(1:K1).' <= 1:K1);
   
    %% Threshold entries
    B2_update(:,2:end) = thres(B2_update(:,2:end),tau);
    B1_update(:,2:end) = thres(B1_update(:,2:end),tau);

    %% hierarchical gating
    B2_update(ind_inact1 ,:) = 0;
   
    if it_num > burn
        temp = min(1, temp + 0.01*((it_num - burn)-1)); %% raise the temperature
    end
    

 

    
   


   
    %% Check whether to terminate based on rolling window of relative Frobenius norm
    rel_err1 = norm(B1_update(:,2:end) - B1(:,2:end), 'fro') / (norm(B1(:,2:end), 'fro') + 1e-6);
    rel_err2 = norm(B2_update(:,2:end) - B2(:,2:end), 'fro') / (norm(B2(:,2:end), 'fro') + 1e-6);
    rel_hist1 = [rel_hist1 rel_err1];
    rel_hist2 = [rel_hist2 rel_err2];


    if length(rel_hist1) >= window
            recent1 = rel_hist1(end-window+1:end);
            recent2 = rel_hist2(end-window+1:end);

   
        
        if std(recent1) < .01 && std(recent2) < .01
    
            A1_new = double(mean(A1_sample_long(:,2:end,:), 3) > 0.5); 
            A2_new = double(mean(A2_sample_long(:,2:end,:), 3) > 0.5);

            prop = prop_update;

            B1_update(:,ind_inact1 + 1) = 0;
            B2_update(:,ind_inact2 + 1) = 0;
            B1 = B1_update;
            B2 = B2_update;
            gamma = gamma_update;
            disp("ending algorithm")
       
            iter_indicator = false;
            break

        end
    end

            



    %% Update parameters
    A1_new = double(mean(A1_sample_long(:,2:end,:), 3) > 0.5); 
    A2_new = double(mean(A2_sample_long(:,2:end,:), 3) > 0.5);


    prop = prop_update;
    B1 = B1_update;
    B2 = B2_update;
    gamma = gamma_update;

    
    
     

   %% end for iteration limit.
    if iter_indicator
        iter_indicator = it_num <it;
        if ~iter_indicator
                B1(:,ind_inact1 + 1) = 0;
                B2(:,ind_inact2 + 1) = 0;
            display("Ending algorithm for iteration budget")
        end
    end 
end



end 