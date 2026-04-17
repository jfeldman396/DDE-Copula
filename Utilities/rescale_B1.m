function [B1_scaled, gamma_scaled, mu_used, sd_used] = rescale_B1(prop_true, B2, B1, gamma_vec, emp, Z)
% RESCALE_B1 Rescale rows of B1 so the implied marginal Z_j has mean 0 and variance 1.
%
% If emp == false:
%   Uses model-implied marginal mean and variance by summing over all
%   configurations of A2 and A1.
%
% If emp == true:
%   Uses empirical marginal means and standard deviations from Z.
%
% Inputs:
%   prop_true : K2 x 1 or 1 x K2 vector of top-layer Bernoulli probs
%   B2        : K1 x (K2+1) matrix for A1 | A2 logistic model
%   B1        : p x (K1+1) matrix for Z | A1 Gaussian model
%   gamma_vec : p x 1 vector of conditional variances for each Z_j
%   emp       : logical; if true use empirical mean/sd from Z
%   Z         : N x p matrix of latent Gaussian draws (used only if emp=true)
%
% Outputs:
%   B1_scaled    : p x (K1+1) rescaled loading matrix
%   gamma_scaled : p x 1 rescaled conditional variances
%   mu_used      : p x 1 vector of means used in rescaling
%   sd_used      : p x 1 vector of sds used in rescaling

    K2 = length(prop_true);
    K1 = size(B2,1);
    p  = size(B1,1);

  

    gamma_vec = gamma_vec(:);

    if emp
        if nargin < 6 || isempty(Z)
            error('When emp=true, you must supply Z.');
        end
        if size(Z,2) ~= p
            error('Z must have p columns, where p = size(B1,1).');
        end

        % Empirical marginal moments of Z
        mu_used = mean(Z, 1, 'omitnan')';
        sd_used = std(Z, 0, 1, 'omitnan')';

    else
        % Enumerate configs
        A2_configs = dec2bin(0:(2^K2 - 1)) - '0';
        A1_configs = dec2bin(0:(2^K1 - 1)) - '0';

        mu_used = zeros(p,1);
        second_moment = zeros(p,1);

        for i2 = 1:size(A2_configs,1)
            A2 = A2_configs(i2,:)';
            pA2 = prod(prop_true(:).^A2(:) .* (1 - prop_true(:)).^(1 - A2(:)));

            logits = B2 * [1; A2];
            pA1_given_A2 = 1 ./ (1 + exp(-logits));

            for i1 = 1:size(A1_configs,1)
                A1 = A1_configs(i1,:)';
                pA1 = prod(pA1_given_A2.^A1 .* (1 - pA1_given_A2).^(1 - A1));
                pJoint = pA2 * pA1;

                for j = 1:p
                    mu_jA1 = B1(j,:) * [1; A1];
                    var_j  = gamma_vec(j);

                    mu_used(j) = mu_used(j) + pJoint * mu_jA1;
                    second_moment(j) = second_moment(j) + pJoint * (mu_jA1^2 + var_j);
                end
            end
        end

        varZ = second_moment - mu_used.^2;
        sd_used = sqrt(varZ);
    end

    % Guard against zero or tiny standard deviations
    sd_used(sd_used < 1e-12) = 1;

    % Rescale each row of B1
    B1_scaled = zeros(size(B1));
    for j = 1:p
        B1_scaled(j,:) = (B1(j,:) - [mu_used(j), zeros(1,K1)]) / sd_used(j);
    end

    % Variance rescales by sd^2
    gamma_scaled = gamma_vec ./ (sd_used.^2);

end