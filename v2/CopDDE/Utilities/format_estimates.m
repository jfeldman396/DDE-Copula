function[B1_est,B2_est, gamma_est, prop_est, assignment1, assignment2] = format_estimates(B1_true_unscale, B2_true, B1_final, B2_final, gamma_final, prop_final, scale,Z,emp) 
    
    [J,K1] = size(B1_true_unscale(:,2:end)); 
  
    costmat = zeros(K1,K1); B1_est = B1_final(:,2:end); 
    for k = 1:K1 
        costmat(k,:) = - sum(B1_final(find(B1_true_unscale(:,k+1) > 0), 2:end), 1); 
    end 
    [assignment1, ~] = munkres(costmat); 
    B1_est = [B1_final(:,1),B1_est(:, assignment1)];
    
    K2 = size(B2_true(:,2:end),2); 
    
    costmat = zeros(K2, K2);
    B2_est = B2_final(assignment1,2:end);
    for k = 1:K2 
        costmat(k,:) = - sum(B2_est(find(B2_true(:,k+1) > 0), :), 1); 
    end 
    [assignment2, ~] = munkres(costmat);
    B2_est = [B2_final(assignment1,1),B2_est(:, assignment2)];
   
    if length(assignment2) == length(prop_final)
        prop_est = prop_final(assignment2);
    else
        prop_est = NaN;
    end
    gamma_est = ones(J,1);
    if scale 
        [B1_est, gamma_est]= rescale_B1(prop_final,B2_est,B1_est, gamma_final,emp,Z);
        
    end 
    
end