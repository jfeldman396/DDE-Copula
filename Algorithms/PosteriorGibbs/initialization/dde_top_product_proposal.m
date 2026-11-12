function [proposed,logits]=dde_top_product_proposal(A1,A2,B2,prop,beta)
%DDE_TOP_PRODUCT_PROPOSAL Simultaneous independent Bernoulli proposals at OLD state.
logits=beta*dde_top_logodds(A1,A2,B2,prop);
proposed=double(rand(size(A2))<dde_logistic(logits));
end
