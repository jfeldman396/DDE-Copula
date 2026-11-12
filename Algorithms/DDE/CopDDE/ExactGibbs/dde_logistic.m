function p = dde_logistic(x)
%DDE_LOGISTIC Overflow-safe inverse logit.
p = zeros(size(x));
positive = x>=0;
p(positive) = 1 ./ (1+exp(-x(positive)));
ex = exp(x(~positive));
p(~positive) = ex ./ (1+ex);
end
