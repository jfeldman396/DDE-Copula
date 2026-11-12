function y = dde_softplus(x)
%DDE_SOFTPLUS Stable log(1+exp(x)), including infinite endpoints.
y = max(x,0) + log1p(exp(-abs(x)));
end
