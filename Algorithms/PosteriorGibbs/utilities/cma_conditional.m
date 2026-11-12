function value=cma_conditional(x,a,b,mu1,c1,mu2,c2,gamma,last)
if ~(a<x&&x<b),value=-Inf;return;end
value=cma_logh(a,x,mu1,c1,gamma);
if last,value=value+sum(c2.*agr_log_interval((x-mu2)/sqrt(gamma),Inf(size(mu2))));else,value=value+cma_logh(x,b,mu2,c2,gamma);end
end
