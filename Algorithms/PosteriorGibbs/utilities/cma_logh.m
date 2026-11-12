function v=cma_logh(a,b,mu,count,gamma)
if ~(a<b),v=-Inf;return;end
q=agr_log_interval((a-mu)/sqrt(gamma),(b-mu)/sqrt(gamma));lf=-.5*((b-mu).^2/gamma)-.5*log(2*pi*gamma);
if numel(mu)==1&&count==1,v=lf;return;end
h=log(count)+lf-q;t=max(h);v=sum(count.*q)+t+log(sum(exp(h-t)));assert(~isnan(v),'CMA:LoghNaN');
end
