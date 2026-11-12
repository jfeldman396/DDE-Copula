function [theta,v,c,prob]=gfc_csp(L,theta,v,alpha,a,b,spike)
% Legramanti et al. (2020), Algorithm 1, finite-H cumulative shrinkage prior.
[J,H]=size(L);omega=v.*[1 cumprod(1-v(1:end-1))];
c=zeros(1,H);prob=zeros(H,H);ss=sum(L.^2,1);
for h=1:H
 normal=-.5*J*log(2*pi*spike)-ss(h)/(2*spike);
 % Multivariate Student t obtained by integrating theta~IG(a,b).
 slab=gammaln(a+J/2)-gammaln(a)-J/2*log(2*pi*b)-(a+J/2)*log1p(ss(h)/(2*b));
 lp=log(max(omega,realmin))+[normal*ones(1,h) slab*ones(1,H-h)];
 p=exp(lp-max(lp));p=p/sum(p);prob(h,:)=p;c(h)=find(rand<cumsum(p),1);
 if c(h)<=h,theta(h)=spike;else,theta(h)=1/gamrnd(a+J/2,1/(b+ss(h)/2));end
end
for h=1:H-1,v(h)=betarnd(1+sum(c==h),alpha+sum(c>h));end
v(H)=1;
assert(all(theta>0 & isfinite(theta)) && max(abs(sum(prob,2)-1))<1e-12);
end
