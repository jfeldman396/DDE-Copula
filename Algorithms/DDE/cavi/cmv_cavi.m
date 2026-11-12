function [q,r,d]=cmv_cavi(logw,bits,old,options)
% Reverse-KL Bernoulli mean field for an enumerated tempered block.
% logw MUST already include beta on all log-joint terms. Sequential exact
% coordinate maximization finds a local stationary optimum, not a certified
% global optimum. Finite-support coordinates must be removed upstream.
if nargin<4, options=struct; end
if ~isfield(options,'max_sweeps'),options.max_sweeps=10000;end
if ~isfield(options,'elbo_tolerance'),options.elbo_tolerance=1e-10;end
if ~isfield(options,'stationarity_tolerance'),options.stationarity_tolerance=1e-7;end
clock=tic; [N,C]=size(logw); m=size(bits,2);
assert(m>=1 && C==2^m && isequal(size(old),[N m]));
assert(isequal(bits,double(dec2bin(0:C-1,m)-'0')),'CMVI:StateOrder','States must have binary ascending order.');
assert(all(isfinite(logw),'all'),'CMVI:Support','Remove deterministic coordinates before CAVI.');
shift=max(logw,[],2); centered=logw-shift;
logZ=shift+log(sum(exp(centered),2)); logp=logw-logZ;
% Start at existing CP law, so the final ELBO cannot be worse than CP.
powers=2.^(m-1:-1:0); code=double(old)*powers'; r=zeros(N,m);
iz=cell(1,m); io=cell(1,m); other=cell(1,m);
for j=1:m
 iz{j}=find(bits(:,j)==0); io{j}=iz{j}+powers(j); other{j}=setdiff(1:m,j);
 zero=code-old(:,j)*powers(j); one=zero+powers(j);
 r(:,j)=sigmoid(logw((1:N)'+N*one)-logw((1:N)'+N*zero));
end
q=product(r,bits); initial=elbo(q,logp); current=initial;
done=false(N,1); sweeps=zeros(N,1); change=zeros(N,1);
history=nan(N,min(options.max_sweeps,32)+1); history(:,1)=initial;
for sweep=1:options.max_sweeps
 previous=current; before=r;
 for j=1:m
  weight=product(r(:,other{j}),bits(iz{j},other{j}));
  delta=sum(weight.*(logw(:,io{j})-logw(:,iz{j})),2);
  candidate=sigmoid(delta); r(~done,j)=candidate(~done);
 end
 q=product(r,bits); current=elbo(q,logp);
 assert(all(current>=previous-1e-10*(1+abs(previous))),'CMVI:ELBODecrease');
 residual=zeros(N,1);
 for j=1:m
  weight=product(r(:,other{j}),bits(iz{j},other{j}));
  optimum=sigmoid(sum(weight.*(logw(:,io{j})-logw(:,iz{j})),2));
  residual=max(residual,abs(optimum-r(:,j)));
 end
 change(~done)=max(abs(r(~done,:)-before(~done,:)),[],2); sweeps(~done)=sweep;
 history(:,sweep+1)=current;
 done=done | (abs(current-previous)<=options.elbo_tolerance*(1+abs(previous)) & residual<=options.stationarity_tolerance);
 if all(done),break;end
end
assert(all(done),'CMVI:Nonconvergence','CAVI did not converge; increase max_sweeps, do not silently accept a cap.');
d=struct('sweeps',sweeps,'initial_elbo',initial,'final_elbo',current, ...
 'max_change',change,'capped',~done,'history',history(:,1:sweep+1), ...
 'stationarity_residual',residual,'log_normalizer',logZ, ...
 'initial_kl',-initial,'final_kl',-current,'seconds',toc(clock));
assert(max(abs(sum(q,2)-1))<1e-12 && all(current>=initial-1e-10));
end
function r=sigmoid(x)
r=zeros(size(x)); pos=x>=0; r(pos)=1./(1+exp(-x(pos))); e=exp(x(~pos));r(~pos)=e./(1+e);
end
function q=product(r,bits)
q=ones(size(r,1),size(bits,1));
for j=1:size(bits,2),q=q.*(r(:,j)*bits(:,j)'+(1-r(:,j))*(1-bits(:,j))');end
end
function value=elbo(q,logp)
value=sum(q.*(logp-log(max(q,realmin))),2);
end
