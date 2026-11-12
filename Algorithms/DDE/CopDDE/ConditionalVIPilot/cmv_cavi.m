function [q,r,d]=cmv_cavi(logw,bits,old)
% Exact-enumeration conditional mean-field sequential coordinate ascent.
clock=tic; [N,C]=size(logw); m=size(bits,2); assert(C==2^m && isequal(size(old),[N m]));
logp=logw-max(logw,[],2); logp=logp-log(sum(exp(logp),2));
r=min(.999,max(.001,double(old))); q=product(r,bits); initial=elbo(q,logp);
current=initial; done=false(N,1); sweeps=zeros(N,1); change=zeros(N,1); history=nan(N,26); history(:,1)=current;
for sweep=1:25
 before=r; previous=current;
 for j=1:m
  iz=find(bits(:,j)==0); io=iz+2^(m-j); others=setdiff(1:m,j);
  weight=product(r(:,others),bits(iz,others));
  delta=sum(weight.*(logp(:,io)-logp(:,iz)),2); next=dde_logistic(delta);
  candidate=r; candidate(~done,j)=next(~done); qc=product(candidate,bits); value=elbo(qc,logp);
  assert(all(value>=current-2e-10*(1+abs(current))),'CMVI:CoordinateELBODecrease');
  r=candidate; current=value;
 end
 changeNow=max(abs(r-before),[],2); sweeps(~done)=sweep; change(~done)=changeNow(~done);
 assert(all(current>=previous-2e-10*(1+abs(previous))),'CMVI:SweepELBODecrease');
 history(:,sweep+1)=current;
 done=done | changeNow<1e-5 | abs(current-previous)./(1+abs(previous))<1e-7;
 if all(done), break; end
end
q=product(r,bits);
d=struct('sweeps',sweeps,'initial_elbo',initial,'final_elbo',current,'max_change',change,'capped',~done,'history',history(:,1:sweep+1),'seconds',toc(clock));
assert(all(r>=0 & r<=1,'all') && max(abs(sum(q,2)-1))<1e-12);
end
function q=product(r,bits)
if isempty(r), q=ones(size(r,1),size(bits,1)); return; end
lp=log(max(r,realmin))*bits'+log(max(1-r,realmin))*(1-bits)';
q=exp(lp-max(lp,[],2)); q=q./sum(q,2);
end
function value=elbo(q,logp)
value=sum(q.*(logp-log(max(q,realmin))),2);
end
