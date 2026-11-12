function a=cpdde_accumulate(a,A1,A2,rho)
% Exact one-state objective history for a genuinely absent upper layer.
% Nonempty-top models delegate unchanged to the frozen helper.
if size(A2,2)>0,a=foa_accumulate(a,A1,A2,rho);return;end
assert(size(A1,1)==size(A2,1)&&rho>0&&rho<=1);
N=size(A1,1);T=sum(A1,1);
if isempty(a)||rho==1
 a=struct('config',zeros(1,0),'N',N,'T',T,'top',zeros(1,0));
else
 assert(isequal(size(a.config),[1 0])&&isequal(size(a.T),size(T)));
 a.N=(1-rho)*a.N+rho*N;a.T=(1-rho)*a.T+rho*T;
end
assert(abs(a.N-N)<1e-8*(1+N)&&all(a.T>=0)&&all(a.T<=a.N+1e-10));
end
