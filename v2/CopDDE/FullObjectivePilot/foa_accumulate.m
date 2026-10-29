function a=foa_accumulate(a,A1,A2,rho)
% Exact finite-support likelihood history, with all absent cells decayed.
assert(rho>0&&rho<=1&&all(A2==0|A2==1,'all'));
[config,~,g]=unique(A2,'rows');
N=accumarray(g,1,[size(config,1),1]);T=zeros(size(config,1),size(A1,2));
for k=1:size(A1,2),T(:,k)=accumarray(g,A1(:,k),[size(config,1),1]);end
if isempty(a)||rho==1
 a=struct('config',config,'N',N,'T',T,'top',sum(A2,1));
else
 assert(size(a.config,2)==size(A2,2)&&size(a.T,2)==size(A1,2));
 [u,~,ix]=unique([a.config;config],'rows');old=ix(1:size(a.config,1));cur=ix(size(a.config,1)+1:end);
 n=zeros(size(u,1),1);t=zeros(size(u,1),size(A1,2));
 n(old)=(1-rho)*a.N;t(old,:)=(1-rho)*a.T;
 n(cur)=n(cur)+rho*N;t(cur,:)=t(cur,:)+rho*T;
 a=struct('config',u,'N',n,'T',t,'top',(1-rho)*a.top+rho*sum(A2,1));
end
assert(all(a.N>0)&&all(a.T>=0,'all')&&all(a.T<=a.N+1e-10,'all'));
assert(abs(sum(a.N)-size(A2,1))<1e-8*(1+size(A2,1)));
end
