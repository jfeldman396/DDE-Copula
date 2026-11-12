function [A,diagnostic]=r1t_binary(A,Z,B,gamma,prop,blocks,beta,lambda,record,method)
% Arbitrary depth, exact enumerated BLOCK conditional at beta=lambda=1.
% Every layer is visited top-to-bottom; own state-independent row permutation.
if nargin<9,record=false;end;if nargin<10,method='CP';end
D=numel(A);n=size(Z,1);diagnostic=struct('blocks',{{}},'beta',beta,'lambda',lambda);
for d=D:-1:1
 K=size(A{d},2);[~,order]=sort(rand(n,K),2);
 if d>1,childEta=[ones(n,1),A{d}]*B{d}';else,childEta=[ones(n,1),A{1}]*B{1}';end
 if d<D,parentEta=[ones(n,1),A{d+1}]*B{d+1}';else,parentEta=repmat(log(prop./(1-prop)),n,1);end
 for first=1:blocks(d):K
  if record,beforeA=A;end
  S=order(:,first:min(first+blocks(d)-1,K));m=size(S,2);bits=double(dec2bin(0:2^m-1,m)-'0');C=size(bits,1);old=A{d}((1:n)'+n*(S-1));eta0=childEta;
  for k=1:m,eta0=eta0-old(:,k).*B{d}(:,S(:,k)+1)';end
  logw=zeros(n,C);
  for c=1:C
   eta=eta0;
   for k=1:m,eta=eta+bits(c,k)*B{d}(:,S(:,k)+1)';end
   if d==1,logw(:,c)=-.5*sum((Z-eta).^2./gamma',2);else,logw(:,c)=sum(A{d-1}.*eta-dde_softplus(eta),2);end
   for k=1:m,h=parentEta((1:n)'+n*(S(:,k)-1));logw(:,c)=logw(:,c)+bits(c,k)*h;end
  end
  logw=beta*logw;P=exp(logw-max(logw,[],2));P=P./sum(P,2);draw=P;
  if lambda<1
   if strcmp(method,'CAVI'),[~,draw]=pilot_mix(logw,P,bits,old,lambda,'VI');else,[~,draw]=bl_mix(logw,P,bits,old,lambda,'CP');end
  end
  cum=cumsum(draw,2);cum(:,end)=1;chosen=1+sum((1-rand(n,1))>cum,2);next=bits(chosen,:);A{d}((1:n)'+n*(S-1))=next;
  for k=1:m,childEta=childEta+(next(:,k)-old(:,k)).*B{d}(:,S(:,k)+1)';end
  if record,diagnostic.blocks{end+1}=struct('layer',d,'S',S,'old',old,'bits',bits,'P',P,'logw',logw,'beforeA',{beforeA});end
 end
end
end
