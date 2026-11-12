function pred=r1t_predict(f,Ytrain,m,seed)
% Exact finite-mixture marginal CDF, not random ranks of generated batches.
rng(seed,'twister');D=numel(f.B);J=size(f.B{1},1);A=cell(1,D);A{D}=double(rand(m,numel(f.prop))<f.prop);
for d=D-1:-1:1,P=dde_logistic([ones(m,1),A{d+1}]*f.B{d+1}');A{d}=double(rand(size(P))<P);end
Z=[ones(m,1),A{1}]*f.B{1}'+randn(m,J).*sqrt(f.gamma(:)');
% Enumerate layers from top; marginalize every upper layer exactly.
K=cellfun(@(b)size(b,2)-1,f.B);S=double(dec2bin(0:2^K(D)-1,K(D))-'0');w=prod(f.prop.^S.*(1-f.prop).^(1-S),2);
for d=D-1:-1:1
 T=double(dec2bin(0:2^K(d)-1,K(d))-'0');P=dde_logistic([ones(size(S,1),1),S]*f.B{d+1}');W=ones(size(S,1),size(T,1));
 W=exp(log(P)*T'+log1p(-P)*(1-T)');
 w=W'*w;S=T;
end
assert(abs(sum(w)-1)<1e-10&&all(w>=0));mu=[ones(size(S,1),1),S]*f.B{1}';U=zeros(m,J);
for j=1:J,U(:,j)=normcdf((Z(:,j)-mu(:,j)')./sqrt(f.gamma(j)))*w;end
U=min(max(U,eps),1-eps);Y=zeros(m,J);n=size(Ytrain,1);
for j=1:J,s=sort(Ytrain(:,j));Y(:,j)=s(max(1,ceil(n*U(:,j))));end
pred=struct('Y',Y,'U',U,'Z',Z,'bottom_states',S,'bottom_weights',w,'seed',seed);
end
