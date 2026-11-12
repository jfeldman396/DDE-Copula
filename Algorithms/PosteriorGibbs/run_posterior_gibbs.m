function run_posterior_gibbs(a,f,options,outputDir)
% Full-parameter Gibbs/slice sampler initialized from the full MAP state.
% a.R: observed ranks; f: B1,B2,gamma,Z,A1,A2,prop.
% options: seed,iterations,spike1,spike2,alpha1,alpha2.
% Use the seeds and settings recorded in the associated study protocol.
assert(java.io.File(outputDir).isAbsolute() && ~isfolder(outputDir) && ~isfile(outputDir));
oldpath=path;oldrng=rng;cleanup=onCleanup(@()restore_state(oldpath,oldrng));
addpath(fullfile(fileparts(mfilename('fullpath')),'utilities'));
rng(options.seed);iterations=options.iterations;
spike1=options.spike1;spike2=options.spike2;alpha1=options.alpha1;alpha2=options.alpha2;
N=size(a.R,1);d=outputDir;mkdir(d);
B=f.B1(:,2:end);b0=f.B1(:,1);gamma=f.gamma(:);C=f.B2;Z=f.Z;A=f.A1;U=f.A2;p=f.prop;
[J,K]=size(B);L=size(U,2);
assert(isequal(size(a.R),size(Z)) && all(isfinite(Z(:))));
assert(isequal([b0 B],f.B1)&&isequal(C,f.B2)&&isequal(gamma(:),f.gamma(:))&&isequal(p(:),f.prop(:)));
assert(isequal(A,f.A1)&&isequal(U,f.A2)&&isequal(Z,f.Z));
save(fullfile(d,'INITIAL.mat'),'B','b0','C','gamma','p','A','U','Z','spike1','spike2','alpha1','alpha2','-v7.3');
state1=dde_csp_init([b0 B],spike1,alpha1);state2=dde_csp_init(C,spike2,alpha2);drawB=zeros(J,K,iterations);drawC=zeros(K,L+1,iterations);drawp=zeros(iterations,L);drawgamma=zeros(iterations,J);drawb0=zeros(iterations,J);drawtheta1=zeros(iterations,K);drawtheta2=zeros(iterations,L);drawactive1=false(iterations,K);drawactive2=false(iterations,L);timer=tic;
for it=1:iterations
 state1=dde_csp_gibbs(state1,[b0 B]);state2=dde_csp_gibbs(state2,C);rate=repmat(state1.precision,J,1);rate2=repmat(state2.precision,K,1);
 for k=1:K
  residual=Z-b0'-A*B'+A(:,k)*B(:,k)';ell=C(k,1)+U*C(k,2:end)'+residual*(B(:,k)./gamma)-.5*sum(B(:,k).^2./gamma);A(:,k)=rand(N,1)<1./(1+exp(-ell));
 end
 for l=1:L
  eta=[ones(N,1),U]*C';eta0=eta-U(:,l)*C(:,l+1)';eta1=eta0+C(:,l+1)';ell=log(p(l)/(1-p(l)))+sum(A.*(eta1-eta0)-softplus(eta1)+softplus(eta0),2);U(:,l)=rand(N,1)<1./(1+exp(-ell));
 end
 for j=1:J,Z(:,j)=cma_column(Z(:,j),a.R(:,j),b0(j)+A*B(j,:)',gamma(j),1,'sequential');end
 for j=1:J
  for k=1:K
   res=Z(:,j)-b0(j)-A*B(j,:)'+A(:,k)*B(j,k);q=sum(A(:,k).^2)/gamma(j);h=A(:,k)'*res/gamma(j);logp=@(x)-.5*q*x.^2+h*x-rate(j,k)*abs(x);B(j,k)=agr_slice(B(j,k),logp,-Inf,Inf,1/sqrt(max(q,1)),100,10000);
  end
 end
 for j=1:J
  v=1/(N/gamma(j)+.01);b0(j)=v*sum(Z(:,j)-A*B(j,:)')/gamma(j)+sqrt(v)*randn;res=Z(:,j)-b0(j)-A*B(j,:)';gamma(j)=1/gamrnd(1+N/2,1/(1+.5*sum(res.^2)));
 end
 X=[ones(N,1),U];
 for k=1:K
  for l=1:L+1
   other=X*C(k,:)'-X(:,l)*C(k,l);if l==1,prior=@(x)-.005*x.^2;else,rr=rate2(k,l-1);prior=@(x)-rr*abs(x);end
   logp=@(x)sum(A(:,k).*(other+X(:,l)*x)-softplus(other+X(:,l)*x))+prior(x);C(k,l)=agr_slice(C(k,l),logp,-Inf,Inf,.1,100,10000);
  end
 end
 p=betarnd(1+sum(U),1+N-sum(U));drawB(:,:,it)=B;drawC(:,:,it)=C;drawp(it,:)=p;drawgamma(it,:)=gamma';drawb0(it,:)=b0';drawtheta1(it,:)=state1.theta;drawtheta2(it,:)=state2.theta;drawactive1(it,:)=state1.active;drawactive2(it,:)=state2.active;if mod(it,50)==0 || it==iterations,save(fullfile(d,'CHAIN.mat'),'drawB','drawC','drawp','drawgamma','drawb0','drawtheta1','drawtheta2','drawactive1','drawactive2','state1','state2','gamma','b0','it','B','C','A','U','Z','p','-v7.3');end;fprintf('GIBBS n%d iteration%d seconds%.1f\n',N,it,toc(timer));
end
fid=fopen(fullfile(d,'COMPLETE.json'),'w');fprintf(fid,'%s',jsonencode(struct('iterations',iterations,'seconds',toc(timer),'converged',false)));fclose(fid);
end
function y=softplus(x),y=max(x,0)+log1p(exp(-abs(x)));end

function restore_state(p,r),path(p);rng(r);end
