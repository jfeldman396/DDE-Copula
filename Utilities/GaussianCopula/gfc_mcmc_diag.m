function d=gfc_mcmc_diag(X)
% Rotation-invariant correlation diagnostics. Rank-normalized split Rhat;
% ESS uses initial-positive autocorrelation pairs on rank-normalized splits.
[N,C,Q]=size(X);H=floor(N/2);S=cat(2,X(1:H,:,:),X(end-H+1:end,:,:));K=2*C;
R=tiedrank(reshape(S,H*K,Q));R=norminv((R-3/8)/(H*K+1/4));R=reshape(R,H,K,Q);
[rhat,ess]=statistics(R);
med=median(reshape(S,H*K,Q),1);F=abs(reshape(S,H*K,Q)-med);
F=norminv((tiedrank(F)-3/8)/(H*K+1/4));F=reshape(F,H,K,Q);
[folded,~]=statistics(F);rhat=max(rhat,folded);
d=struct('split_rank_Rhat_median',median(rhat,'omitnan'),'split_rank_Rhat_max',max(rhat),...
 'split_rank_Rhat_fraction_gt1_1',mean(rhat>1.1),'rank_ESS_median',median(ess,'omitnan'),...
 'rank_ESS_min',min(ess),'rank_ESS_fraction_lt100',mean(ess<100),...
 'ESS_scope','Initial-positive autocorrelation-pair ESS on rank-normalized split correlation chains; retained thinned draws only.');
end
function [rhat,ess]=statistics(X)
[N,C,Q]=size(X);means=squeeze(mean(X,1));W=squeeze(mean(var(X,0,1),2))';
B=N*var(means,0,1);V=(N-1)/N*W+B/N;rhat=sqrt(V./W);
X=X-mean(X,1);nfft=2^nextpow2(2*N);F=fft(X,nfft,1);ac=real(ifft(abs(F).^2,[],1));
ac=ac(1:N,:,:)./(N-(0:N-1)');ac=squeeze(mean(ac,2));
rho=1-(W-ac)./V;rho(1,:)=1;sumrho=zeros(1,Q);alive=true(1,Q);previous=inf(1,Q);
for lag=2:2:N-1
 pair=rho(lag,:)+rho(lag+1,:);alive=alive & pair>0;pair=min(pair,previous);previous=pair;
 sumrho(alive)=sumrho(alive)+pair(alive);
end
ess=min(N*C,N*C./max(1,1+2*sumrho));
end
