function [K1init,K2init,diagnostic]=dde_spectral_widths(Z,K1max,buffer)
%DDE_SPECTRAL_WIDTHS Largest consecutive covariance eigenvalue ratio + buffer.
% Uses only initialized Gaussian observations, never true latent factors.
% Search ranks1:K1max, retaining lambda_(K1max+1) for the last ratio.
% Cap proposed first-layer width at original candidate maximum and usable rank.
if nargin<2,K1max=floor(size(Z,2)/3);end
if nargin<3,buffer=5;end
[N,J]=size(Z);assert(N>=4&&J>=4&&all(isfinite(Z),'all'));
assert(isscalar(K1max)&&K1max>=3&&K1max==fix(K1max));
assert(isscalar(buffer)&&buffer>=0&&buffer==fix(buffer));
Zc=Z-mean(Z,1);C=(Zc'*Zc)/(N-1);eigenvalues=sort(eig((C+C')/2),'descend');
eigenvalues=max(eigenvalues,0);assert(eigenvalues(1)>0,'SpectralWidth:NoVariance','Initialized Z has no variance.');
cap=min([K1max,J,N-1]);searchMax=min(cap,J-1);
floorValue=eps(eigenvalues(1))*max(J,N);ratios=eigenvalues(1:searchMax)./max(eigenvalues(2:searchMax+1),floorValue);
[gapRatio,gapRank]=max(ratios);K1init=min(cap,max(3,gapRank+buffer));K2init=floor(K1init/3);
diagnostic=struct('gap_rank',gapRank,'gap_ratio',gapRatio,'buffer',buffer,'K1_init',K1init,'K2_init',K2init, ...
 'K1_original_max',K1max,'cap',cap,'search_max',searchMax,'eigenvalues',eigenvalues,'consecutive_ratios',ratios, ...
 'definition','argmax lambda_k/lambda_(k+1), centered covariance of initial Z; add5,cap at original max; K2=floor(K1/3)');
end
