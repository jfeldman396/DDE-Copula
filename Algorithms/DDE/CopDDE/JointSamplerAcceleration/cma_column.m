function [z,info]=cma_column(z,R,mu,gamma,sweeps,order)
clock=tic;[levels,~,group]=unique(R,'sorted');L=numel(levels);groups=cell(L,1);means=groups;counts=groups;mapping=groups;M=zeros(L-1,1);
for l=1:L
 groups{l}=find(group==l);[means{l},~,mapping{l}]=unique(mu(groups{l}));counts{l}=accumarray(mapping{l},1);if l<L,M(l)=max(z(groups{l}));end
end
old=M;evals=0;expansions=0;shrinks=0;
for sweep=1:sweeps
 if strcmp(order,'odd_even'),plan=[1:2:L-1 2:2:L-1];else,plan=1:L-1;end
 for l=plan
  a=-Inf;b=Inf;if l>1,a=M(l-1);end;if l<L-1,b=M(l+1);end
  logp=@(x)cma_conditional(x,a,b,means{l},counts{l},means{l+1},counts{l+1},gamma,l==L-1);
  width=.5*sqrt(gamma)/sqrt(numel(groups{l})+numel(groups{l+1}));
  [M(l),e,s,h]=agr_slice(M(l),logp,a,b,width,100,10000);evals=evals+e;expansions=expansions+s;shrinks=shrinks+h;
 end
end
for l=1:L
 ids=groups{l};a=-Inf;if l>1,a=M(l-1);end
 if l<L
  b=M(l);q=agr_log_interval((a-means{l})/sqrt(gamma),(b-means{l})/sqrt(gamma));lf=-.5*((b-means{l}).^2/gamma)-.5*log(2*pi*gamma);logw=log(counts{l})+lf-q;w=exp(logw-max(logw));w=w/sum(w);pat=find(rand<cumsum(w),1);choices=ids(mapping{l}==pat);winner=choices(randi(numel(choices)));z(winner)=b;ids=ids(ids~=winner);
 else,b=Inf;end
 if ~isempty(ids),z(ids)=draw(mu(ids),gamma,a,b);end
end
dde_assert_rank_consistency(z,R);info=struct('evaluations',evals,'expansions',expansions,'shrinks',shrinks,'boundary_displacement',M-old,'maxima',M,'random_TN_draws',numel(z)-max(L-1,0),'regenerated_observations',numel(z),'seconds',toc(clock),'groups',L);
end
function z=draw(mu,g,a,b)
sd=sqrt(g);lo=(a-mu)/sd;hi=(b-mu)/sd;z=mu+sd*agr_truncated_standard_normal(lo,hi);bad=~(z>a&z<b);tries=0;
while any(bad),tries=tries+1;assert(tries<100,'CMA:Representability');z(bad)=mu(bad)+sd*agr_truncated_standard_normal(lo(bad),hi(bad));bad=~(z>a&z<b);end
end
