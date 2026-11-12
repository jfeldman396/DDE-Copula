function b=gfc_beta(it)
% Same 30-iteration complete-likelihood power schedule as current lsa_fit.
curve=zeros(1,15);beta=.7;
for t=1:15,curve(t)=beta;if t>5,beta=min(1,beta+.01*(t-6));end,end
position=1+14*min(1,(it-1)/29);b=interp1(1:15,curve,position,'linear');
end
