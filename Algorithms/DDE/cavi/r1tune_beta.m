function beta=r1tune_beta(it,beta0)
% Original slow schedule shape, with explicit initial inverse temperature.
if nargin<2,beta0=.7;end
assert(it>=1 && beta0>0 && beta0<=1);
curve=zeros(1,15);b=.7;
for t=1:15,curve(t)=b;if t>5,b=min(1,b+.01*(t-6));end,end
position=1+14*min(1,(it-1)/29);
beta=interp1(1:15,curve,position,'linear');
if beta0~=.7,beta=beta0+(1-beta0)*(beta-.7)/.3;end
if it>=30,beta=1;end
end
