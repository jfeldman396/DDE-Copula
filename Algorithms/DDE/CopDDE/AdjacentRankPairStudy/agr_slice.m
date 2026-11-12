function [m,evals,steps,shrinks]=agr_slice(m,logp,L,U,width,maxsteps,maxshrink)
current=logp(m);assert(isfinite(current),'AGR:CurrentSliceDensity');height=current+log(rand);evals=1;
% Random offset before clipping to exact external bounds.
offset=rand;left=max(L,m-width*offset);right=min(U,m+width*(1-offset));
j=floor(maxsteps*rand);k=maxsteps-1-j;steps=0;
while j>0&&left>L
 v=logp(left);evals=evals+1;if v<=height,break;end
 left=max(L,left-width);j=j-1;steps=steps+1;
end
while k>0&&right<U
 v=logp(right);evals=evals+1;if v<=height,break;end
 right=min(U,right+width);k=k-1;steps=steps+1;
end
origin=m;
for shrinks=0:maxshrink-1
 proposal=left+(right-left)*rand;v=logp(proposal);evals=evals+1;
 if v>=height,m=proposal;return;end
 if proposal<origin,left=proposal;else,right=proposal;end
end
error('AGR:SliceShrinkLimit','Slice shrinkage limit reached; do not clip/reuse failed draw.');
end
