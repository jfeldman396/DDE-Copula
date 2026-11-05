function [q,draw,v]=pilot_mix(logw,exact,bits,old,lambda,method)
% Mixture of optimized mean field and the same tempered exact block law.
v=zeros(1,8);
if ~strcmp(method,'VI') || lambda==1
 [q,draw]=bl_mix(logw,exact,bits,old,lambda,method); return;
end
[product,~,d]=cmv_cavi(logw,bits,old);
q=(1-lambda)*product+lambda*exact;
useProduct=rand(size(old,1),1)<1-lambda;
draw=exact; draw(useProduct,:)=product(useProduct,:);
v=[size(old,1),sum(d.initial_kl),sum(d.final_kl),max(d.initial_kl-d.final_kl),sum(d.sweeps),max(d.stationarity_residual),d.seconds,size(bits,2)];
end
