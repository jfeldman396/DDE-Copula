function [q,draw,d]=cmv_mix(logw,exact,bits,old,lambda)
if lambda==1, q=exact; draw=exact; d=[]; return; end
[vi,r,d]=cmv_cavi(logw,bits,old);
[~,prod]=bl_mix(logw,exact,bits,old,0,'CP');
pold=prod*bits;
d.probability_difference=mean(abs(r-pold),2);
d.tv_exact=.5*sum(abs(vi-exact),2); d.product_tv_exact=.5*sum(abs(prod-exact),2);
q=(1-lambda)*vi+lambda*exact;
choose=rand(size(old,1),1)<1-lambda; draw=exact; draw(choose,:)=vi(choose,:);
end
