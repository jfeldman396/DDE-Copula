function [yes,d]=lsa_stop(history,it,arm,c)
yes=false;d=struct('eligible',false,'B1',NaN,'B2',NaN,'gamma',NaN,'prop',NaN,'active_stable',false);
if it-arm.warmup<c.minimum_post||mod(it-arm.warmup,c.check_every)~=0||it<2*c.window,return;end
d.eligible=true;h=history(it-2*c.window+1:it);d.active_stable=all(cellfun(@(x)isequal(x.active1,h{1}.active1)&&isequal(x.active2,h{1}.active2),h));
for key=["B1" "B2" "gamma" "prop"]
 x=cellfun(@(s)s.(key),h,'UniformOutput',false);old=mean(cat(3,x{1:c.window}),3);new=mean(cat(3,x{c.window+1:end}),3);
 d.(key)=norm(new-old,'fro')/max(norm(old,'fro'),1e-8);
end
yes=d.active_stable&&d.B1<=c.weight_tolerance&&d.B2<=c.weight_tolerance&&d.gamma<=c.gamma_tolerance&&d.prop<=c.prop_tolerance;
end
