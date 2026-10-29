function metrics=psr_score(fit,truth)
% Pad to original paper maximum width BEFORE existing evaluation alignment.
J=truth.J;M1=truth.K1_max;M2=truth.K2_max;K1=size(fit.B1,2)-1;K2=size(fit.B2,2)-1;
assert(K1<=M1&&K2<=M2);
f=struct('B1',zeros(J,M1+1),'B2',zeros(M1,M2+1),'prop',.5*ones(1,M2),'gamma',fit.gamma,'Z',fit.Z, ...
 'active1',false(1,M1),'active2',false(1,M2));
f.B1(:,1:K1+1)=fit.B1;f.B2(1:K1,1:K2+1)=fit.B2;f.prop(1:K2)=fit.prop;f.active1(1:K1)=fit.active1;f.active2(1:K2)=fit.active2;
e=evaluate_exact_gibbs_fit(f,truth);m=cpd_metrics(e.B1_aligned,e.B2_aligned,truth);
metrics=m;metrics.K1=e.K1;metrics.K2=e.K2;metrics.correct1=e.K1==truth.K1_true;metrics.correct2=e.K2==truth.K2_true;metrics.joint_correct=metrics.correct1&&metrics.correct2;
metrics.B1_aligned=e.B1_aligned;metrics.B2_aligned=e.B2_aligned;metrics.assignment1=e.assignment1;metrics.assignment2=e.assignment2;
metrics.support_accuracy1=e.support_accuracy1;metrics.support_accuracy2=e.support_accuracy2;
metrics.definition='Paper-maximum-width MSE INCLUDING intercepts, zero padded then existing alignment; conditional summaries require both dimensions correct. True-block errors also supplied.';
end
