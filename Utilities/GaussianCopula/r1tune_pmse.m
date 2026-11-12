function [score,detail]=r1tune_pmse(Yreal,Ysynthetic,Ytraining,seed,trees,folds)
% Paper pMSE: equal real/synthetic sizes, out-of-fold random-forest scores.
% No parameter selection or imputation uses the final test set.
if nargin<5,trees=100;end;if nargin<6,folds=5;end
assert(isequal(size(Yreal),size(Ysynthetic)) && size(Yreal,2)==size(Ytraining,2));
assert(size(Yreal,1)>=folds);
Ysynthetic(isnan(Yreal))=NaN;
med=median(Ytraining,1,'omitnan');assert(all(isfinite(med)));
for j=1:size(Yreal,2)
 Yreal(isnan(Yreal(:,j)),j)=med(j);
 Ysynthetic(isnan(Ysynthetic(:,j)),j)=med(j);
end
assert(all(isfinite(Yreal),'all') && all(isfinite(Ysynthetic),'all'));
N=size(Yreal,1);X=[Yreal;Ysynthetic];y=[ones(N,1);zeros(N,1)];
rng(seed,'twister');cv=cvpartition(y,'KFold',folds);pred=zeros(2*N,1);
for k=1:folds
 fit=TreeBagger(trees,X(training(cv,k),:),y(training(cv,k)),...
  'Method','classification','OOBPrediction','Off','MinLeafSize',5,...
  'Options',statset('UseParallel',false));
 [~,p]=predict(fit,X(test(cv,k),:));id=find(strcmp(fit.ClassNames,'1'));
 assert(isscalar(id));pred(test(cv,k))=p(:,id);
end
score=mean((pred-.5).^2);
[~,~,~,auc]=perfcurve(y,pred,1);
detail=struct('pMSE',score,'AUC',auc,'Brier',mean((pred-y).^2),...
 'rows_per_class',N,'folds',folds,'trees',trees,'min_leaf',5,'seed',seed);
end
