function s=r1tt_score(f,Yfit,Yvalidate,seed,reps,folds,trees)
% No generating U or final test data enters this function.
f=r1tune_selected_model(f);values=zeros(reps,1);detail=cell(1,reps);synthetic_data=cell(1,reps);n=size(Yvalidate,1);timer=tic;
for r=1:reps
 pred=r1t_predict(f,Yfit,n,seed+1000*r);synthetic_data{r}=pred.Y;[values(r),detail{r}]=r1tune_pmse(Yvalidate,pred.Y,Yfit,seed+50*r,trees,folds);
end
s=struct('pmse',mean(values),'replicate_pmse',values,'synthetic_sd',std(values),'synthetic_se',std(values)/sqrt(reps),'detail',{detail},'synthetic_data',{synthetic_data},'seed',seed,'reps',reps,'folds',folds,'trees',trees,'elapsed',toc(timer));
end
