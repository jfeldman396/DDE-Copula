function a=pfs_arm(id)
assert(ismember(id,1:12));scheme=ceil(id/4);setting=mod(id-1,4)+1;names={'H1','H0','Z'};settings={'temp_CP','CP_only','temp_only','neither'};
a=struct('id',id,'scheme',names{scheme},'setting',settings{setting},'tempering',ismember(setting,[1 3]),'CP',ismember(setting,[1 2]));
a.warmup=0;if a.tempering,a.warmup=30;end;if a.CP,a.warmup=60;end;a.SA_start=a.warmup+21;a.cap=a.warmup+100;a.label=[a.scheme '_' a.setting];
end
