function T=summarize_v2(studyDir,outputCSV)
%SUMMARIZE_V2 Export fresh-wrapper endpoints; no bootstrap or silent exclusions.
% Does not read/modify the original production archive or compact paper tables.
assert(~isfile(outputCSV),'V2:NoOverwrite');jobs=readtable(fullfile(studyDir,'jobs.csv'));rows=cell(height(jobs),1);
for k=1:height(jobs)
 name=sprintf('J%d_n%d_r%d_a%d',jobs.J(k),jobs.n(k),jobs.replication(k),jobs.arm_id(k));
 file=fullfile(studyDir,name,'result.mat');assert(isfile(file),'V2:IncompleteStudy','Missing %s; retain and diagnose failed/incomplete cases.',name);
 s=load(file,'result');r=s.result;f=r.fit;m=r.metrics;a=r.job.arm;
 assert(r.job.J==jobs.J(k)&&r.job.n==jobs.n(k)&&r.job.replication==jobs.replication(k)&&a.id==jobs.arm_id(k),'V2:CaseMismatch');
 rows{k}=struct('J',r.job.J,'n',r.job.n,'replication',r.job.replication,'arm_id',a.id,'method',string(a.scheme),'setting',string(a.setting),'K1',m.K1,'K2',m.K2,'joint_correct',m.joint_correct,'B1_MSE',m.B1_full,'B2_MSE',m.B2_full,'CPU',f.CPU,'iterations',f.iteration,'stabilized',f.stabilized,'cap_reached',f.iteration==a.cap,'optimizer_flags',sum(cellfun(@(x)x.flags,f.trace)));
end
T=struct2table(vertcat(rows{:}));writetable(T,outputCSV);
end
