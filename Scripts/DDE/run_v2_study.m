function run_v2_study(outputDir,jobs)
%RUN_V2_STUDY Explicit serial case list; does not start a pool or skip failures.
% jobs is a table with J,n,replication,arm_id. Output must be a NEW absolute path.
assert(istable(jobs)&&all(ismember({'J','n','replication','arm_id'},jobs.Properties.VariableNames)));
assert(height(jobs)>0&&height(unique(jobs(:,{'J','n','replication','arm_id'})))==height(jobs),'V2:DuplicateJobs','Case list must be nonempty and unique.');
assert(all(ismember(jobs.J,[50 100 150]))&&all(ismember(jobs.n,[500 1000 2000 4000 8000 16000])));
assert(all(ismember(jobs.replication,1:100))&&all(ismember(jobs.arm_id,1:12)));
outputDir=char(outputDir);file=java.io.File(outputDir);
assert(file.isAbsolute()&&~isfolder(outputDir)&&~isfile(outputDir),'V2:FreshOutput','Use a new absolute output directory.');
verify_v2_sources;mkdir(outputDir);writetable(jobs,fullfile(outputDir,'jobs.csv'));
for k=1:height(jobs)
 name=sprintf('J%d_n%d_r%d_a%d',jobs.J(k),jobs.n(k),jobs.replication(k),jobs.arm_id(k));
 run_v2_case(jobs.J(k),jobs.n(k),jobs.replication(k),jobs.arm_id(k),fullfile(outputDir,name));
end
end
