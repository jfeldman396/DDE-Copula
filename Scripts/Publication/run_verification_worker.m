function run_verification_worker(planfile)
% Independent full-budget cases, claimed atomically across MATLAB processes.
% A completed case is not a numerical parity check; collectors check results.
plan=jsondecode(fileread(planfile));repo=fileparts(fileparts(fileparts(mfilename('fullpath'))));
restoredefaultpath;addpath(fullfile(repo,'Scripts','Publication'),fullfile(repo,'Scripts','DDE'));
settings=parallel.Settings;settings.Pool.AutoCreate=false;
assert(isempty(gcp('nocreate')),'Start without an active parallel pool.');
for i=1:numel(plan.jobs)
 j=plan.jobs(i);receipt=fullfile(plan.output,'status',[j.id '.json']);
 if isfile(receipt),continue;end
 lock=java.io.File(fullfile(plan.output,'claims',j.id));if ~lock.mkdir(),continue;end
 if java.io.File(plan.output).getUsableSpace()<25*1024^3,error('Verification:DiskSpace','Less than 25 GiB free; stopping before the next case.');end
 out=fullfile(plan.output,'runs',j.id);started=datetime('now','TimeZone','UTC');timer=tic;
 diary(fullfile(plan.output,'logs',[j.id '.log']));
 record=struct('id',j.id,'kind',j.kind,'source_sha256',plan.source_sha256,'started_utc',char(started),'status','RUNNING','full_budget',true);
 writejson(receipt,record);
 try
  switch j.kind
   case 'ordinal_selection'
    run_ordinal_selection(out,j.rep);
    assert(isfile(fullfile(out,'SELECTION_CHECK.json')));
   case 'ordinal'
    r=run_ordinal_case(j.family,j.method,out);assert(r.complete);
   case 'gaussian'
    run_gaussian_study(plan.survey_input,out);
    assert(isfile(fullfile(out,'stage8k','RANKING.csv')));
   case 'empirical'
    r=run_empirical_case(plan.survey_input,j.method,j.rep,out);assert(r.complete);
   case 'b5'
    fit=fullfile(out,'selected');chains=fullfile(out,'chains');
    r=run_empirical_case(plan.survey_input,'maxima',9,fit);assert(r.complete);
    run_b5_posterior(fullfile(fit,'fit'),chains);
    for c=1:4,assert(isfile(fullfile(chains,sprintf('B5_chain%d',c),'COMPLETE.json')));end
   case 'uncertainty'
    run_uncertainty_case(j.J,j.n,j.rep,out);
    for c=1:4,assert(isfile(fullfile(out,'chains',sprintf('J%d_n%d_r%d_chain%d',j.J,j.n,j.rep,c),'COMPLETE.json')));end
   otherwise
    r=run_revision_case(j.kind,j.J,j.n,j.rep,j.method,out);assert(r.complete);
  end
  record.status='COMPLETED_AWAITING_NUMERICAL_COMPARISON';
 catch err
  record.status='FAILED';record.error_identifier=err.identifier;
  record.error=getReport(err,'extended','hyperlinks','off');fprintf('%s\n',record.error);
 end
 record.seconds=toc(timer);record.finished_utc=char(datetime('now','TimeZone','UTC'));
 writejson(receipt,record);diary off;
end
end
function writejson(p,r)
tmp=[p '.tmp'];f=fopen(tmp,'w');assert(f>=0);fprintf(f,'%s\n',jsonencode(r,PrettyPrint=true));fclose(f);movefile(tmp,p,'f');
end
