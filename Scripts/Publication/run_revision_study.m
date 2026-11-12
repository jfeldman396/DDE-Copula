function run_revision_study(study,out)
% Explicit full synthetic-study driver; expensive, no saved-endpoint fallback.
% study: shallow, comparison, deep, ordinal, uncertainty.
% comparison fits only RL/Poisson; collect with MAP endpoints from shallow.
assert(ismember(study,{'shallow','comparison','deep','ordinal','uncertainty'}));
assert(java.io.File(out).isAbsolute()&&~isfolder(out)&&~isfile(out));
repo=fileparts(fileparts(fileparts(mfilename('fullpath'))));oldpath=path;cleanup=onCleanup(@()path(oldpath));
addpath(fullfile(repo,'Scripts','DDE'),fullfile(repo,'Scripts','Publication'));mkdir(out);
if strcmp(study,'ordinal')
 for family={'clayton','survival_clayton','student_t','gaussian'}
  for method={'gibbs','maxima'},run_ordinal_case(family{1},method{1},fullfile(out,[family{1} '_' method{1}]));end
 end
elseif strcmp(study,'uncertainty')
 for J=[50 100]
  for n=[1000 2000]
   for rep=1:10,run_uncertainty_case(J,n,rep,fullfile(out,sprintf('J%d_n%d_r%d',J,n,rep)));end
  end
 end
else
 if strcmp(study,'deep'),js=108;methods={'map'};elseif strcmp(study,'shallow'),js=[50 100 150];methods={'map','vi','tempering'};else,js=[50 100 150];methods={'rl','poisson'};end
 for J=js
  for n=[500 1000 2000 4000 8000 16000]
   for rep=1:100
    for method=methods
     kind=study;if strcmp(study,'comparison'),if strcmp(method{1},'map'),kind='shallow';else,kind='comparator';end;end
     run_revision_case(kind,J,n,rep,method{1},fullfile(out,sprintf('J%d_n%d_r%d_%s',J,n,rep,method{1})));
    end
   end
  end
 end
end
fid=fopen(fullfile(out,'STUDY_COMPLETE.json'),'w');assert(fid>=0);fprintf(fid,'%s\n',jsonencode(struct('study',study,'all_cases_returned',true,'convergence_claim',false)));fclose(fid);
end
