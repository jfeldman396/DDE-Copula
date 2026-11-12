function endpoint=run_revision_case(study,J,N,rep,method,out,limit)
% Fresh case -> input generation -> fitting -> scored portable endpoint.
% study: shallow (map/vi/tempering), comparator (rl/poisson), deep (map).
% LIMIT is for smoke checks only; omitted uses the original full budget.
assert(ismember(study,{'shallow','comparator','deep'}));
assert(java.io.File(out).isAbsolute()&&~isfolder(out)&&~isfile(out));
repo=fileparts(fileparts(fileparts(mfilename('fullpath'))));
oldpath=path;oldcwd=pwd;oldrng=rng;cleanup=onCleanup(@()restore(oldpath,oldcwd,oldrng));
addpath(fullfile(repo,'Utilities'));
addpath(fullfile(repo,'Algorithms','DDE'),fullfile(repo,'Scripts','DDE'),fullfile(repo,'Scripts','Publication'));
verify_revision_sources;
assert(ismember(N,[500 1000 2000 4000 8000 16000])&&ismember(rep,1:100));
if strcmp(study,'deep'),assert(J==108&&strcmp(method,'map'));else,assert(ismember(J,[50 100 150]));end
mkdir(out);endpoint=struct('study',study,'method',method,'J',J,'n',N,'replication',rep,'complete',false,'initialization_failed',false);
if strcmp(study,'shallow')
 methods={'map','vi','tempering'};arm=find(strcmp(method,methods));assert(isscalar(arm));
 if nargin<7,r=run_cavi_case(J,N,rep,arm,fullfile(out,'fit'));else,r=run_cavi_case(J,N,rep,arm,fullfile(out,'fit'),limit);end
 endpoint.arm=arm;endpoint.complete=r.complete;endpoint.K1=r.metrics.K1;endpoint.K2=r.metrics.K2;
 endpoint.eligible=r.metrics.joint_correct;endpoint.CPU=r.fit.CPU;endpoint.numerical_wall=r.fit.wall;
 endpoint.warmup_wall=r.fit.warm_wall;endpoint.post_wall=r.fit.wall-r.fit.warm_wall;
 % Use the native 15/5 comparison only when the actual fit has those widths.
 endpoint.native15_5=isequal(size(r.fit.B1),[J 16])&&isequal(size(r.fit.B2),[15 6]);
 if endpoint.native15_5
  addpath(fullfile(repo,'Algorithms','Native15_5','frozen','common'));
  t=load(fullfile(repo,'Algorithms','Native15_5','frozen','truth',sprintf('truth_J%d.mat',J)),'truth');t=t.truth;
  t.B1_unscaled=t.B1_unscaled(:,1:16);t.B1_scaled=t.B1_scaled(:,1:16);t.B2=t.B2(1:15,1:6);
  [b1,b2]=revision_align_poisson(r.fit.B1,r.fit.B2,t);
  mu=mean(r.fit.Z,1)';sd=std(r.fit.Z,0,1)';assert(all(sd>0));
  endpoint.B1=[(b1(:,1)-mu)./sd,b1(:,2:end)./sd];endpoint.B2=b2;
  endpoint.truth1=t.B1_scaled;endpoint.truth2=t.B2;
  endpoint.MSE1=mean((endpoint.B1-endpoint.truth1).^2,'all');endpoint.MSE2=mean((endpoint.B2-endpoint.truth2).^2,'all');
 end
elseif strcmp(study,'comparator')
 assert(nargin<7,'Comparators have fixed original budgets; no shortened-limit override.');
 assert(ismember(method,{'rl','poisson'}));
 % Generate the exact paired input via the same fresh lsa_input path as MAP.
 root=fullfile(repo,'Algorithms','DDE');source=fullfile(out,'source');mkdir(source);stage_dde_sources(root,source,false);
 addpath(fullfile(repo,'Utilities'));addpath(genpath(fullfile(source,'CopDDE')));addpath(fullfile(source,'cavi'),'-begin');cd(source);
 mkdir(fullfile('results','layer_specific_saem_study','inputs'));mkdir(fullfile('results','paper_cp_comparison'));
 copyfile(fullfile(source,'cavi','truth','*.mat'),fullfile('results','paper_cp_comparison'));
 record=lsa_input(J,N,rep);assert(isequal(size(record.initial.B1),[J 16])&&isequal(size(record.initial.B2),[15 6]),'DDE:PairedWidths','Fresh input is not native 15/5; do not force different widths.');
 inputFile=fullfile(out,'input.mat');save(inputFile,'record','-v7.3');cd(oldcwd);
 r=run_native15_5_case(inputFile,method,fullfile(out,'fit'));
 endpoint.complete=true;endpoint.eligible=true;endpoint.native15_5=true;
 endpoint.B1=r.metrics.B1_aligned;endpoint.B2=r.metrics.B2_aligned;
 t=load(fullfile(repo,'Algorithms','Native15_5','frozen','truth',sprintf('truth_J%d.mat',J)),'truth');
 endpoint.truth1=t.truth.B1_scaled(:,1:16);endpoint.truth2=t.truth.B2(1:15,1:6);
 endpoint.MSE1=mean((endpoint.B1-endpoint.truth1).^2,'all');endpoint.MSE2=mean((endpoint.B2-endpoint.truth2).^2,'all');
 endpoint.seconds=r.native.seconds;
else
 inputFile=fullfile(out,'input.mat');
 try,input=create_deep_nofloor_input(N,rep,inputFile);
 catch err
  if strcmp(err.identifier,'Gap5:InvalidWidth')
   endpoint.complete=true;endpoint.initialization_failed=true;endpoint.reason=err.message;
   save(fullfile(out,'endpoint.mat'),'endpoint','-v7.3');return
  end
  rethrow(err);
 end
 if nargin<7,limit=160;end
 r=run_deep_saved_case(inputFile,fullfile(out,'fit'),limit);
 truthFile=fullfile(repo,'Algorithms','DDE','deep','input_generation','truth.mat');
 m=score_deep_saved_case(fullfile(out,'fit','result.mat'),inputFile,truthFile);t=load(truthFile,'record');
 endpoint.complete=r.complete;endpoint.eligible=m.correct_all;endpoint.K=m.K;
 for d=1:3
  endpoint.(sprintf('B%d',d))=m.B_aligned{d};endpoint.(sprintf('MSE%d',d))=m.MSE(d);target=t.record.B_padded{d};
  if d==1,target=[(target(:,1)-input.truth_marginal_mean)./input.truth_marginal_sd,target(:,2:end)./input.truth_marginal_sd];end
  endpoint.(sprintf('truth%d',d))=target;
 end
end
save(fullfile(out,'endpoint.mat'),'endpoint','-v7.3');
end
function restore(p,d,r),cd(d);path(p);rng(r);end
