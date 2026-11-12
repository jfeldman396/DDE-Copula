function result=run_native15_5_case(inputFile,method,outputDir)
% Reproduce the completed 15/5 comparator from a paired synthetic input.
% inputFile contains record.R, record.initial, data_seed, algorithm_seed.
assert(ismember(method,{'rl','poisson'}));
assert(java.io.File(inputFile).isAbsolute() && java.io.File(outputDir).isAbsolute());
assert(isfile(inputFile) && ~isfolder(outputDir) && ~isfile(outputDir));
here=fileparts(mfilename('fullpath'));oldpath=path;oldrng=rng;
cleanup=onCleanup(@()restore_state(oldpath,oldrng));
assert(isempty(gcp('nocreate')),'Run without an active parallel pool.');
addpath(fullfile(here,'frozen','common'));
if strcmp(method,'rl')
 addpath(fullfile(here,'frozen','original_rl'),'-begin');
else
 addpath(fullfile(here,'frozen','poisson','utilities'),'-begin');
 addpath(fullfile(here,'frozen','poisson','main algorithms','Poisson'),'-begin');
end
s=load(inputFile,'record');input=s.record;[N,J]=size(input.R);
mkdir(outputDir);
  inputPath=inputFile;ss=load(inputPath,'record');input=ss.record;
  assert(isequal(size(input.R),[N J]));assert(input.data_seed==input.data_seed&&input.algorithm_seed==input.algorithm_seed);
  tt=load(fullfile(here,'frozen','truth',sprintf('truth_J%d.mat',J)),'truth');truth=tt.truth;
  rng(input.data_seed,'twister');[~,X]=generate_X_Cop_marg_emp(N,truth.prop(:)',truth.lambdas,truth.gamma,truth.B1_scaled(:,1:11),truth.B2(1:10,1:4));
  R=zeros(N,J);for j=1:J,[~,~,R(:,j)]=unique(X(:,j),'sorted');end
  assert(isequal(R,input.R),'MATCHED15:Ranks','Paired data regeneration mismatch');
  a=input.initial;
  assert(isequal(size(a.B1),[J 16])&&isequal(size(a.B2),[15 6]));
  if strcmp(method,'rl')
   rng(input.data_seed+700000,'twister');
   [a.prop,a.B1,a.B2,a.gamma,a.A1,a.A2]=Normal_init_v2(a.Z,15,15,5,1e-4);
  end
  assert(isequal(size(a.B1),[J 16])&&isequal(size(a.B2),[15 6])&&isequal(size(a.A1),[N 15])&&isequal(size(a.A2),[N 5]));
  assert(all(isfinite([a.prop(:);a.B1(:);a.B2(:)])),'Nonfinite initial coefficients');
  initial_summary=struct('K1',15,'K2',5,'B1',a.B1,'B2',a.B2,'prop',a.prop,'data_seed',input.data_seed,'algorithm_seed',input.algorithm_seed,'initialization_seed',input.data_seed+700000*strcmp(method,'rl'));
  rng(input.algorithm_seed,'twister');timer=tic;
  if strcmp(method,'rl')
   [gamma,~,~,~,~,Z,prop,B1,B2,~,~,it]=get_SAEM_RL(X,a.Z,input.R,max(input.R,[],1),a.prop,a.B1,a.B2,a.gamma(:)',N^.25,N^.25,max(.3,3*N^(-.3)),a.A1,a.A2,1,ceil(size(a.A2,2)/2),50,.7);
   mu=mean(Z,1)';sd=std(Z,0,1)';assert(all(isfinite(sd)&sd>0));
  else
   [prop,B1,B2,~,~,it]=get_SAEM_poisson(X,a.prop,a.B1,a.B2,N^.25,N^.25,max(.3,3*N^(-.3)),a.A1,a.A2,1,ceil(size(a.A2,2)/2));
   gamma=[];mu=[];sd=[];
  end
  seconds=toc(timer);native=struct('method',method,'J',J,'n',N,'replication',input.replication,'B1',B1,'B2',B2,'prop',prop,'gamma',gamma,'Z_mean',mu,'Z_sd',sd,'iterations',it,'seconds',seconds,'initial',initial_summary,'input_sha256','not-recorded-by-portable-wrapper');save(fullfile(outputDir,'native.mat'),'native','-v7.3');assert(isequal(size(B1),[J 16])&&isequal(size(B2),[15 6]));assert(all(isfinite([B1(:);B2(:);prop(:)])));
  % Native common15/5 evaluation; this is not a crop of a wider fitted model.
  truth.B1_unscaled=truth.B1_unscaled(:,1:16);truth.B1_scaled=truth.B1_scaled(:,1:16);truth.B2=truth.B2(1:15,1:6);truth.K1_max=15;truth.K2_max=5;
  [b1,b2]=revision_align_poisson(B1,B2,truth);
  if strcmp(method,'rl'),b1=[(b1(:,1)-mu)./sd,b1(:,2:end)./sd];end
  metrics=cpd_metrics(b1,b2,truth);metrics.B1_aligned=b1;metrics.B2_aligned=b2;
  metrics.support_accuracy1=mean((b1(:,2:end)~=0)==(truth.B1_scaled(:,2:end)~=0),'all');metrics.support_accuracy2=mean((b2(:,2:end)~=0)==(truth.B2(:,2:end)~=0),'all');

result=struct('native',native,'metrics',metrics,'method',method,'J',J,'n',N,'replication',input.replication);
save(fullfile(outputDir,'result.mat'),'result','-v7.3');
end
function restore_state(p,r),path(p);rng(r);end
