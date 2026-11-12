function run_uncertainty_case(J,N,rep,out,iterations)
% Original maximal-width 50-step MAP initializer and four posterior chains.
% Omit ITERATIONS for the published 2000 sweeps. Short runs cannot be collected.
assert(ismember(J,[50 100])&&ismember(N,[1000 2000])&&ismember(rep,1:10));
assert(java.io.File(out).isAbsolute()&&~isfolder(out)&&~isfile(out));
if nargin<5,iterations=2000;end
repo=fileparts(fileparts(fileparts(mfilename('fullpath'))));oldpath=path;oldrng=rng;oldcwd=pwd;cleanup=onCleanup(@()restore(oldpath,oldrng,oldcwd));
addpath(fullfile(repo,'Algorithms','DDE'),fullfile(repo,'Scripts','DDE'));verify_revision_sources;
mkdir(out);source=fullfile(out,'source');mkdir(source);stage_dde_sources(fullfile(repo,'Algorithms','DDE'),source,false);
addpath(fullfile(repo,'Utilities'));addpath(genpath(fullfile(source,'CopDDE')));addpath(fullfile(source,'cavi'));
addpath(fullfile(repo,'Algorithms','PosteriorGibbs'),fullfile(repo,'Algorithms','PosteriorGibbs','initialization'));
cd(source);mkdir(fullfile('results','paper_cp_comparison'));copyfile(fullfile(source,'cavi','truth','*.mat'),fullfile('results','paper_cp_comparison'));
truth=pcp_truth(J);truth.K1_max=double(truth.K1_max);truth.K2_max=double(truth.K2_max);rng(50+rep,'twister');[~,X]=generate_X_Cop_marg_emp(N,truth.prop(:)',truth.lambdas,truth.gamma,truth.B1_scaled(:,1:11),truth.B2(1:10,1:4));
R=zeros(N,J);for j=1:J,[~,~,R(:,j)]=unique(X(:,j),'sorted');end
Z=simulate_gaussian_mixture(X,J);[prop,B1,B2,gamma,A1,A2]=Normal_init_v2(Z,truth.K1_max,truth.K1_max,truth.K2_max,1e-4);
initial=struct('Z',Z,'B1',B1,'B2',B2,'gamma',gamma(:),'prop',prop(:)','A1',A1,'A2',A2);dde_assert_rank_consistency(Z,R);
assert(size(A1,2)==floor(J/3)&&size(A2,2)==floor(floor(J/3)/3));
c=pcp_protocol;options=c.options;if J==50,options.spike1=.02;options.spike2=.04;else,options.spike1=.05;options.spike2=.1;end
seed=91000000+J*100000+N*2+rep;name=sprintf('J%d_n%d_r%d',J,N,rep);dataset=fullfile(out,'datasets',name);mkdir(dataset);
save(fullfile(dataset,'INPUT.mat'),'X','R','initial','truth','options','seed','-v7.3');
fit=pcp_fit_stage1(initial,R,max(.3,3*N^(-.3)),'CP',options,seed,truth);assert(isfield(fit,'state50'));
save(fullfile(dataset,'FIT.mat'),'fit','-v7.3');a=struct('R',R);f=fit.state50;
for chain=1:4
 op=struct('seed',930031+J*10000+N+chain*1000000+rep*17000000,'iterations',iterations,'spike1',options.spike1,'spike2',options.spike2,'alpha1',size(f.B1,2)-1,'alpha2',size(f.B2,2)-1);
 run_posterior_gibbs(a,f,op,fullfile(out,'chains',sprintf('%s_chain%d',name,chain)));
end
end
function restore(p,r,w),path(p);rng(r);cd(w);end
