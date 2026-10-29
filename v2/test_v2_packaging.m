function test_v2_packaging
% Static dependency and deterministic helper checks; no fitting/simulation.
manifest=verify_v2_sources;root=fileparts(mfilename('fullpath'));
oldpath=path;oldcwd=pwd;oldrng=rng;tmp=tempname;mkdir(tmp);
cleanup=onCleanup(@()restore(oldpath,oldcwd,oldrng,tmp));
copyfile(fullfile(root,'CopDDE'),fullfile(tmp,'CopDDE'));
restoredefaultpath;addpath(genpath(fullfile(tmp,'CopDDE')));cd(tmp);
roots=cell(numel(manifest.roots),1);
for k=1:numel(roots),roots{k}=fullfile(tmp,erase(manifest.roots{k},'v2/'));end
[dependencies,products]=matlab.codetools.requiredFilesAndProducts(roots);
assert(numel(dependencies)==sum(endsWith({manifest.files.path},'.m')),'V2:ClosureCount');
for k=1:numel(dependencies),assert(startsWith(dependencies{k},[tmp filesep]),'V2:ExternalSource');end
A=[1 0;0 1;1 1];U=[0 0;1 0;1 0];B=[0 1;1 0;0 0];V=[0 1;0 1;1 0];
a=foa_accumulate([],A,U,1);a=foa_accumulate(a,B,V,.25);
assert(isequal(a.config,[0 0;0 1;1 0]));assert(isequal(a.N,[.75;.5;1.75]));assert(isequal(a.T,[.75 0;.25 .25;.75 1.5]));assert(isequal(a.top,[1.75 .5]));
% Aggregated objective equals the explicitly weighted observation likelihood.
W=[.2 -.1 .3;-.3 .4 .1];[value,grad]=foa_objective(a,W);
X=[ones(3,1) U];Y=[ones(3,1) V];old=X*W';new=Y*W';
expected=.75*sum(A.*old-dde_softplus(old),'all')+.25*sum(B.*new-dde_softplus(new),'all');
g=.75*(A-dde_logistic(old))'*X+.25*(B-dde_logistic(new))'*Y;
assert(abs(value-expected)<1e-12&&max(abs(grad-g),[],'all')<1e-12);
cf_save(fullfile(tmp,'accumulator.mat'),a);s=load(fullfile(tmp,'accumulator.mat'),'record');
assert(isequaln(foa_accumulate(a,A,U,.2),foa_accumulate(s.record,A,U,.2)));
R=[1;1;2;3;NaN];Z=[-2;-1;0;1;4];dde_assert_rank_consistency(Z,R);dde_assert_rank_consistency(Z,exp(R));
failed=false;try,dde_assert_rank_consistency([-2;1;0;2;4],R);catch e,failed=strcmp(e.identifier,'DDE:RankViolation');end;assert(failed);
for id=1:12
 arm=pfs_arm(id);assert(arm.SA_start==arm.warmup+21&&arm.cap==arm.warmup+100);
end
assert(cms_beta(1,true)==.7&&cms_beta(30,true)==1&&cms_beta(160,true)==1);
c=lsa_protocol;arm=pfs_arm(4);h=repmat({struct('B1',ones(2),'B2',ones(2),'gamma',ones(2,1),'prop',ones(1,2),'active1',true(1,2),'active2',true(1,2))},1,40);
assert(lsa_stop(h,40,arm,c));assert(~lsa_stop(h,39,arm,c));
for t=31:40,h{t}.gamma=1.011*ones(2,1);end;assert(~lsa_stop(h,40,arm,c));
fprintf('PASS: %d source files; %d MATLAB dependency files; weighted objective/gradient; accumulator serialization; rank checks; arm/temperature/stop checks.\n',numel(manifest.files),numel(dependencies));
disp({products.Name});
end
function restore(p,d,r,t)
cd(d);path(p);rng(r);rmdir(t,'s');
end
