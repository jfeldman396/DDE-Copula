function input=create_deep_nofloor_input(N,rep,outputFile)
% Recreate synthetic gap+5 inputs using the archived marginal parameters.
% Zero top-layer width is an initialization failure, not a fitted endpoint.
assert(ismember(N,[500 1000 2000 4000 8000 16000]));
validateattributes(rep,{'numeric'},{'scalar','integer','>=',1,'<=',100});
outputFile=char(outputFile);
assert(java.io.File(outputFile).isAbsolute(),'V2:AbsolutePaths','Use an absolute output path.');
assert(~isfile(outputFile)&&~isfolder(outputFile),'V2:FreshOutput','Output must not exist.');
verify_revision_sources;
root=fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))),'Algorithms','DDE');oldpath=path;oldrng=rng;
cleanup=onCleanup(@()restore(oldpath,oldrng));
addpath(genpath(fullfile(root,'CopDDE')));
addpath(fullfile(root,'deep','input_generation'),'-begin');
truthFile=fullfile(root,'deep','input_generation','truth.mat');
s=load(truthFile,'record');truth=s.record;
rng(50+rep,'twister');
[X,Ztrue,Atrue]=generate_X_cop_D(N,truth.lambdas,truth.prop,3,truth.B,truth.gamma);
R=zeros(N,108);for j=1:108,[~,~,R(:,j)]=unique(X(:,j),'sorted');end
clock=tic;Z=simulate_gaussian_mixture(X,108);
% Preserve the historical initialization call and random-number sequence.
Normal_init_D(Z,3,num2cell([36 12 4]),1e-4);
dde_assert_rank_consistency(Z,R);
input=struct('n',N,'replication',rep,'seed_data_initialization',50+rep, ...
 'X',X,'R',R,'Rlevels',max(R,[],1),'Z_true',Ztrue,'A_true',{Atrue}, ...
 'initialization_seconds',toc(clock),'tau',max(.3,3*N^(-.3)), ...
 'rng_before_fit',rng,'truth_path',truthFile);
input.truth_marginal_mean=mean(Ztrue,1)';input.truth_marginal_sd=std(Ztrue,0,1)';
[k1,k2,diagnostic]=dde_spectral_widths(Z,36,5);k3=floor(k2/3);
assert(k3>=1,'Gap5:InvalidWidth','Original rule produced zero top-layer width');
widths=[k1 k2 k3];input.initialization_seed=710000+rep;
rng(input.initialization_seed,'twister');
[prop,B,gamma,A]=Normal_init_D(Z,3,num2cell(widths),1e-4);
input.initial=struct('prop',prop(:)','B',{B},'gamma',gamma(:),'A',{A},'Z',Z,'widths',widths);
input.J=108;input.spectral=diagnostic;input.spectral.widths=widths;
input.spectral.third_layer_rule='floor(K2/3); no minimum-three or pyramid adjustment';
input.algorithm_seed=910000+rep;
assert(all(cellfun(@(v)all(isfinite(v),'all'),B))&&all(isfinite(gamma))&&all(gamma>0));
dde_assert_rank_consistency(Z,R);
% The fit wrapper sets the no-floor threshold; tau above is archived metadata.
save(outputFile,'input','-v7.3');
end
function restore(p,r)
path(p);rng(r);
end
