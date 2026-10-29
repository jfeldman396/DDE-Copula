function input=lsa_input(J,N,rep)
c=lsa_protocol;path=fullfile(c.out,'inputs',sprintf('input_J%d_n%05d_r%03d.mat',J,N,rep));cached=fullfile('results/paper_factorial_study/inputs',sprintf('input_J%d_n%05d_r%03d.mat',J,N,rep));if isfile(cached),s=load(cached,'record');input=s.record;return;end
old=fullfile('results/spectral_paper_study/inputs',sprintf('input_J%d_n%05d_r%03d.mat',J,N,rep));
if isfile(path),s=load(path,'record');input=s.record;return;end
if isfile(old),s=load(old,'record');input=s.record;return;end
lock=[path '.lock'];f=java.io.File(lock);
while ~f.mkdir()
 if isfile(path),s=load(path,'record');input=s.record;return;end
 pause(.2);
end
cleanup=onCleanup(@()rmdir(lock));
truth=pcp_truth(J);rng(50+rep,'twister');[~,X]=generate_X_Cop_marg_emp(N,truth.prop(:)',truth.lambdas,truth.gamma,truth.B1_scaled(:,1:11),truth.B2(1:10,1:4));R=zeros(N,J);for j=1:J,[~,~,R(:,j)]=unique(X(:,j),'sorted');end
Z=simulate_gaussian_mixture(X,J);postZ=rng;[K1,K2,spectral]=dde_spectral_widths(Z,floor(J/3),5);rng(postZ);[prop,B1,B2,gamma,A1,A2]=Normal_init_v2(Z,K1,K1,K2,1e-4);initial=struct('Z',Z,'B1',B1,'B2',B2,'gamma',gamma(:),'prop',prop(:)','A1',A1,'A2',A2);dde_assert_rank_consistency(Z,R);
input=struct('J',J,'n',N,'replication',rep,'R',R,'initial',initial,'spectral',spectral,'data_seed',50+rep,'algorithm_seed',91000000+J*100000+N*2+rep,'post_Z_rng',postZ,'blocks',psr_blocks(J,K1,K2));cf_save(path,input);
end
