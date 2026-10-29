function truth=pcp_truth(J)
c=pcp_protocol; path=fullfile(c.output_dir,sprintf('truth_J%d.mat',J));
if isfile(path), s=load(path,'truth'); truth=s.truth; return; end
K1=floor(J/3); K2=floor(K1/3);
if ismember(J,[50 100])
    original=fullfile(c.root,'CopDDE','Results',sprintf('J%d_overspecified.mat',J));
    s=load(original,'B1_true_aug_unscale','B1_true_aug_scale','B2_true_aug', ...
        'gamma_scale','prop_true','lambdas');
    truth=struct('B1_unscaled',s.B1_true_aug_unscale,'B1_scaled',s.B1_true_aug_scale, ...
        'B2',s.B2_true_aug,'gamma',s.gamma_scale,'prop',s.prop_true,'lambdas',s.lambdas, ...
        'source',original,'source_hash',fullwidth_gibbs_file_hash(original));
else
    B1=sim_block_loadings(J,10,J/10,10,5,0,1);
    B2=zeros(10,3); ix=floor(linspace(1,10,4));
    for i=1:3
        B2(ix(i):ix(i+1),i)=4;
        if mod(i,2)==0, B2(ix(i):ix(i+1),i-1)=-4/3;
        elseif i<2, B2(ix(i):ix(i+1),i+1)=-4/3; end
    end
    b1=[-2*ones(floor(J/3),1);-4*ones(floor(J/3),1);-2*ones(J-2*floor(J/3),1)];
    b2=[-2*ones(3,1);-ones(3,1);-2*ones(4,1)];
    truth.B1_unscaled=[b1 B1 zeros(J,K1-10)];
    truth.B2=[[b2;zeros(K1-10,1)] [B2;zeros(K1-10,3)] zeros(K1,K2-3)];
    truth.prop=.5*ones(3,1);
    [scaled,truth.gamma]=rescale_B1(truth.prop,[b2 B2],[b1 B1],ones(J,1),false,truth.prop);
    truth.B1_scaled=[scaled zeros(J,K1-10)];
    truth.lambdas=randi([1 10],1,J);
    truth.source='CopDDE/Scripts/Sim_Overspecified.m, truth seed 1';
end
truth.J=J; truth.K1_true=10; truth.K2_true=3; truth.K1_max=K1; truth.K2_max=K2;
assert(isequal(size(truth.B1_scaled),[J K1+1]) && isequal(size(truth.B2),[K1 K2+1]));
save(path,'truth');
end
