function [input,audit,marginals]=cpdde_initial(train,o)
% Fresh fold-training-only initialization. This function cannot load old fits.
assert(isfield(train,'role')&&strcmp(train.role,'training'),'ConditionalDDE:TrainingRole','Input must be declared training.');
Y=double(train.Y);[n,J]=size(Y);o=cpdde_options(o,J);
assert(n>=max(20,o.max_widths(1)+2)&&n<=o.maximum_training_n);
assert(all(isfinite(Y(:))|isnan(Y(:)))&&numel(train.row_ids)==n&&numel(unique(train.row_ids))==n);
assert(numel(train.item_ids)==J&&numel(unique(string(train.item_ids)))==J);
assert(isfield(train,'split_id')&&~isempty(train.split_id));
if isfield(train,'forbidden_row_ids'),assert(isempty(intersect(train.row_ids,train.forbidden_row_ids)));end
rng(o.initialization_seed,'twister');R=NaN(n,J);Z=randn(n,J);marginals=cell(1,J);
for j=1:J
 observed=isfinite(Y(:,j));values=Y(observed,j);[levels,~,g]=unique(values,'sorted');
 assert(numel(levels)>=2,'ConditionalDDE:ConstantTrainingItem','Each training item needs at least two observed levels.');
 counts=accumarray(g,1,[numel(levels),1]);low=[0;cumsum(counts(1:end-1))];
 u=(low(g)+rand(sum(observed),1).*counts(g))/sum(observed);
 u=max(min(u,1-eps),realmin);Z(observed,j)=norminv(u);R(observed,j)=g;
 categories=levels(:)';if isfield(train,'categories'),categories=reshape(train.categories,1,[]);end
 assert(issorted(categories)&&numel(unique(categories))==numel(categories));
 [covered,codes]=ismember(values,categories);assert(all(covered));
 category_counts=accumarray(codes,1,[numel(categories),1])';
 pseudo=0;if isfield(train,'pseudocount'),pseudo=train.pseudocount;end;assert(pseudo>=0);
 probabilities=(category_counts+pseudo)/(sum(category_counts)+numel(categories)*pseudo);
 edges=[0,cumsum(probabilities)];edges(end)=1;
 if isfield(train,'counts'),assert(isequal(category_counts,train.counts(j,:)));end
 if isfield(train,'probabilities'),assert(max(abs(probabilities-train.probabilities(j,:)))<1e-12);end
 if isfield(train,'cdf_edges'),assert(max(abs(edges-train.cdf_edges(j,:)))<1e-12);end
 marginals{j}=struct('levels',categories,'counts',category_counts,'observed_levels',levels(:)', ...
  'probabilities',probabilities,'cdf_edges',edges,'pseudocount',pseudo, ...
  'cumulative_probabilities',edges(2:end),'observed_n',sum(observed),'missing_n',sum(~observed));
end
if isfield(train,'R'),assert(isequaln(R,double(train.R)),'ConditionalDDE:RankInputMismatch');end
K1=o.max_widths(1);K2=o.max_widths(2);centered=Z-mean(Z,1);
if strcmp(o.init_method,'spectral')
 [~,~,V]=svd(centered,'econ');[L,~]=rotatefactors(V(:,1:K1),'method','varimax','Maxit',1000);
 L=L.*(2*(mean(L,1)>0)-1);A1=double((centered*L/(L'*L))>0);
else,A1=double(rand(n,K1)<.5);end
[B1,ridge1]=regress_initial(A1,centered,o.regression_ridge);
B1(:,1)=mean(Z,1)'-B1(:,2:end)*mean(A1,1)';
gamma=max(mean((Z-[ones(n,1),A1]*B1').^2,1)',o.minimum_gamma);
if K2==0
 A2=zeros(n,0);p=min(max(mean(A1,1)',o.epsilon),1-o.epsilon);
 B2=log(p./(1-p));prop=zeros(1,0);ridge2=false;
else
 if strcmp(o.init_method,'spectral')
  [U,S,V]=svd(A1,'econ');mm=min(size(S,1),max(K2+1,sum(diag(S)>1.01*sqrt(n))));
  lowrank=U(:,1:mm)*S(1:mm,1:mm)*V(:,1:mm)';lowrank=min(max(lowrank,o.epsilon),1-o.epsilon);
  eta=log(lowrank./(1-lowrank));eta=eta-mean(eta,1);[~,~,V]=svd(eta,'econ');
  if K2==1,L=V(:,1);else,[L,~]=rotatefactors(V(:,1:K2),'method','varimax','Maxit',1000);L(abs(L)<1/3.5/sqrt(K1))=0;end
  L=L.*(2*(mean(L,1)>0)-1);A2=double(eta*pinv(L')>0);
 else
  A2=double(rand(n,K2)<.5);eta=log((o.epsilon+(1-2*o.epsilon)*A1)./(1-o.epsilon-(1-2*o.epsilon)*A1));eta=eta-mean(eta,1);
 end
 [B2,ridge2]=regress_initial(A2,eta,o.regression_ridge);B2(:,2:end)=B2(:,2:end)/2;
 if strcmp(o.init_method,'spectral')&&K2>1,for k=1:K2,B2(k,k+1)=max(B2(k,k+1),2.5);end;end
 B2(:,1)=mean(eta,1)'-B2(:,2:end)*mean(A2,1)';
 prop=min(max(mean(A2,1),o.epsilon),1-o.epsilon);
end
initial=struct('Z',Z,'A1',A1,'A2',A2,'B1',B1,'B2',B2,'gamma',gamma,'prop',prop);
assert(all(isfinite([B1(:);B2(:);gamma(:);prop(:)]))&&all(gamma>0));
dde_assert_binary_model(A1,A2,Z,B1,B2,gamma,prop);dde_assert_rank_consistency(Z,R);
identity=cpdde_digest({Y,reshape(double(train.row_ids),[],1),cellstr(string(train.item_ids)),char(train.split_id)});
initial.training_identity=identity;
initial.initialization_fingerprint=cpdde_digest({Z,A1,A2,B1,B2,gamma,prop});
input=struct('initial',initial,'R',R,'n',n,'J',J,'blocks',o.blocks, ...
 'algorithm_seed',o.algorithm_seed,'replication',1,'training_identity',identity);
audit=struct('split_id',train.split_id,'training_identity',identity,'n',n,'J',J, ...
 'row_ids',reshape(train.row_ids,[],1),'item_ids',{cellstr(string(train.item_ids))}, ...
 'initialization_seed',o.initialization_seed,'algorithm_seed',o.algorithm_seed, ...
 'method',o.init_method,'max_widths',[K1 K2],'training_depth',o.depth, ...
 'lower_ridge_fallback',ridge1,'upper_ridge_fallback',ridge2, ...
 'unbounded_initial_fingerprint',initial.initialization_fingerprint, ...
 'initial_B1_bound_clip_count',sum(abs(B1(:))>10), ...
 'initial_B2_bound_clip_count',sum(B2(:,1)<-5|B2(:,1)>10)+sum(abs(B2(:,2:end))>10,'all'), ...
 'training_only',true,'no_saved_full1500_initials',true);
for key={'input_path','input_file_sha256','data_manifest_sha256'}
 if isfield(train,key{1}),audit.(key{1})=train.(key{1});end
end
end

function [B,used]=regress_initial(A,response,ridge)
centered=A-mean(A,1);G=centered'*centered;used=rcond(G)<1e-10;
if used,G=G+ridge*max(1,trace(G)/max(1,size(G,1)))*eye(size(G));end
loadings=(G\(centered'*response))';B=[zeros(size(response,2),1),loadings];
end
