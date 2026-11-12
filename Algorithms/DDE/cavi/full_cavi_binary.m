function [A1,A2,stats]=full_cavi_binary(A1,A2,Z,B1,B2,gamma,prop,b,sweeps,beta,lambda,method,options)
%FULL_CAVI_BINARY Original layerwise block kernel with converged CAVI product.
% Only the enabled VI product branch differs from the snapshot cmv_binary.
% Requires pilot_mix and cmv_cavi; logw includes beta on every target term.
% Each observation has its OWN state-independent permutation per layer/sweep.
% Top layer precedes bottom; blocks update immediately. b=Inf means full layer.
% beta is INVERSE temperature: beta=1 gives the ordinary latent conditional.
if nargin<13, options=struct; end
if ~isfield(options,'vi_layers'), options.vi_layers=[false false]; end
if ~isfield(options,'random_order'), options.random_order=true; end
if ~isfield(options,'record_states'), options.record_states=false; end
if ~isfield(options,'record_blocks'), options.record_blocks=false; end
[N,~,K1,K2]=dde_assert_binary_model(A1,A2,Z,B1,B2,gamma,prop);
assert(beta>0 && isfinite(beta) && sweeps>=1 && sweeps==fix(sweeps));
assert(numel(b)==2 && all(b>=1) && all(b==fix(b) | isinf(b))); b=reshape(b,1,2);
partitioned=isfield(options,'layer_partitions');
if partitioned
 assert(iscell(options.layer_partitions)&&numel(options.layer_partitions)==2);
 for layer=1:2
  sizes=reshape(double(options.layer_partitions{layer}),1,[]);width=[K1 K2];
  assert(all(sizes>=1)&&all(sizes==fix(sizes))&&sum(sizes)==width(layer),'Partitioned:CoverageMismatch','Partition sizes must cover each raw layer exactly.');
  options.layer_partitions{layer}=sizes;
 end
end
timer=tic; A1=double(A1); A2=double(A2);
vi=zeros(1,8); % rows, CP KL sum, VI KL sum, max gain, sweeps sum, max residual, seconds, max width
L=B1(:,2:end); W=L./gamma(:); Q=L'*W;
base=(Z-B1(:,1)')*W;
stats=struct('switchCount1',0,'switchCount2',0,'multi_blocks',0,'eligible_multi_blocks',0, ...
    'visited1',0,'visited2',0,'multi_blocks1',0,'multi_blocks2',0, ...
    'eligible_multi_blocks1',0,'eligible_multi_blocks2',0,'beta',beta,'block_size',b,'sweeps',sweeps);
if options.record_states, stats.states=false(N,K1+K2,sweeps); end
if options.record_blocks
    stats.blocks={}; stats.probability1=zeros(N,K1); stats.probability2=zeros(N,K2);
end
if K2<=10
    topStates=double(dec2bin(0:2^K2-1,K2)-'0');
    if K2==0, topStates=zeros(1,0); end
    etas=[ones(size(topStates,1),1),topStates]*B2';
    constants=-sum(dde_softplus(etas),2)+top_prior(topStates,prop);
    powers=2.^(K2-1:-1:0);
end
for sweep=1:sweeps
    if K2<=10, scores=A1*etas'+constants'; end
    eta=[ones(N,1),A2]*B2'; field=base-A1*Q;
    for layer=[2 1]
        if layer==2, K=K2; else, K=K1; end
        if K==0, continue; end
        if options.random_order, [~,order]=sort(rand(N,K),2);
        else, order=repmat(1:K,N,1); end
        if partitioned
            sizes=options.layer_partitions{layer};ends=cumsum(sizes);starts=[1 ends(1:end-1)+1];
        else
            step=min(b(layer),K);starts=1:step:K;ends=min(starts+step-1,K);
        end
        for block=1:numel(starts)
            S=order(:,starts(block):ends(block)); m=size(S,2);
            bits=double(dec2bin(0:2^m-1,m)-'0'); C=size(bits,1);
            if options.record_blocks, before1=A1; before2=A2; end
            if layer==2
                old=gather_rows(A2,S);
                if K2<=10
                    selectedPowers=reshape(powers(S),size(S));
                    code=A2*powers'-sum(old.*selectedPowers,2);
                    codes=code+sum(reshape(selectedPowers,N,1,m).*reshape(bits,1,C,m),3);
                    logw=scores((1:N)'+N*codes);
                else
                    eta0=eta;
                    for k=1:m, eta0=eta0-old(:,k).*B2(:,S(:,k)+1)'; end
                    logw=zeros(N,C);
                    for c=1:C
                        ec=eta0;
                        for k=1:m, ec=ec+bits(c,k)*B2(:,S(:,k)+1)'; end
                        logw(:,c)=sum(A1.*ec-dde_softplus(ec),2);
                        for k=1:m
                            pp=prop(S(:,k));
                            if bits(c,k), logw(:,c)=logw(:,c)+log(pp(:));
                            else, logw(:,c)=logw(:,c)+log1p(-pp(:)); end
                        end
                    end
                end
            else
                old=gather_rows(A1,S); h=gather_rows(field+eta,S);
                for k=1:m
                    for l=1:m
                        cross=reshape(Q(S(:,k)+K1*(S(:,l)-1)),N,1); h(:,l)=h(:,l)+old(:,k).*cross;
                    end
                    h(:,k)=h(:,k)-.5*reshape(Q(S(:,k)+K1*(S(:,k)-1)),N,1);
                end
                logw=h*bits';
                for k=1:m
                    for l=k+1:m
                        cross=reshape(Q(S(:,k)+K1*(S(:,l)-1)),N,1);
                        logw=logw-cross*(bits(:,k).*bits(:,l))';
                    end
                end
            end
            logw=beta*logw; maximum=max(logw,[],2);
            assert(all(isfinite(maximum)),'RandomBlock:ImpossibleCondition','Conditioned state outside target support.');
            probability=exp(logw-maximum); probability=probability./sum(probability,2);
            if lambda(layer)<1
                if options.vi_layers(layer)
                    [probability,drawProbability,vdiag]=pilot_mix(logw,probability,bits,old,lambda(layer),'VI');
                    vi([1 2 3 5 7])=vi([1 2 3 5 7])+vdiag([1 2 3 5 7]);
                    vi([4 6 8])=max(vi([4 6 8]),vdiag([4 6 8]));
                else
                    [probability,drawProbability]=bl_mix(logw,probability,bits,old,lambda(layer),method);
                end
            else, drawProbability=probability; end
            cdf=cumsum(drawProbability,2); cdf(:,end)=1;
            % Reversed uniform agrees with existing Bernoulli rand<p at b=1.
            chosen=1+sum((1-rand(N,1))>cdf,2); next=bits(chosen,:); delta=next-old;
            changes=sum(abs(delta),2);
            if m>1
                stats.eligible_multi_blocks=stats.eligible_multi_blocks+N;
                stats.multi_blocks=stats.multi_blocks+sum(changes>1);
                key=sprintf('eligible_multi_blocks%d',layer); stats.(key)=stats.(key)+N;
                key=sprintf('multi_blocks%d',layer); stats.(key)=stats.(key)+sum(changes>1);
            end
            if layer==2
                A2=scatter_rows(A2,S,next);
                for k=1:m, eta=eta+delta(:,k).*B2(:,S(:,k)+1)'; end
                stats.switchCount2=stats.switchCount2+sum(changes); stats.visited2=stats.visited2+N*m;
            else
                A1=scatter_rows(A1,S,next);
                for k=1:m, field=field-delta(:,k).*Q(S(:,k),:); end
                stats.switchCount1=stats.switchCount1+sum(changes); stats.visited1=stats.visited1+N*m;
            end
            if options.record_blocks
                trace=struct('layer',layer,'indices',S,'before1',before1,'before2',before2, ...
                    'bits',bits,'probability',probability,'chosen',chosen);
                stats.blocks{end+1}=trace;
                key=sprintf('probability%d',layer); stats.(key)=scatter_rows(stats.(key),S,probability*bits);
            end
        end
    end
    if options.record_states, stats.states(:,:,sweep)=[A1 A2]; end
end
if partitioned,stats.layer_partitions=options.layer_partitions;end
stats.switching1=stats.switchCount1/max(1,N*K1*sweeps);
stats.switching2=stats.switchCount2/max(1,N*K2*sweeps);
stats.multi_fraction=stats.multi_blocks/max(1,stats.eligible_multi_blocks);
for layer=1:2
    stats.(sprintf('multi_fraction%d',layer))=stats.(sprintf('multi_blocks%d',layer))/max(1,stats.(sprintf('eligible_multi_blocks%d',layer)));
end
stats.vi=vi;
stats.top_conditional_seconds=NaN; % Existing layerwise kernel times only the complete sweep.
stats.runtime_seconds=toc(timer); stats.seconds_per_sweep=stats.runtime_seconds/sweeps;
end

function value=gather_rows(A,S)
N=size(A,1); value=A((1:N)'+N*(S-1));
end
function A=scatter_rows(A,S,value)
N=size(A,1); A((1:N)'+N*(S-1))=value;
end
function value=top_prior(states,prop)
value=zeros(size(states,1),1);
for l=1:numel(prop)
    if prop(l)==0, value(states(:,l)==1)=-inf;
    elseif prop(l)==1, value(states(:,l)==0)=-inf;
    else, value=value+states(:,l)*log(prop(l))+(1-states(:,l))*log1p(-prop(l)); end
end
end
