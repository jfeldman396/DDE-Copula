function [A1,A2,stats]=pcp_binary(A1,A2,Z,B1,B2,gamma,prop,b,sweeps,beta,lambda,method,options)
%PCP_BINARY Layer 1 random blocks; layer 2 always whole-layer joint CP.
% Each observation has its OWN state-independent permutation per layer/sweep.
% Top layer precedes bottom; blocks update immediately. b=Inf means full layer.
% beta is INVERSE temperature: beta=1 gives the ordinary latent conditional.
if nargin<13, options=struct; end
if ~isfield(options,'random_order'), options.random_order=true; end
if ~isfield(options,'record_states'), options.record_states=false; end
if ~isfield(options,'record_blocks'), options.record_blocks=false; end
[N,~,K1,K2]=dde_assert_binary_model(A1,A2,Z,B1,B2,gamma,prop);
assert(beta>0 && isfinite(beta) && sweeps>=1 && sweeps==fix(sweeps));
assert(b>=1 && (b==fix(b) || isinf(b)));
timer=tic; topSeconds=0; A1=double(A1); A2=double(A2);
L=B1(:,2:end); W=L./gamma(:); Q=L'*W;
base=(Z-B1(:,1)')*W;
stats=struct('switchCount1',0,'switchCount2',0,'multi_blocks',0,'eligible_multi_blocks',0, ...
    'visited1',0,'visited2',0,'beta',beta,'block_size',b,'sweeps',sweeps);
if options.record_states, stats.states=false(N,K1+K2,sweeps); end
if options.record_blocks
    stats.blocks={}; stats.probability1=zeros(N,K1); stats.probability2=zeros(N,K2);
end
for sweep=1:sweeps

    eta=[ones(N,1),A2]*B2'; field=base-A1*Q;
    for layer=[2 1]
        if layer==2, K=K2; else, K=K1; end
        if K==0, continue; end
        if layer==2
            topClock=tic;
            oldTop=A2;
            [A2,topStats]=pcp_top(A1,A2,B2,prop,beta,lambda);
            stats.top_joint_width=topStats.joint_width;
            stats.top_enumerated_width=topStats.enumerated_width;
            stats.switchCount2=stats.switchCount2+sum(abs(A2-oldTop),'all');
            stats.visited2=stats.visited2+N*K2;
            eta=[ones(N,1),A2]*B2'; topSeconds=topSeconds+toc(topClock); continue;
        end
        if options.random_order, [~,order]=sort(rand(N,K),2);
        else, order=repmat(1:K,N,1); end
        if K<=6, step=K; else, step=min(b,K); end
        for start=1:step:K
            S=order(:,start:min(start+step-1,K)); m=size(S,2);
            bits=double(dec2bin(0:2^m-1,m)-'0'); C=size(bits,1);
            if options.record_blocks, before1=A1; before2=A2; end
            if layer==2
                old=gather_rows(A2,S);
                if K2<=10 && ~strcmp(method,'P') && ~(strcmp(method,'CP') && lambda<1)
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
            if layer==2 && lambda<1
                probability=dde_dependence_mixture(probability,bits,lambda); drawProbability=probability;
            elseif layer==1 && lambda<1 && ismember(method,{'C1','C2','CP'})
                [probability,drawProbability]=bl_mix(logw,probability,bits,old,lambda,method);
            else, drawProbability=probability; end
            cdf=cumsum(drawProbability,2); cdf(:,end)=1;
            % Reversed uniform agrees with existing Bernoulli rand<p at b=1.
            chosen=1+sum((1-rand(N,1))>cdf,2); next=bits(chosen,:); delta=next-old;
            changes=sum(abs(delta),2);
            if m>1
                stats.eligible_multi_blocks=stats.eligible_multi_blocks+N;
                stats.multi_blocks=stats.multi_blocks+sum(changes>1);
            end
            if layer==2
                A2=scatter_rows(A2,S,next);
                for k=1:m, eta=eta+delta(:,k).*B2(:,S(:,k)+1)'; end
                stats.switchCount2=stats.switchCount2+sum(changes); stats.visited2=stats.visited2+N*m;
                topSeconds=topSeconds+toc(topClock);
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
stats.switching1=stats.switchCount1/max(1,N*K1*sweeps);
stats.switching2=stats.switchCount2/max(1,N*K2*sweeps);
stats.multi_fraction=stats.multi_blocks/max(1,stats.eligible_multi_blocks);
stats.top_conditional_seconds=topSeconds; stats.runtime_seconds=toc(timer); stats.seconds_per_sweep=stats.runtime_seconds/sweeps;
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
