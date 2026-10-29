function [B,info]=cpd_weighted_l1_cm(kind,X,Y,B,weights,variance,options)
%DDE_WEIGHTED_L1_CM Bounded convex Gaussian/logistic loading CM updates.
% All coefficients are free; exact zeros can enter or leave the solution.
% Objective: mean-over-MC-samples negative complete log likelihood + weighted
% L1 penalty. Gaussian rows use cyclic coordinate descent. Logistic rows use
% proximal Newton with backtracking and a globally safe proximal fallback.
if nargin<7, options=struct; end
if ~isfield(options,'sample_count'), options.sample_count=1; end
if ~isfield(options,'tolerance'), options.tolerance=1e-5; end
if ~isfield(options,'max_iterations'), options.max_iterations=200; end
[D,P]=size(B); assert(size(X,2)==P && isequal(size(Y),[size(X,1),D]));
if size(weights,1)==1, weights=repmat(weights,D,1); end
assert(isequal(size(weights),size(B)) && all(weights>=0 & isfinite(weights),'all'));
assert(all(weights(:,1)==0),'DDE:InterceptPenalty','Intercept must be unpenalized.');
assert(all(isfinite(X),'all') && all(isfinite(Y),'all') && all(isfinite(B),'all'));
assert(options.sample_count>=1 && options.sample_count==fix(options.sample_count));
lower=-10*ones(1,P); upper=10*ones(1,P);
if strcmp(kind,'logistic'), lower(1)=-5; end
B=min(max(B,lower),upper); start=B; C=options.sample_count;
G=(X'*X)/C;
info=struct('converged',false(D,1),'iterations',zeros(D,1), ...
    'scaled_kkt_residual',inf(D,1),'objective_before',zeros(D,1), ...
    'objective_after',zeros(D,1),'rejected',false(D,1),'fallbacks',zeros(D,1),'line_search_halvings',zeros(D,1));
if strcmp(kind,'gaussian')
    variance=variance(:); assert(numel(variance)==D && all(variance>0));
    H=(Y'*X)/C;
    scale=max(1,max(abs(H)./variance,[],2)+max(weights,[],2));
    for iteration=1:options.max_iterations
        for j=1:P
            if G(j,j)<=0
                B(weights(:,j)>0,j)=0;
                continue;
            end
            rho=H(:,j)-B*G(:,j)+G(j,j)*B(:,j);
            B(:,j)=min(max(soft(rho,variance.*weights(:,j))/G(j,j),lower(j)),upper(j));
        end
        if mod(iteration,5)==0 || iteration==options.max_iterations
            residual=kkt(B,(B*G-H)./variance,weights,lower,upper)./scale;
            info.scaled_kkt_residual=residual;
            newly=residual<=options.tolerance & ~info.converged;
            info.iterations(newly)=iteration; info.converged=residual<=options.tolerance;
            if all(info.converged), break; end
        end
    end
    info.iterations(~info.converged)=iteration;
elseif strcmp(kind,'logistic')
    assert(all(Y>=0 & Y<=1,'all'));
    lipschitz=max(0.25*max(eig((G+G')/2)),1e-8);
    for row=1:D
        b=B(row,:)'; y=Y(:,row); w=weights(row,:)';
        scale=max(1,max(abs(X'*y/C))+max(w));
        for iteration=1:options.max_iterations
            eta=X*b; probability=dde_logistic(eta);
            gradient=X'*(probability-y)/C;
            residual=kkt(b',gradient',w',lower,upper)/scale;
            if residual<=options.tolerance
                info.converged(row)=true; break;
            end
            curvature=probability.*(1-probability);
            Hessian=(X'*(X.*curvature))/C;
            Hessian=Hessian+1e-8*(1+max(diag(Hessian)))*eye(P);
            target=Hessian*b-gradient;
            candidate=quadratic_cd(Hessian,target,b,w,lower',upper',options.tolerance/10);
            direction=candidate-b;
            descent=gradient'*direction+sum(w.*(abs(candidate)-abs(b)));
            old=logistic_value(X,y,b,w,C); accepted=false;
            if descent<0
                step=1;
                for attempt=1:25
                    trial=b+step*direction;
                    if logistic_value(X,y,trial,w,C)<=old+1e-4*step*descent+1e-12
                        b=trial; accepted=true; break;
                    end
                    info.line_search_halvings(row)=info.line_search_halvings(row)+1;
                    step=step/2;
                end
            end
            if ~accepted
                info.fallbacks(row)=info.fallbacks(row)+1;
                % Global upper curvature gives a valid proximal gradient
                % update even if a Newton surrogate did not converge.
                b=min(max(soft(b-gradient/lipschitz,w/lipschitz),lower'),upper');
            end
        end
        probability=dde_logistic(X*b); gradient=X'*(probability-y)/C;
        info.scaled_kkt_residual(row)=kkt(b',gradient',w',lower,upper)/scale;
        info.converged(row)=info.scaled_kkt_residual(row)<=options.tolerance;
        info.iterations(row)=iteration; B(row,:)=b';
    end
else
    error('DDE:CMKind','Unknown loading likelihood %s.',kind);
end
for row=1:D
    if strcmp(kind,'gaussian')
        info.objective_before(row)=sum((Y(:,row)-X*start(row,:)').^2)/(2*variance(row)*C) ...
            +sum(weights(row,:).*abs(start(row,:)));
        info.objective_after(row)=sum((Y(:,row)-X*B(row,:)').^2)/(2*variance(row)*C) ...
            +sum(weights(row,:).*abs(B(row,:)));
    else
        info.objective_before(row)=logistic_value(X,Y(:,row),start(row,:)',weights(row,:)',C);
        info.objective_after(row)=logistic_value(X,Y(:,row),B(row,:)',weights(row,:)',C);
    end
    if info.objective_after(row)>info.objective_before(row)+1e-9*(1+abs(info.objective_before(row)))
        B(row,:)=start(row,:); info.rejected(row)=true; info.converged(row)=false;
        info.objective_after(row)=info.objective_before(row);
    end
end
end

function b=quadratic_cd(H,target,b,w,lower,upper,tolerance)
scale=max(1,max(abs(target))+max(w));
for iteration=1:200
    for j=1:numel(b)
        rho=target(j)-H(j,:)*b+H(j,j)*b(j);
        b(j)=min(max(soft(rho,w(j))/H(j,j),lower(j)),upper(j));
    end
    if mod(iteration,5)==0 && kkt(b',(H*b-target)',w',lower',upper')<=tolerance*scale
        break;
    end
end
end

function result=soft(x,threshold)
result=sign(x).*max(abs(x)-threshold,0);
end

function residual=kkt(B,gradient,weights,lower,upper)
total=gradient+weights.*sign(B);
zero=B==0;
total(zero)=sign(gradient(zero)).*max(abs(gradient(zero))-weights(zero),0);
atLower=B<=lower+1e-10; atUpper=B>=upper-1e-10;
total(atLower)=min(total(atLower),0); total(atUpper)=max(total(atUpper),0);
residual=max(abs(total),[],2);
end

function value=logistic_value(X,y,b,w,C)
eta=X*b;
value=sum(dde_softplus(eta)-y.*eta)/C+sum(w.*abs(b));
end
