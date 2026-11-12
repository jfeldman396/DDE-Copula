function Z = dde_rank_gibbs(Z,R,A1,B1,gamma)
%DDE_RANK_GIBBS One exact rank-constrained Gaussian block sweep, at temp=1.
% Tied observations are conditionally independent within their rank block.
% Rank blocks are visited sequentially, using their current adjacent bounds.
% Rejection sampling avoids CDF clipping, including in extreme tails.
assert(size(A1,1)==size(Z,1) && size(B1,2)==size(A1,2)+1);
gamma=gamma(:)';
assert(numel(gamma)==size(Z,2) && all(gamma>0 & isfinite(gamma)));
dde_assert_rank_consistency(Z,R);
mu=[ones(size(A1,1),1),A1]*B1';
for j=1:size(Z,2)
    sd=sqrt(gamma(j));
    missing=isnan(R(:,j));
    Z(missing,j)=mu(missing,j)+sd*randn(sum(missing),1);
    ranks=unique(R(~missing,j),'sorted');
    groups=cell(numel(ranks),1);
    for r=1:numel(ranks), groups{r}=find(R(:,j)==ranks(r)); end
    for r=1:numel(ranks)
        ids=groups{r};
        lb=-inf; ub=inf;
        if r>1, lb=max(Z(groups{r-1},j)); end
        if r<numel(ranks), ub=min(Z(groups{r+1},j)); end
        a=(lb-mu(ids,j))/sd; b=(ub-mu(ids,j))/sd;
        z=mu(ids,j)+sd*truncated_standard_normal(a,b);
        % A representable interior value is required. Resampling handles
        % rounding to a boundary without shifting/clipping the draw.
        bad=~(z>lb & z<ub);
        tries=0;
        while any(bad)
            tries=tries+1;
            assert(tries<100, 'DDE:RankPrecision', ...
                'No representable interior Gaussian draw; rank bounds too close.');
            z(bad)=mu(ids(bad),j)+sd*truncated_standard_normal(a(bad),b(bad));
            bad=~(z>lb & z<ub);
        end
        Z(ids,j)=z;
    end
end
dde_assert_rank_consistency(Z,R);
end

function x=truncated_standard_normal(a,b)
assert(all(a<b),'DDE:TruncationBounds','Empty truncation interval.');
finite=isfinite(a)&isfinite(b); midpoint=a(finite)/2+b(finite)/2;
assert(all(midpoint>a(finite) & midpoint<b(finite)), ...
    'DDE:RankPrecision','Truncation interval has no representable midpoint.');
x=zeros(size(a));
negative=b<=0;
if any(negative)
    x(negative)=-positive_tail(-b(negative),-a(negative));
end
positive=a>=0;
if any(positive), x(positive)=positive_tail(a(positive),b(positive)); end
middle=~negative & ~positive;
ids=find(middle);
rounds=0;
while ~isempty(ids)
    rounds=rounds+1;
    assert(rounds<=10000,'DDE:RankPrecision','Truncated Gaussian rejection limit.');
    aa=a(ids); bb=b(ids);
    narrow=(bb-aa)<2;
    proposal=randn(size(ids));
    proposal(narrow)=aa(narrow)+(bb(narrow)-aa(narrow)).*rand(sum(narrow),1);
    accept=proposal>aa & proposal<bb;
    accept(narrow)=accept(narrow) & ...
        log(rand(sum(narrow),1)) < -0.5*proposal(narrow).^2;
    x(ids(accept))=proposal(accept);
    ids=ids(~accept);
end
end

function x=positive_tail(a,b)
% Exponential proposal restricted to [a,b]. Its acceptance ratio is bounded
% by one and proportional to the target normal density / proposal density.
x=zeros(size(a));
alpha=a/2+hypot(a,2)/2;
ids=(1:numel(a))';
rounds=0;
while ~isempty(ids)
    rounds=rounds+1;
    assert(rounds<=10000,'DDE:RankPrecision','Truncated tail rejection limit.');
    aa=a(ids); al=alpha(ids); width=b(ids)-aa;
    mass=-expm1(-al.*width);
    proposal=aa-log1p(-rand(size(ids)).*mass)./al;
    accept=proposal>aa & proposal<b(ids) & ...
        log(rand(size(ids))) < -0.5*(proposal-al).^2;
    x(ids(accept))=proposal(accept);
    ids=ids(~accept);
end
end
