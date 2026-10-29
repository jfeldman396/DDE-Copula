function x=agr_truncated_standard_normal(a,b)
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
