function y=agr_log_interval(a,b)
% log(Phi(b)-Phi(a)): stable lower/upper tails and narrow intervals.
a=a(:);b=b(:);assert(numel(a)==numel(b)&&all(a<b));y=zeros(size(a));
left=b<=0;right=a>=0;middle=~left&~right;
if any(left),u=logPhi(b(left));v=logPhi(a(left));y(left)=u+log1minus(v-u);end
if any(right),u=logPhi(-a(right));v=logPhi(-b(right));y(right)=u+log1minus(v-u);end
if any(middle)
 aa=a(middle);bb=b(middle);tail=.5*erfc(-aa/sqrt(2))+.5*erfc(bb/sqrt(2));y(middle)=log1p(-tail);
end
width=b-a;mid=a/2+b/2;narrow=isfinite(width)&width.*(1+abs(mid))<1e-5;
% In this regime the omitted next term is O((width*(1+|mid|))^4)<1e-20.
% This is a roundoff-safe evaluation of the same normal interval mass.
if any(narrow),h=width(narrow);m=mid(narrow);y(narrow)=-.5*m.^2-.5*log(2*pi)+log(h)+log1p((m.^2-1).*h.^2/24);end
assert(all(isfinite(y)|y==-Inf),'AGR:LogIntervalNaN');
end
function y=logPhi(x)
y=zeros(size(x));negative=x<0;y(negative)=log(.5)+log(erfcx(-x(negative)/sqrt(2)))-.5*x(negative).^2;
y(~negative)=log1p(-.5*erfc(x(~negative)/sqrt(2)));y(x==-Inf)=-Inf;y(x==Inf)=0;
end
function y=log1minus(x)
x=min(x,0);y=zeros(size(x));near=x>-log(2);y(near)=log(-expm1(x(near)));y(~near)=log1p(-exp(x(~near)));
end
