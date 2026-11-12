function data=r1t_data(family,n,J,rep)
% Known continuous uniforms retained; ONLY count Y enters estimation.
seed=2091000+1000*rep+sum(double(family));rng(seed,'twister');m=n+2000;rho=sin(pi*.5/2);theta=2;nu=4;
switch family
 case {'clayton','survival_clayton'}
  V=gamrnd(1/theta,1,m,1);U=(1-log(rand(m,J))./V).^(-1/theta);
  if strcmp(family,'survival_clayton'),U=1-U;end
 case {'student_t','gaussian'}
  X=sqrt(rho)*randn(m,1)+sqrt(1-rho)*randn(m,J);
  if strcmp(family,'student_t'),X=X./sqrt(chi2rnd(nu,m,1)/nu);U=tcdf(X,nu);else,U=normcdf(X);end
 otherwise,error('Unknown family');
end
means=linspace(4,8,J);Y=poissinv(U,repmat(means,m,1));assert(all(isfinite(Y),'all'));
data=struct('family',family,'seed',seed,'J',J,'n',n,'replication',rep,'Ytrain',Y(1:n,:),'Utrain',U(1:n,:),'Ytest',Y(n+1:end,:),'Utest',U(n+1:end,:),'means',means,'theta',theta,'rho',rho,'nu',nu);
end
