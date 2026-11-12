function value=cf_lambda(method,it,warm)
if strcmp(method,'CP'), method='C1'; end
value=dc_lambda(method,it,warm);
end
