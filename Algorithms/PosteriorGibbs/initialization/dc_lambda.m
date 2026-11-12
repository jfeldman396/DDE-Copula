function lambda=dc_lambda(method,it,warm)
switch method
    case 'G', lambda=1;
    case {'P','I'}, lambda=0;
    case 'F', lambda=.5;
    case 'C1', lambda=min(1,(it-1)/(warm-1));
    case 'C2', lambda=.5+.5*min(1,(it-1)/(warm-1));
    otherwise, error('Unknown dependence method');
end
end
