function [B,info]=sob_cm(X,Z,B,w,gamma,S)
o=struct('sample_count',1,'tolerance',1e-6,'max_iterations',1000,'statsG',S.G,'statsH',S.H','statsZZ',S.ZZ);[B,info]=sao_cm('gaussian',X,Z,B,w,gamma,o);
end
