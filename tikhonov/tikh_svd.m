function x = tikh_svd(A,b,lambda)
% Solves Tikhonov regularization in closed form using SVD
%   minimise lambda*||x||_2^2 + (1/2)||Ax-b||^2
%
% INPUT: Dictionary, A
%        Signal, b
%        Smoothness parameter, lambda
% OUTPUT: Sparse code, x
%
% Author: Abijith J Kamath
% kamath-abhijith@github.io

% Initialise
[m,n] = size(A);
[U,S,V] = svd(A);
s = diag(S);

% Solver in closed form
x = V(:,1:m)*(s.*(U'*b)./(s.^2+lambda^2));    

end