function x = lasso_ista(A,b,lambda)
% Solves lasso using ISTA
%   minimise lambda*||x||_1 + (1/2)||Ax-b||^2
%
%   Updates on opt. variable are soft thresholding
%   operations
%
% INPUT: Dictionary, A
%       Signal, b
%       l1 penalty weight, lambda
% OUTPUT: Sparse code, z=x
%
% Author: Abijith J Kamath
% kamath-abhijith@github.io
%
% For more information, check: Daubechies et. al.
% https://onlinelibrary.wiley.com/doi/abs/10.1002/cpa.20042

% Global constraints
MAX_ITER = 500000;
ABSTOL = 1e-10;

% Preprocessing and zero initialization
[m,n] = size(A);
x = zeros(n,1);
t = 1/max(eig(A'*A));

% ISTA iterations
for k = 1:MAX_ITER
    xold = x;
    
    % x-update
    x = shrinkage(x+t*A'*(b-A*x),lambda*t);
    
    % Stopping criterion
    if (norm(x-xold)/norm(xold)) < ABSTOL
        break;
    end
end
end

function s = shrinkage(v,a)
% Soft thresholding
%
% INPUT:  Thersholding variable, v
%         Shrinkage factor, a
%
% OUTPUT: Output of soft thresholding

    s = max(0,v-a) - max(0,-v-a);
end