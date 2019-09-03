function x = lasso_ista(A,b,lambda,varargin)
% Solves lasso using ISTA
%   minimise lambda*||x||_1 + (1/2)||Ax-b||^2
%
%   Updates on opt. variable are soft thresholding
%   operations
%
% INPUT: Dictionary, A
%       Signal, b
%       l1 penalty weight, lambda
%
%       Optional input arguments:
%           Maximum number of iterations, MAX_ITER
%           Absolute error tolerance, ABSTOL
%
% OUTPUT: Sparse code, z=x
%
% Author: Abijith J Kamath
% kamath-abhijith@github.io
%
% For more information, check: Daubechies et. al.
% https://onlinelibrary.wiley.com/doi/abs/10.1002/cpa.20042

% Reading arguments / setting global constraints
if nargin>=5
    MAX_ITER = varargin{1};
else
    MAX_ITER = 1000;
end

if nargin>=6
    ABSTOL = varargin{2};
else
    ABSTOL = 1e-6;
end

% Preprocessing and zero initialization
[~,n] = size(A);
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