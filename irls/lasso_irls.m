function x = lasso_irls(A,b,lambda,varargin)
% Solves lasso using IRLS
%   minimise lambda*||x||_1 + ||Ax-b||^2
%
%   Solves the l2 version of the l1 problem
%   by reweighting least-squares
%
% INPUT: Dictionary, A
%        Signal, b
%        l1 penalty weight, lambda
%
%        Optional input arguments:
%            Maximum number of iterations, MAX_ITER
%            Absolute error tolerance, ABSTOL
%
% OUTPUT: Sparse code, z=x
%
% Author: Abijith J Kamath
% kamath-abhijith@github.io

% Reading arguments / setting global constraints
if nargin>=4
    MAX_ITER = varargin{1};
else
    MAX_ITER = 1000;
end

if nargin>=5
    ABSTOL = varargin{2};
else
    ABSTOL = 1e-6;
end

% Preprocessing and constants
[~,n] = size(A);
AtA = A'*A;

% Initialisation
x = zeros(n,1);
W = eye(n);

% IRLS iterations
for i = 1:MAX_ITER
    xold = x;
    
    % x-update
    x = 0.5*(lambda*W+2*AtA)\(A'*b);
    
    % W-update
    W = diag(1./abs(x));
    
    % Stopping criterion
    if norm(x-xold)/norm(xold) < ABSTOL
        break;
    end    
end
end