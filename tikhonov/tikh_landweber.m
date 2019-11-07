function x = tikh_landweber(A,b,lambda,varargin)
% Solves Tikhonov regularization using Majorizer-Minimization
%   minimise lambda*||x||_2^2 + (1/2)||Ax-b||^2
%
% INPUT: Dictionary, A
%        Signal, b
%        Smoothness parameter, lambda
%        Variable input argument list for
%            Maximum iterations, MAX_ITER
%            Absolute tolerance, ABSTOL
%            Step size, alpha
%
% OUTPUT: Solution, x
%
% Author: Abijith J Kamath
% kamath-abhijith@github.io

% Global constraints
if nargin >=4
    MAX_ITER = varargin{1};
else
    MAX_ITER = 1000;
end

if nargin >=5
    ABSTOL = varargin{2};
else
    ABSTOL = 1e-6;
end

if nargin >=6
    alpha = varargin{3};
else
    alpha = max(eig(A'*A));

% Initialise
[m,n] = size(A);
x = zeros(n,1);
beta = alpha + lambda;

% Landweber iterations
for k = 1:MAX_ITER
   xold = x;
    
   % Update solution
   x =  (alpha/beta)*x + (1/beta)*A'*(b-A*x);
   
   % Stopping conditon
   if (norm(x-xold)/norm(xold)) < ABSTOL
       break;
   end    
end
end