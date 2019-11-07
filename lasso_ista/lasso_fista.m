function x = lasso_fista(A,b,lambda,varargin)
% Solves lasso using FISTA
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
% For more information, check: Beck et. al.
% https://people.rennes.inria.fr/Cedric.Herzet/Cedric.Herzet ...
% /Sparse_Seminar/Entrees/2012/11/12_A_Fast_Iterative_Shrinkage-Thresholding ...
% _Algorithmfor_Linear_Inverse_Problems_(A._Beck,_M._Teboulle)_files/Breck_2009.pdf

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

alpha = 1/max(eig(A'*A));
y = x; t = 1;

% ISTA iterations
for k = 1:MAX_ITER
    xold = x;
    
    % x-update
    y = y + A'*(b-A*y)*alpha;
    x = shrinkage(y,lambda*alpha);
    
    % update step size
    told = t;
    t = 0.5*(1+sqrt(1+4*t^2));
    
    % y-update
    y = x + ((told-1)/t)*(x-xold);
    
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