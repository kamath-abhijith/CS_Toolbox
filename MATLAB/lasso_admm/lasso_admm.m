function [z,obj] = lasso_admm(A,b,rho,alpha,varargin)
% Solves lasso using ADMM
%   minimise ||x||_1 + (1/2)||Ax-b||^2
%
%   ADMM form: (1/2)||Ax-b||^2 + ||z||_1
%   subject to x-z=0
%
%   Updates on opt. variable are projections onto
%   the column space Ax=b
%
%   Object obj is contains the history of the
%   updates
%
% INPUT: Dictionary, A
%       Signal, b
%       Augmented Lagrange parameter, rho
%       Relaxation weight, alpha (typically in 1-1.8)
%       
%       Optional input arguments:
%           Maximum number of iterations, MAX_ITER
%           Absolute error tolerance, ABSTOL
%           Relative error tolerance, RELTOL
%
% OUTPUT: Sparse code, z=x
%         History object, obj
%
% Author: Abijith J Kamath
% kamath-abhijith@github.io
%
% For more information, check: Lectures from Boyd
% https://web.stanford.edu/~boyd/papers/pdf/admm_slides.pdf

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

if nargin>=7
    RELTOL = varargin{3};
else
    RELTOL = 1e-4;
end

% Preprocessing and zero initialization
[m,n] = size(A);
x = zeros(n,1);
u = zeros(n,1);
z = zeros(n,1);

% Static variables
AAt = A*A';
P = eye(n) - A'*(AAt\A);
q = A'*(AAt\b);

% ADMM iterations
for k = 1:MAX_ITER
   % x-update
   x = P*(z-u) + q;
   
   % z-update with relaxation
   zold = z;
   x_hat = alpha*x + (1-alpha)*zold;
   z = shrinkage(x+u,1/rho);
   
   % dual with change of variable update
   u = u + (x_hat-z);
   
   % Stopping criterion
   obj.valobj(k) = norm(x,1);
   
   obj.rnorm(k) = norm(x-z);
   obj.snorm(k) = norm(rho*(zold-z));
   
   obj.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
   obj.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u);
   
   if obj.rnorm(k)<obj.eps_pri(k) && obj.snorm(k)<obj.eps_dual(k)
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