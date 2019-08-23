function [fomp_est, residual_omp, sparse_code] = orthogonal_pursuit(f,psi,lim)

% INPUT: sparse signal f
%        dictionary, psi
%        iteration count, lim
% OUTPUT: reconstruction, fomp_est
%         pointwise residual error, residual_omp
%         sparse vector, sparse_code
% 
% Orthogonal Matching Pursuit:
% Optimized MP algorithm to compute the sparse code of f
% on a dictionary D. MP solves min_x || f - Dx ||^2
% subject to || x ||_0 < N
%
%
% Author: Abijith J Kamath
% Website: kamath-abhijith.github.io


%% Initialization
[m,n] = size(psi);
fomp_est = zeros(m,1);
residual_omp = f;
gs_basis = zeros(m,1); gs_gram = gs_basis;
sparse_code = zeros(n,1);

%% Greedy search
for k = 1:lim
    weights_omp = psi'*residual_omp;
    [~,idx_omp] = max(abs(weights_omp));
    sparse_code(idx_omp) = sparse_code(idx_omp) + weights_omp(idx_omp);
    
    back_prop = zeros(length(f),1);
    for l = 1:k
       back_prop = back_prop + (psi(:,idx_omp)'*gs_gram(:,l))*gs_gram(:,l);
    end
    
    gs_basis = psi(:,idx_omp) - back_prop;
    gs_basis = gs_basis/norm(gs_basis,2);
    
    update_omp = (residual_omp'*gs_basis)*gs_basis;
    fomp_est = fomp_est + update_omp;
    residual_omp = residual_omp - update_omp;
    
    gs_gram = [gs_gram gs_basis];
end
end

