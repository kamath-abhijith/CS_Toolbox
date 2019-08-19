function [fomp_est, residual_omp] = orthogonal_pursuit(f,psi,lim)

% INPUT: sparse signal f
%        dictionary, psi
%        iteration count, lim
% OUTPUT: reconstruction, fomp_est
%         pointwise residual error, residual_omp
%
% Author: Abijith J Kamath
% Website: kamath-abhijith.github.io
% Check website for theory


fomp_est = zeros(length(f),1);
residual_omp = f;
gs_basis = zeros(length(f),1); gs_gram = gs_basis;

for k = 1:lim
    weights_omp = psi'*residual_omp;
    [~,idx_omp] = max(abs(weights_omp));
    
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

