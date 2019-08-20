function [fmp_est, residual_mp, sparse_code] = matching_pursuit(f,psi,lim)

% INPUT: sparse signal f
%        dictionary, psi
%        iteration count, lim
% OUTPUT: reconstruction, fmp_est
%         pointwise residual error, residual_mp
%         sparse vector, sparse_code
% 
% Matching Pursuit:
% Greedy algorithm to compute the sparse code of f on a
% dictionary D. MP solves min_x || f - Dx ||^2 subject to
% || x ||_0 < N
% 
%
% Author: Abijith J Kamath
% Website: kamath-abhijith.github.io


%% Initialization
[m,n] = size(psi);
fmp_est = zeros(m,1);
residual_mp = f;
sparse_code = zeros(n,1);

%% Greedy search
for k = 1:lim
    weights_mp = psi'*residual_mp;
    [~,idx_mp] = max(abs(weights_mp));
    sparse_code(idx_mp) = sparse_code(idx_mp) + weights_mp(idx_mp);
    
    update_mp = weights_mp(idx_mp)*psi(:,idx_mp);
    fmp_est = fmp_est + update_mp;
    residual_mp = residual_mp - update_mp;
end
end