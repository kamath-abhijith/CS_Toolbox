function [fmp_est, residual_mp] = matching_pursuit(f,psi,lim)

% INPUT: sparse signal f
%        dictionary, psi
%        iteration count, lim
% OUTPUT: reconstruction, fmp_est
%         pointwise residual error, residual_mp
%
% Author: Abijith J Kamath
% Website: kamath-abhijith.github.io
% Check website for theory


fmp_est = zeros(length(f),1);
residual_mp = f;

for k = 1:lim
    weights_mp = psi'*residual_mp;
    [~,idx_mp] = max(abs(weights_mp));

    update_mp = weights_mp(idx_mp)*psi(:,idx_mp);
    fmp_est = fmp_est + update_mp;
    residual_mp = residual_mp - update_mp;
end
end