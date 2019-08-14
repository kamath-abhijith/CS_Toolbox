clear
clc
close all

%% Load signal and dictionary
load_dict_f = load('hw1problem3.mat');
f = load_dict_f.f;
psi = load_dict_f.Psi;

%% Matching Pursuit
fmp_est = zeros(length(f),1);
residual_mp = f;
k = 0;

for k = 1:50
    weights_mp = psi'*residual_mp;
    [~,idx_mp] = max(abs(weights_mp));

    update_mp = weights_mp(idx_mp)*psi(:,idx_mp);
    fmp_est = fmp_est + update_mp;
    residual_mp = residual_mp - update_mp;
end

%% Orthogonal Matching Pursuit
fomp_est = zeros(length(f),1);
residual_omp = f;
gs_basis = zeros(length(f),1); gs_gram = gs_basis;

for k = 1:10
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
    
%% Plots
figure, subplot(2,1,1)
stem(fmp_est,'-r',"LineWidth",2)
hold on, grid on
stem(f,'--b',"LineWidth",1)
legend('Ground Truth','Reconstruction')
xlabel('n'), ylabel('f(n)')
title('Reconstruction using Matching Pursuit')

subplot(2,1,2)
stem(fomp_est,'-r',"LineWidth",2)
hold on, grid on
stem(f,'--b',"LineWidth",1)
legend('Ground Truth','Reconstruction')
xlabel('n'), ylabel('f(n)')
title('Reconstruction using Orthogonal Matching Pursuit')