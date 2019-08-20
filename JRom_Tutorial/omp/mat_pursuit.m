clear
clc
close all

%% Load signal and dictionary
load_dict_f = load('hw1problem3.mat');
f = load_dict_f.f;
psi = load_dict_f.Psi;

%% Matching Pursuit
[fmp_est, residual_mp, sparse_mpcode] = matching_pursuit(f,psi,32);

%% Orthogonal Matching Pursuit
[fomp_est, residual_omp, sparse_ompcode] = orthogonal_pursuit(f,psi,32);    

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