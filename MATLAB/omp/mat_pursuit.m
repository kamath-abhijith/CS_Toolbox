clear
clc
close all

%% Load signal and dictionary
load_dict_f = load('hw1problem3.mat');
f = load_dict_f.f;
psi = load_dict_f.Psi;

%% Matching Pursuit
sparsemp_code = matching_pursuit(f,psi,32);
fmp_est = psi*sparsemp_code;

%% Orthogonal Matching Pursuit
sparseomp_code = orthogonal_pursuit(f,psi,32);
fomp_est = psi*sparseomp_code;

%% Plots
figure, subplot(2,2,1)
stem(fmp_est,'-r',"LineWidth",2)
hold on, grid on
stem(f,'--b',"LineWidth",1)
legend('Ground Truth','Reconstruction')
xlabel('n'), ylabel('f(n)')
title('Reconstruction using Matching Pursuit')

subplot(2,2,2)
stem(sparsemp_code,'-b',"LineWidth",2)
grid on
xlabel('k')
title('Sparse Code found using Matching Pursuit')

subplot(2,2,3)
stem(fomp_est,'-r',"LineWidth",2)
hold on, grid on
stem(f,'--b',"LineWidth",1)
legend('Ground Truth','Reconstruction')
xlabel('n'), ylabel('f(n)')
title('Reconstruction using Orthogonal Matching Pursuit')

subplot(2,2,4)
stem(sparseomp_code,'-b',"LineWidth",2)
grid on
xlabel('k')
title('Sparse Code found using Orthogonal Matching Pursuit')