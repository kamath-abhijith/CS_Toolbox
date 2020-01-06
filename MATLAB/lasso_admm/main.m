clear
clc
close all

%% Load signal and dictionary
load_dict_f = load('hw1problem3.mat');
f = load_dict_f.f;
psi = load_dict_f.Psi;

%% Solve using lasso and matching pursuit
x = lasso_admm(psi,f,1,1.5);
sparse_mpcode = matching_pursuit(f,psi,64);

%% Plots
figure, subplot(2,2,1)
stem(f,'--b',"LineWidth",2)
hold on, grid on
stem(psi*x,'-r',"LineWidth",1)
title('Reconstruction - Basis Pursuit')
xlabel('n'), ylabel('f(n)')

subplot(2,2,3)  
stem(x,'-b',"LineWidth",2), grid on
title('Sparse Code - Basis Pursuit')
xlabel('n'), ylabel('x')

subplot(2,2,2)
stem(f,'--b',"LineWidth",2)
hold on, grid on
stem(psi*sparse_mpcode,'-r',"LineWidth",1)
title('Reconstruction - Matching Pursuit')
xlabel('n'), ylabel('f(n)')

subplot(2,2,4)
stem(sparse_mpcode,'-b',"LineWidth",2), grid on
title('Sparse Code - Matching Pursuit')
xlabel('n'), ylabel('x')