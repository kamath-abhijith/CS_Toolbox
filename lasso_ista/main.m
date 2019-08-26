clear
clc
close all

%% Load signal and dictionary
load_dict_f = load('hw1problem3.mat');
f = load_dict_f.f;
psi = load_dict_f.Psi; [m,n] = size(psi);

%% Solve using lasso and matching pursuit
x_admm = lasso_admm(psi,f,1,1.5);
x_ista = lasso_ista(psi,f,0.05);

%% Plots
figure, subplot(2,2,1)
stem(f,'--b',"LineWidth",2)
hold on, grid on
stem(psi*x_admm,'-r',"LineWidth",1)
title('Reconstruction - LASSO, ADMM')
xlabel('n'), ylabel('f(n)')
axis([0 m -1.5 1.5])

subplot(2,2,3)  
stem(x_admm,'-b',"LineWidth",2), grid on
title('Sparse Code - LASSO, ADMM')
xlabel('n'), ylabel('x')
axis([0 n -1.5 1.5])

subplot(2,2,2)
stem(f,'--b',"LineWidth",2)
hold on, grid on
stem(psi*x_ista,'-r',"LineWidth",1)
title('Reconstruction - LASSO, ISTA')
xlabel('n'), ylabel('f(n)')
axis([0 m -1.5 1.5])

subplot(2,2,4)
stem(x_ista,'-b',"LineWidth",2), grid on
title('Sparse Code - LASSO, ISTA')
xlabel('n'), ylabel('x')
axis([0 n -1.5 1.5])