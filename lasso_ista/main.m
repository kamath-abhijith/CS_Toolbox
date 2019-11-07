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
x_fista = lasso_fista(psi,f,0.05);

%% Error metrics
MSE_ADMM = (norm(f-psi*x_admm))/norm(f)^2;
MSE_ADMM_DB = 10*log10(MSE_ADMM);

MSE_ISTA = (norm(f-psi*x_ista))/norm(f)^2;
MSE_ISTA_DB = 10*log10(MSE_ISTA);

MSE_FISTA = (norm(f-psi*x_fista))/norm(f)^2;
MSE_FISTA_DB = 10*log10(MSE_FISTA);

%% Plots
figure, subplot(3,2,1)
stem(f,'--b',"LineWidth",2)
hold on, grid on
stem(psi*x_admm,'-r',"LineWidth",1)
title('Reconstruction - LASSO, ADMM')
xlabel('n'), ylabel('f(n)')
axis([0 m -1.5 1.5])

subplot(3,2,2)  
stem(x_admm,'-b',"LineWidth",2), grid on
title('Sparse Code - LASSO, ADMM')
xlabel('n'), ylabel('x')
axis([0 n -1.5 1.5])

subplot(3,2,3)
stem(f,'--b',"LineWidth",2)
hold on, grid on
stem(psi*x_ista,'-r',"LineWidth",1)
title('Reconstruction - LASSO, ISTA')
xlabel('n'), ylabel('f(n)')
axis([0 m -1.5 1.5])

subplot(3,2,4)
stem(x_ista,'-b',"LineWidth",2), grid on
title('Sparse Code - LASSO, ISTA')
xlabel('n'), ylabel('x')
axis([0 n -1.5 1.5])

subplot(3,2,5)
stem(f,'--b',"LineWidth",2)
hold on, grid on
stem(psi*x_fista,'-r',"LineWidth",1)
title('Reconstruction - LASSO, FISTA')
xlabel('n'), ylabel('f(n)')
axis([0 m -1.5 1.5])

subplot(3,2,6)
stem(x_fista,'-b',"LineWidth",2), grid on
title('Sparse Code - LASSO, FISTA')
xlabel('n'), ylabel('x')
axis([0 n -1.5 1.5])