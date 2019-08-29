clear
clc
close all

%% Initialise
A = randn(10,50);
x = 10*randn(50,1);
b = A*x;

%% Solvers
x_pinv = pinv(A)*b;
x_tikh = tikh_svd(A,b,0.01);

%% Error metrics
MSE_PINV = (norm(x-x_pinv)/norm(x))^2;
MSE_PINV_DB = 10*log10(MSE_PINV);

MSE_TIKH = (norm(x-x_tikh)/norm(x))^2;
MSE_TIKH_DB = 10*log10(MSE_TIKH);

%% Plots
figure, subplot(2,2,1)
stem(b,'-g',"LineWidth",2), hold on, grid on
stem(A*x_pinv,'--b',"LineWidth",2)
stem(A*x_tikh,'--r',"LineWidth",2)
xlabel('$n$','Interpreter','latex')
ylabel('$b$','Interpreter','latex')
title('Signal')

subplot(2,2,2)
stem(x,'-g',"LineWidth",2), grid on
xlabel('$n$','Interpreter','latex')
ylabel('$x$','Interpreter','latex')
title('Original Sparse Code')

subplot(2,2,3)
stem(x_pinv,'-b',"LineWidth",2), grid on
xlabel('$n$','Interpreter','latex')
ylabel('$\tilde{x}^{PINV}$','Interpreter','latex')
title('Reconstructed Sparse Code using MPI')

subplot(2,2,4)
stem(x_tikh,'-r',"LineWidth",2), grid on
xlabel('$n$','Interpreter','latex')
ylabel('$\tilde{x}^{TIKH}$','Interpreter','latex')
title('Reconstructed Sparse Code using Tikhonov')