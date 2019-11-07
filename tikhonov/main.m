clear
clc
close all

%% Initialise
A = randn(50,40);
x = 10*randn(40,1);
b = A*x + 5*randn(50,1);

%% Solvers
x_pinv = pinv(A)*b;
x_tikhsvd = tikh_svd(A,b,0.01);
x_tikhlwb = tikh_landweber(A,b,0.01,10000,1e-36);

%% Error metrics
MSE_PINV = (norm(A*(x-x_pinv))/norm(x))^2;
MSE_PINV_DB = 10*log10(MSE_PINV);

MSE_TIKHSVD = (norm(A*(x-x_tikhsvd))/norm(x))^2;
MSE_TIKHSVD_DB = 10*log10(MSE_TIKHSVD);

MSE_TIKHLWB = (norm(A*(x-x_tikhlwb))/norm(x))^2;
MSE_TIKHLWB_DB = 10*log10(MSE_TIKHLWB);

RSNR_PINV = (norm(x-x_pinv)/norm(x))^2;
RSNR_PINV_DB = 10*log10(RSNR_PINV);

RSNR_TIKHSVD = (norm(x-x_tikhsvd)/norm(x))^2;
RSNR_TIKHSVD_DB = 10*log10(RSNR_TIKHSVD);

RSNR_TIKHLWB = (norm(x-x_tikhlwb)/norm(x))^2;
RSNR_TIKHLWB_DB = 10*log10(RSNR_TIKHLWB);

%% Plots
figure, subplot(2,3,[1,2])
stem(b,'-g',"LineWidth",4), hold on, grid on
stem(A*x_pinv,'--b',"LineWidth",2)
stem(A*x_tikhsvd,'--r',"LineWidth",2)
stem(A*x_tikhsvd,'--m',"LineWidth",2)
xlabel('$n$','Interpreter','latex')
ylabel('$b$','Interpreter','latex')
title('Signal')

subplot(2,3,3)
stem(x,'-g',"LineWidth",2), grid on
xlabel('$n$','Interpreter','latex')
ylabel('$x$','Interpreter','latex')
title('Original Sparse Code')

subplot(2,3,4)
stem(x_pinv,'-b',"LineWidth",2), grid on
xlabel('$n$','Interpreter','latex')
ylabel('$\tilde{x}^{PINV}$','Interpreter','latex')
title('Reconstructed Sparse Code using MPI')

subplot(2,3,5)
stem(x_tikhsvd,'-r',"LineWidth",2), grid on
xlabel('$n$','Interpreter','latex')
ylabel('$\tilde{x}^{TIKHSVD}$','Interpreter','latex')
title('Reconstructed Sparse Code using Tikhonov')

subplot(2,3,6)
stem(x_tikhlwb,'-m',"LineWidth",2), grid on
xlabel('$n$','Interpreter','latex')
ylabel('$\tilde{x}^{TIKHLWB}$','Interpreter','latex')
title('Reconstructed Sparse Code using Tikhonov MM')