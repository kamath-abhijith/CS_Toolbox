clear
clc
close all

%% Initialise
A = randn(50,10); [m,n] = size(A);
x = 10*sprandn(10,1,0.2);
b = A*x;

%% Solvers
x_pinv = pinv(A)*b;
x_irls = lasso_irls(A,b,0.01);

%% Error metrics
MSE_PINV = (norm(x-x_pinv)/norm(x))^2;
MSE_PINV_DB = 10*log10(MSE_PINV);

MSE_IRLS = (norm(x-x_irls)/norm(x))^2;
MSE_IRLS_DB = 10*log10(MSE_IRLS);

%% Plots
figure, subplot(2,2,1)
stem(b,'-g',"LineWidth",2), hold on, grid on
stem(A*x_pinv,'--b',"LineWidth",2)
stem(A*x_irls,'--r',"LineWidth",2)
axis([0 m -1.2*max(max(b),min(b)) 1.2*max(max(b),min(b))])
xlabel('$n$','Interpreter','latex')
ylabel('$b$','Interpreter','latex')
title('Signal')

subplot(2,2,2)
stem(x,'-g',"LineWidth",2), grid on
axis([0 n -1.2*max(max(x),-min(x)) 1.2*max(max(x),-min(x))])
xlabel('$n$','Interpreter','latex')
ylabel('$x$','Interpreter','latex')
title('Original Sparse Code')

subplot(2,2,3)
stem(x_pinv,'-b',"LineWidth",2), grid on
axis([0 n -1.2*max(max(x),-min(x)) 1.2*max(max(x),-min(x))])
xlabel('$n$','Interpreter','latex')
ylabel('$\tilde{x}^{PINV}$','Interpreter','latex')
title('Reconstructed Sparse Code using MPI')

subplot(2,2,4)
stem(x_irls,'-r',"LineWidth",2), grid on
axis([0 n -1.2*max(max(x),-min(x)) 1.2*max(max(x),-min(x))])
xlabel('$n$','Interpreter','latex')
ylabel('$\tilde{x}^{IRLS}$','Interpreter','latex')
title('Reconstructed Sparse Code using IRLS')