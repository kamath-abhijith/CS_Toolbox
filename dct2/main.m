clear
clc
close all

%% Load image and dictionary
load dctimg.mat

x_image = dctimg.image;
DICT = dctimg.dict;

%% Preprocessing
[m,n] = size(x_image);
x = x_image(:);

%% Sparse recovery
% Pseudo inverse
c_pinv = pinv(DICT)*x;

% ADMM
max_iter = 5000;
abstol = 1e-12; reltol = 1e-10;
c_admm = lasso_admm(DICT,x,1,1.5,max_iter,abstol,reltol);

% ISTA
max_iter = 5000;
abstol = 1e-12;
c_ista = lasso_ista(DICT,x,0.05,max_iter,abstol);

% FISTA
max_iter = 5000;
abstol = 1e-12;
c_fista = lasso_fista(DICT,x,0.05,max_iter,abstol);

%% Reconstruction
x_pinv = DICT*c_pinv(:);
x_pinv_im = reshape(x_pinv,[m,n]);

x_admm = DICT*c_admm(:);
x_admm_im = reshape(x_admm,[m,n]);

x_ista = DICT*c_ista(:);
x_ista_im = reshape(x_ista,[m,n]);

x_fista = DICT*c_fista(:);
x_fista_im = reshape(x_fista,[m,n]);

%% Error Metrics
MSE_PINV = (norm(x-x_pinv)/norm(x))^2;
MSE_PINV_DB = 10*log10(MSE_PINV);

MSE_ADMM = (norm(x-x_admm)/norm(x))^2;
MSE_ADMM_DB = 10*log10(MSE_ADMM);

MSE_ISTA = (norm(x-x_ista)/norm(x))^2;
MSE_ISTA_DB = 10*log10(MSE_ISTA);

MSE_FISTA = (norm(x-x_fista)/norm(x))^2;
MSE_FISTA_DB = 10*log10(MSE_FISTA);

%% Plots
figure, subplot(2,4,1)
imshow(x_image,'InitialMagnification','fit')
colorbar, axis on
xlabel('$x$','Interpreter','latex')
ylabel('$y$','Interpreter','latex')
title('Original Image','Interpreter','latex')
set(gca,'FontSize',24)

subplot(2,4,5)
imshow(x_pinv_im,'InitialMagnification','fit');
colorbar, axis on
xlabel('$x$','Interpreter','latex')
ylabel('$y$','Interpreter','latex')
title('Reconstructed Image using MPI','Interpreter','latex')
set(gca,'FontSize',24)

subplot(2,4,6)
imshow(x_admm_im,'InitialMagnification','fit');
colorbar, axis on
xlabel('$x$','Interpreter','latex')
ylabel('$y$','Interpreter','latex')
title('Reconstructed Image using ADMM','Interpreter','latex')
set(gca,'FontSize',24)

subplot(2,4,7)
imshow(x_ista_im,'InitialMagnification','fit');
colorbar, axis on
xlabel('$x$','Interpreter','latex')
ylabel('$y$','Interpreter','latex')
title('Reconstructed Image using ISTA','Interpreter','latex')
set(gca,'FontSize',24)

subplot(2,4,8)
imshow(x_fista_im,'InitialMagnification','fit');
colorbar, axis on
xlabel('$x$','Interpreter','latex')
ylabel('$y$','Interpreter','latex')
title('Reconstructed Image using FISTA','Interpreter','latex')
set(gca,'FontSize',24)

subplot(2,4,[2,3,4])
stem(c_pinv,'-b',"LineWidth",2)
hold on, grid on
stem(c_admm,'-r',"LineWidth",2)
stem(c_ista,'-m',"LineWidth",2)
stem(c_fista,'-g',"LineWidth",2)
axis([0 m*n -0.5 0.5])
title('Sparse Code','Interpreter','latex')
xlabel('$n$','Interpreter','latex')
ylabel('$c_n$','Interpreter','latex')
set(gca,'color','k','GridColor','w')
set(gca,'FontSize',24)

%% Functions
function DICT_ELE = dct2_dictlm(l,m)
% Matrix DCT-2 basis with unit square support
% 
% INPUT:  Frequencies (l,m)
%         
% OUTPUT: (l,m)th basis matrix

    % Support
    [x,y] = meshgrid(0:0.036:1);
    Nx = length(x); Ny = length(y);
    
    % Weight
    if l == 0
        lambdal = 1/sqrt(2);
    else
        lambdal = 1;
    end
    
    if m == 0
        lambdam = 1/sqrt(2);
    else
        lambdam = 1;
    end
    
    % Build
    DICT_ELE = 2*lambdal*lambdam*cos(pi*l*x/Nx).*cos(pi*m*y/Ny);
end