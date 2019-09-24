clear
clc
close all

%% Support and size
m = 28; n = 28;

%% Define basis images as vectors
phiv = @(l,m) im2col(dct2_dictlm(l,m),[28 28],'distinct');

%% Construct dictionary
DICT = zeros(m*n,m*n);

for i = 0:m-1
    for j = 0:n-1
        DICT(:,(i+1)*(j+1)) = phiv(i,j);
    end
end

%% Define the image // linear combination of basis images
spden = 0.01;
c = sprandn(m*n,1,spden);

x = DICT*c;
% x = 0.1*phiv(1,0) + 0.3*phiv(5,8) - 0.2*phiv(12,10) + .1*phiv(20,24);
x_image = reshape(x,[m,n]);

%% Plots
figure
imshow(x_image,'InitialMagnification','fit')
colorbar, axis on
xlabel('$x$','Interpreter','latex')
ylabel('$y$','Interpreter','latex')
title('$28\times 28$ Image on DCT Basis','Interpreter','latex')
set(gca,'FontSize',24)

%% Save images
% dctimg.dict = DICT;
% dctimg.image = x_image;
% 
% save dctimg.mat

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