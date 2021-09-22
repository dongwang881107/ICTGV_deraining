%% ICTGV deraining demo
% 

clear;
close all;

%% Load data
%% Deraining on the Y channel of the image
img_path = '../data/2.jpg';
img = im2double(imread(img_path));
img_yuv = rgb2ycbcr(img);
img_y = img_yuv(:,:,1);

%% Set up parameters
tgv_reg_param = 0.01;
tgv_model_param = 1.0;
osci_reg_fac = 0.9;
osci_model_param = 1.0;
osci_l1_param = 0.1;

alpha = 0.008;
beta = 2*alpha;
num_d = 3; % Number of directions

alpha_t = zeros(num_d,1);
beta_t = zeros(num_d,1);
gamma_t = zeros(num_d,1);
alpha_t(1) = tgv_reg_param;
beta_t(1) = tgv_model_param*alpha;
gamma_t(1) = 0;
for i=1:num_d
    alpha_t(i) = osci_reg_fac*alpha_t(1);
    beta_t(i) = osci_model_param*alpha_t(i);
    gamma_t(i) = osci_l1_param*alpha_t(i);
end

maxits = 50;
tau = 1/sqrt(12);
sigma = 1/tau/12;
tau1 = 1/sqrt(30);
sigma1 = 1/tau1/30;

params.tgv_reg_param = tgv_reg_param;
params.tgv_model_param = tgv_model_param;
params.osci_reg_fac = osci_reg_fac;
params.osci_model_param = osci_model_param;
params.osci_l1_param = osci_l1_param;
params.alpha = alpha;
params.beta = beta;
params.num_d = num_d;
params.alpha_t = alpha_t;
params.beta_t = beta_t;
params.gamma_t = gamma_t;
params.maxits = maxits;
params.tau = tau;
params.sigma = sigma;
params.tau1 = tau1;
params.sigma1 = sigma1;

%% Run the main function
tic;
img_y_derained = ictgv_deraining(img_y, params);
t = toc;
fprintf('Running time is %.2f s\n', t);

%% Plot and save derained image
img_yuv(:,:,1) = img_y_derained;
img_derained = ycbcr2rgb(img_yuv);

figure;
subplot(121);imshow(img,[]);title('Original image');
subplot(122);imshow(img_derained,[]);title('Derained image');


