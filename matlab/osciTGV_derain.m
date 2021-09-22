%
clc; clear;
close all;
I = imread('image/2.jpg');
YUV = rgb2ycbcr(I);
Y = YUV(:,:,1);
u_orig = double(Y)/255;
%f = imread('Synthetic Data/fretinex.png');
%f= double(f)/255;
%figure(4);imshow(f)
%f = u_orig + R;figure(5);imshow(f)
f = u_orig ;figure(4);imshow(f)


tgv_reg_param = 0.01;
tgv_model_param = 1.0;
osci_reg_fac = 0.9;
osci_model_param = 1.0;
osci_l1_param = 0.1;


% regularization parameter
alpha = 0.008;
beta = 2*alpha;

num = 3;                          % number of texture parts
alpha_t = zeros(num,1);
beta_t = zeros(num,1);
gamma_t = zeros(num,1);
alpha_t(1) = tgv_reg_param;
beta_t(1) = tgv_model_param*alpha(1);
gamma_t(1) = 0;
for i=1:num
    alpha_t(i) = osci_reg_fac*alpha_t(1);
    beta_t(i) = osci_model_param*alpha_t(i);
    gamma_t(i) = osci_l1_param*alpha_t(i);
end

maxits =50;
[M, N] = size(f);

u = f;
w = zeros(M,N,2);
n = zeros(M,N,num);
w_t = zeros(M,N,num,2);

p = zeros(M,N,2);
q = zeros(M,N,3);
p_t = zeros(M,N,num,2);
q_t = zeros(M,N,num,3);
x = zeros(M,N);

L2 =12;L1 = 30;

tau = 1/sqrt(12);
sigma = 1/tau/L2;


tau1 = 1/sqrt(L1);
sigma1 = 1/tau1/L1;

omega = zeros(num,2);
c = zeros(num,3);

for k=1:num
    omega(k,1) = sin(pi/36*(k+15));
    omega(k,2) = cos(pi/36*(k+15));
    c(k,1) = 2-2*cos(omega(k,1));
    c(k,2) = 2-2*cos(omega(k,2));
    c(k,3) = 1-cos(omega(k,1))-cos(omega(k,2))+cos(omega(k,1)-omega(k,2));
end


c(:,3) = 2*c(:,3);
c = reshape(c, [1,1,num,3]);
alpha_t = reshape(alpha_t, [1,1,num]);
beta_t = reshape(beta_t, [1,1,num]);
gamma_t = reshape(gamma_t, [1,1,num]);

% backwards difference w.r.t. x
dxm = @(u) [u(:,1:end-1) zeros(M,1)] - [zeros(M,1) u(:,1:end-1)];

% forward difference w.r.t. x
dxp = @(u) [u(:,2:end) u(:,end)] - u;

% backwards difference w.r.t. y
dym = @(u) [u(1:end-1,:);zeros(1,N)] - [zeros(1,N);u(1:end-1,:)];

% forwards difference w.r.t. y
dyp = @(u) [u(2:end,:); u(end,:)] - u;

% backwards difference w.r.t. x
dxm_t = @(u) [u(:,1:end-1,:) zeros(M,1,num)] - [zeros(M,1,num') u(:,1:end-1,:)];

% forward difference w.r.t. x
dxp_t = @(u) [u(:,2:end,:) u(:,end,:)] - u;

% backwards difference w.r.t. y
dym_t = @(u) [u(1:end-1,:,:); zeros(1,N,num)] - [zeros(1,N,num); u(1:end-1,:,:)];

% forwards difference w.r.t. y
dyp_t = @(u) [u(2:end,:,:); u(end,:,:)] - u;

imagewrite_k = 0;
tic;
for k=1:maxits
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % primal step
    
    % remember old value
    u_old = u;
    w_old = w;
    n_old = n;
    w_t_old = w_t;
    n_sum = sum(n,3);
    
    % gradient descend
    div_p = dxm(p(:,:,1)) + dym(p(:,:,2));
    u = u + tau*div_p;
    u = (u + tau*(f - n_sum) )./(1.0+tau);
    div_p_t = dxm_t(p_t(:,:,:,1)) + dym_t(p_t(:,:,:,2));
    cq_t = sum(bsxfun(@times, c, q_t), 4);
    n = n - tau1*(bsxfun(@plus, x, cq_t - div_p_t));
    n = n - bsxfun(@max, -tau1*gamma_t, bsxfun(@min, tau1*gamma_t, n));
    
    div_q1 = dxm(q(:,:,1)) + dym(q(:,:,3));
    div_q2 = dxm(q(:,:,3)) + dym(q(:,:,2));
    w(:,:,1) = w(:,:,1) + tau*(p(:,:,1)+div_q1);
    w(:,:,2) = w(:,:,2) + tau*(p(:,:,2)+div_q2);
    div_q_t1 = dxp_t(q_t(:,:,:,1)) + dyp_t(q_t(:,:,:,3));
    div_q_t2 = dxp_t(q_t(:,:,:,3)) + dyp_t(q_t(:,:,:,2));
    w_t(:,:,:,1) = w_t(:,:,:,1) + tau1*(p_t(:,:,:,1) + div_q_t1);
    w_t(:,:,:,2) = w_t(:,:,:,2) + tau1*(p_t(:,:,:,2) + div_q_t2);
    
    
   % div_h = dym(h);
    %n = n + tau*div_h;
    %n = (n + tau*(f - u) )./(1.0+tau);
    
    % over-relaxation
    u_bar = 2*u-u_old;
    w_bar = 2*w-w_old;
    n_bar = 2*n-n_old;
    w_t_bar = 2*w_t-w_t_old;
    % dual step
    
    % gradient ascend
    a1 = dxp(u_bar) - w_bar(:,:,1);
    a2 = dyp(u_bar) - w_bar(:,:,2);
    p(:,:,1) = p(:,:,1) + sigma*(a1);
    p(:,:,2) = p(:,:,2) + sigma*(a2);
    reproject = max(1.0, sqrt(p(:,:,1).^2 + p(:,:,2).^2)/alpha);
    p(:,:,1) = p(:,:,1)./reproject;
    p(:,:,2) = p(:,:,2)./reproject;
    a_t1 = dxp_t(n_bar) - w_t_bar(:,:,:,1);
    a_t2 = dyp_t(n_bar) - w_t_bar(:,:,:,2);
    p_t(:,:,:,1) = p_t(:,:,:,1) + sigma1*a_t1;
    p_t(:,:,:,2) = p_t(:,:,:,2) + sigma1*a_t2;
    reproject = max(1.0, bsxfun(@ldivide, alpha_t, hypot(p_t(:,:,:,1), p_t(:,:,:,2))));
    p_t = bsxfun(@rdivide, p_t, reproject);
    
    b1 = dxp(w_bar(:,:,1));
    b2 = dyp(w_bar(:,:,2));
    b3 = (dyp(w_bar(:,:,1)) + dxp(w_bar(:,:,2)))/2;
    q(:,:,1) = q(:,:,1) + sigma*(b1);
    q(:,:,2) = q(:,:,2) + sigma*(b2);
    q(:,:,3) = q(:,:,3) + sigma*(b3);
    reproject = max(1.0, sqrt(q(:,:,1).^2 + q(:,:,2).^2 + 2*q(:,:,3).^2)/beta);
    q(:,:,1) = q(:,:,1)./reproject;
    q(:,:,2) = q(:,:,2)./reproject;
    q(:,:,3) = q(:,:,3)./reproject;
    b_t1 = dxm_t(w_t_bar(:,:,:,1))+ bsxfun(@times, c(:,:,:,1), n_bar);
    b_t2 = dym_t(w_t_bar(:,:,:,2))+ bsxfun(@times, c(:,:,:,2), n_bar);
    b_t3 = 0.5*(dym_t(w_t_bar(:,:,:,1)) + dxm_t(w_t_bar(:,:,:,2)) ...
        + bsxfun(@times, c(:,:,:,3), n_bar));
    q_t(:,:,:,1) = q_t(:,:,:,1) + sigma1*b_t1;
    q_t(:,:,:,2) = q_t(:,:,:,2) + sigma1*b_t2;
    q_t(:,:,:,3) = q_t(:,:,:,3) + sigma1*b_t3;
    reproject = max(1.0, bsxfun(@ldivide, beta_t, ...
        sqrt(q_t(:,:,:,1).^2 + q_t(:,:,:,2).^2 + 2*q_t(:,:,:,3).^2)));
    q_t = bsxfun(@rdivide, q_t, reproject);
    %c1 = dyp(n_bar);
   % h = h + sigma*(c1);
    %reproject = max(1.0, abs(h)/gamma);
    %h = h./reproject;
    n_bar_sum = sum(n_bar,3);
    x = (x + sigma1*(n_bar_sum+u_bar - f))./(1+sigma1);
   % x = (x + sigma1*(u_bar - f))./(1+sigma1);
    
    
    if (mod(k,5) == 0)
        for i=1:num
           sfigure(i);imagesc(n(:,:,i)); colormap(gray(256)); colorbar;
        end
        sfigure(101);
        imagesc(sum(n,3)); colormap(gray(256)); colorbar;
        drawnow;
    end
   
end
toc;

YUV(:,:,1) = uint8(255*rescale(u));J=ycbcr2rgb(YUV);
sfigure(200); imagesc([J]); colormap(gray(256)); title(['rain removal']);
YUV(:,:,1) = im2uint8(f);
Jr=ycbcr2rgb(YUV);
figure(7),imshow([Jr,J]);
  psnr1=psnr(I,J); psnr2=psnr(I,Jr) ;
  SSIM1=ssim(I,J);  SSIM2=ssim(I,Jr);
  %imwrite(J,'result\t2.png');
  %YUV(:,:,1) = uint8(255*rescale(u));
  %J=ycbcr2rgb(YUV);
   %figure(5),imshow(J);