%% Clear workspace
clc
clear
close all

%% Create test case
sigma=0.01;

x_true = imread("392.png");
x_true = rgb2gray(x_true);
x_true = double(x_true);
x_true = x_true./max(x_true(:));

n= size(x_true,1);

opt=PRtomo('defaults');
opt.angles=linspace(0,179,120);
opt.phantomImage = x_true;
[A,b,~]=PRtomo(n,opt);
%x_true=reshape(x_true,n,n);
%rng(1,'v4')
%rng(13,'v5normal')
rng(21,'v5normal')
e=randn(size(b));
e=e/norm(e,'fro')*norm(b,'fro')*sigma;
b=b+e;
noise_norm=norm(e,'fro');

%% Set parameters
TV=TVclass(n,n);
epsilon=1e-1;
q=0.1;
tau=1.01;
iter=500;
tol=1e-4;
Rest=30;
mu=[];

%% ricostruzione l2-lq con TV
xDP=l2lqDPRestarted(A,b,TV,q,epsilon,noise_norm,tau,iter,tol,Rest);
xDP=reshape(xDP,n,n);
RREDP=norm(xDP-x_true,'fro')/norm(x_true,'fro');
PSNRDP=psnr(xDP,x_true);
SSIMDP=ssim(xDP,x_true);
fprintf('L2-Lq TV                 - RRE: %5.4f - PSNR: %4.2f - SSIM: %5.4f \n', ...
    RREDP,PSNRDP,SSIMDP)

%% costruisco Laplaciano con Tikhonov
R=5;
sigmaInt=1e-3;
k=50;
xTik=KTikhonovGenGCV(A,b(:),k,TV);
%xTik=KTikhonovGenDP(A,b(:),k,TV,noise_norm,1);
xTik=reshape(xTik,n,n);
LG=computeL(xTik,sigmaInt,R);
RRETik=norm(xTik-x_true,'fro')/norm(x_true,'fro');
PSNRTik=psnr(xTik,x_true);
SSIMTik=ssim(xTik,x_true);
fprintf('L2-L2 Deriv. 1°          - RRE: %5.4f - PSNR: %4.2f - SSIM: %5.4f \n', ...
    RRETik,PSNRTik,SSIMTik)

%% Costruisce il grafo con L2-Lq TV
% fprintf('Graph Laplacian computed starting from TV \n')
% LG=computeL(xDP,sigmaInt,R);

%% Update the Graph Laplacian with the solution for alpha = 1
options.alpha=1;
options.mu=mu;
options.LG=LG;
options.waitbar=1;
options.noise_norm=noise_norm;
options.rest=20;
options.d=30;
options.epsilon = epsilon;
options.iter = iter;
options.Rest = Rest;
options.tau = tau;
options.tol = tol;
options.q = q;
options.mu = mu;
xA1=l2lqFract(A,b,options);
%LG=computeL(xA1,sigmaInt,R);
RREA1=norm(xA1-x_true,'fro')/norm(x_true,'fro');
PSNRA1=psnr(xA1,x_true);
SSIMA1=ssim(xA1,x_true);
fprintf('L2-Lq alpha = 1 (Tik)    - RRE: %5.4f - PSNR: %4.2f - SSIM: %5.4f \n', ...
    RREA1,PSNRA1,SSIMA1)

%% Costruisce il grafo con alpha = 1
fprintf('Graph Laplacian computed starting from alpha = 1 \n')
LG=computeL(xA1,sigmaInt,R);

%% Fractional Graph Laplacian
% R=5;
% sigmaInt=1e-3;
% epsilon=1e-1;
% q=0.1;
% tau=1.01;
% iter=500;
% tol=1e-4;
% Rest=30;
% mu=[];
% 
% 
% LG=computeL(x_true,sigmaInt,R);
%alpha=[0.2:0.1:2];
alpha=0.4;
XF=zeros([n,n,length(alpha)]);
RREF=zeros(length(alpha),1);
W=zeros(length(alpha),1);
for j=1:length(alpha)
    clear options
    options.alpha=alpha(j);
    options.mu=mu;
    options.LG=LG;
    options.waitbar=1;
    options.rest=20;
    options.d=10;
    options.epsilon = epsilon;
    options.iter = iter;
    options.Rest = Rest;
    options.tau = tau;
    options.tol = tol;
    options.q = q;
    options.mu = mu;
    options.noise_norm=noise_norm;
    [xF,~,regolar]=l2lqFract(A,b,options);
    XF(:,:,j)=xF;
    ResF=A*xF(:)-b(:);
    eta=fft2(reshape(ResF,length(b)/length(opt.angles),length(opt.angles)));
    W(j)=norm(eta.^2,'fro').^2/norm(eta,'fro').^4;
    RREF(j)=norm(xF-x_true,'fro')/norm(x_true,'fro');
    PSNRF(j)=psnr(xF,x_true);
    SSIMF(j)=ssim(xF,x_true);
    %RESNF(j)=norm(ResF);
    fprintf('L2-Lq Fract. Graph Lapl. - RRE: %5.4f - PSNR: %4.2f - SSIM: %5.4f  - alpha = %3.2f - whit. = %8.6f \n', ...
    RREF(j),PSNRF(j),SSIMF(j),alpha(j),W(j))
end
save('RisultatiTomo.mat')

%% Plots
Lw = 3;                % line width
Fw = 22;               % font size
Ms = 12;               % marker size

[wm,wi]=min(W);
figure(1)
h=semilogy(alpha,W,'x-b',alpha(wi),W(wi),'*r');
set(h,'LineWidth',Lw,'MarkerSize',Ms)
set(gca,'FontSize',Fw)
print(gcf, '-depsc2', 'tomo_white')
xlim([0.2 2])

figure(2)
h=semilogy(alpha,RREF,'x-b',alpha(wi),RREF(wi),'*r');
set(h,'LineWidth',Lw,'MarkerSize',Ms)
set(gca,'FontSize',Fw)
print(gcf, '-depsc2', 'tomo_rre')
xlim([0.2 2])

figure(3)
h=semilogy(alpha,PSNRF,'x-b',alpha(wi),PSNRF(wi),'*r');
set(h,'LineWidth',Lw,'MarkerSize',Ms)
set(gca,'FontSize',Fw)
print(gcf, '-depsc2', 'tomo_psnr')
xlim([0.2 2])

figure(4)
h=semilogy(alpha,SSIMF,'x-b',alpha(wi),SSIMF(wi),'*r');
set(h,'LineWidth',Lw,'MarkerSize',Ms)
set(gca,'FontSize',Fw)
print(gcf, '-depsc2', 'tomo_ssim')
xlim([0.2 2])
%% Images
figure(5)
imshow(xTik,[],'border','tight')
figure(6)
imshow(xDP,[],'border','tight')
figure(7)
imshow(xA1,[],'border','tight')
[rm,ri]=min(RREF);
figure(8)
imshow(XF(:,:,ri),[],'border','tight')

return
%% Print results
fprintf('L2-Lq TV                 - RRE: %5.4f - PSNR: %4.2f - SSIM: %5.4f \n', ...
    RREDP,PSNRDP,SSIMDP)
fprintf('L2-L2 Deriv. 1°          - RRE: %5.4f - PSNR: %4.2f - SSIM: %5.4f \n', ...
    RRETik,PSNRTik,SSIMTik)
fprintf('L2-Lq alpha = 1 (Tik)    - RRE: %5.4f - PSNR: %4.2f - SSIM: %5.4f \n', ...
    RREA1,PSNRA1,SSIMA1)
fprintf('L2-Lq Fract. Graph Lapl. - RRE: %5.4f - PSNR: %4.2f - SSIM: %5.4f  - alpha = %3.2f - whit. = %8.6f \n', ...
    RREF(ri),PSNRF(ri),SSIMF(ri),alpha(ri),W(ri))