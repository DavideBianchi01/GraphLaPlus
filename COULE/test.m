%% Clear workspace
clc
clear
close all

%% Load COULE Data
train_set = fileDatastore('../data/COULE_train/*.mat','ReadFcn',@load,'FileExtensions','.mat');
test_set = fileDatastore('../data/COULE_test/*.mat','ReadFcn',@load,'FileExtensions','.mat');

train_set = transform(train_set, @(data) rearrange_datastore(data));
test_set = transform(test_set, @(data) rearrange_datastore(data));

%% Visualize sample image (for reference only)
x_true = read(train_set);
figure; imshow(x_true);

% Get the shape of x
n = size(x_true, 1);

%% Define the projector
% Parameters
n_theta = 120;
n_d = floor(n * sqrt(2)); % Number of rays
theta = linspace(0, 179, n_theta);

% Define the projector
options.phantomImage = x_true;
options.angles = theta;
options.p = n_d;
A = PRtomo(n, options);

%% Compute sinogram and add noise
% Compute sinogram and add noise
y = A * x_true(:);

noise_level = 0.02;
e = randn(size(y));
e = e / norm(e) * norm(y) * noise_level;
y_delta = y + e;

%% Compute TV solution at convergence (for comparison) and GraphLaTV
TV         = TVclass(n,n);
epsilon    = 1e-1;
q          = 1;
tau        = 1.01;
iter       = 500;
tol        = 1e-4;
Rest       = 30;
mu         = [];
noise_norm = norm(e);

x_TV = reshape(l2lqDPRestarted(A, y_delta, TV, q, epsilon, noise_norm, ...
                      tau, iter, tol, Rest), n, n);

% Parameters
R        = 5;
sigmaInt = 1e-3;

LG    = computeL(x_TV, sigmaInt, R);

% Graph Laplacian Settings
options.alpha      = 1;
options.mu         = mu;
options.LG         = LG;
options.waitbar    = 1;
options.noise_norm = noise_norm;
options.rest       = 20;
options.d          = 30;
options.epsilon    = epsilon;
options.iter       = iter;
options.Rest       = Rest;
options.tau        = tau;
options.tol        = tol;
options.q          = q;
options.mu         = mu;

% Compute GraphLaplacian solution
x_LaTV = l2lqFract(A, y_delta, options);

%% Compute GraphLaTik solution
R        = 5;
sigmaInt = 1e-3;
k        = 50;

x_Tik = reshape(KTikhonovGenGCV(A, y_delta, k, TV), n, n);
LG    = computeL(x_Tik, sigmaInt, R);

% Graph Laplacian Settings
options.alpha      = 1;
options.mu         = mu;
options.LG         = LG;
options.waitbar    = 1;
options.noise_norm = noise_norm;
options.rest       = 20;
options.d          = 30;
options.epsilon    = epsilon;
options.iter       = iter;
options.Rest       = Rest;
options.tau        = tau;
options.tol        = tol;
options.q          = q;
options.mu         = mu;

% Compute GraphLaplacian solution
x_LaTik = l2lqFract(A, y_delta, options);

%% Compute GraphLaNet solution
R        = 5;
sigmaInt = 1e-3;

net   = load("..\model_weights\COULE\unet_mae_"+n_theta+".mat").net;
x_FBP = dlarray(reshape(fbp(A, y_delta, theta), [n, n, 1, 1]), "SSCB");

% Compute prediction
x_NN = predict(net, x_FBP);

% Take x_NN and x_FBP from GPU
x_FBP = double(gather(extractdata(x_FBP)));
x_NN = double(gather(extractdata(x_NN)));

% Build Graph Laplacian
LG   = computeL(x_NN, sigmaInt, R);

% GraphLaNet settings
options.alpha      = 1;
options.mu         = mu;
options.LG         = LG;
options.waitbar    = 1;
options.noise_norm = noise_norm;
options.rest       = 20;
options.d          = 30;
options.epsilon    = epsilon;
options.iter       = iter;
options.Rest       = Rest;
options.tau        = tau;
options.tol        = tol;
options.q          = q;
options.mu         = mu;

x_LaNet = l2lqFract(A, y_delta, options);

%% Compute GraphLaFBP solution
R        = 5;
sigmaInt = 1e-3;

% Build Graph Laplacian
LG   = computeL(x_FBP, sigmaInt, R);

% GraphLaNet settings
options.alpha      = 1;
options.mu         = mu;
options.LG         = LG;
options.waitbar    = 1;
options.noise_norm = noise_norm;
options.rest       = 20;
options.d          = 30;
options.epsilon    = epsilon;
options.iter       = iter;
options.Rest       = Rest;
options.tau        = tau;
options.tol        = tol;
options.q          = q;
options.mu         = mu;

x_LaFBP = l2lqFract(A, y_delta, options);

%% Compute GraphLaTrue solution
R        = 5;
sigmaInt = 1e-3;
k        = 50;

% Build Graph Laplacian
LG   = computeL(x_true, sigmaInt, R);

% GraphLaNet settings
options.alpha      = 1;
options.mu         = mu;
options.LG         = LG;
options.waitbar    = 1;
options.noise_norm = noise_norm;
options.rest       = 20;
options.d          = 30;
options.epsilon    = epsilon;
options.iter       = iter;
options.Rest       = Rest;
options.tau        = tau;
options.tol        = tol;
options.q          = q;
options.mu         = mu;

x_LaTrue = l2lqFract(A, y_delta, options);

%% Visualize solution and compute metrics
% Visualization
figure;
subplot(2, 5, 1); imshow(x_true); title("True");
subplot(2, 5, 2); imshow(x_FBP); title("FBP");
subplot(2, 5, 3); imshow(x_Tik); title("Tik");
subplot(2, 5, 4); imshow(x_TV); title("TV");
subplot(2, 5, 5); imshow(x_NN); title("Net");
subplot(2, 5, 6); imshow(x_LaTrue); title("LaTrue");
subplot(2, 5, 7); imshow(x_LaFBP); title("LaFBP");
subplot(2, 5, 8); imshow(x_LaTik); title("LaTik");
subplot(2, 5, 9); imshow(x_LaTV); title("LaTV");
subplot(2, 5, 10); imshow(x_LaNet); title("LaNet");

% Metrics
RMSE_FBP = sqrt(mean((x_true - x_FBP).^2, "all"));
RMSE_Tik = sqrt(mean((x_true - x_Tik).^2, "all"));
RMSE_TV = sqrt(mean((x_true - x_TV).^2, "all"));
RMSE_NN = sqrt(mean((x_true - x_NN).^2, "all"));
RMSE_LaTrue = sqrt(mean((x_true - x_LaTrue).^2, "all"));
RMSE_LaFBP = sqrt(mean((x_true - x_LaFBP).^2, "all"));
RMSE_LaTik = sqrt(mean((x_true - x_LaTik).^2, "all"));
RMSE_LaTV = sqrt(mean((x_true - x_LaTV).^2, "all"));
RMSE_LaNet = sqrt(mean((x_true - x_LaNet).^2, "all"));

SSIM_FBP = ssim(x_FBP, x_true);
SSIM_Tik = ssim(x_Tik, x_true);
SSIM_TV = ssim(x_TV, x_true);
SSIM_NN = ssim(x_NN, x_true);
SSIM_LaTrue = ssim(x_LaTrue, x_true);
SSIM_LaFBP = ssim(x_LaFBP, x_true);
SSIM_LaTik = ssim(x_LaTik, x_true);
SSIM_LaTV = ssim(x_LaTV, x_true);
SSIM_LaNet = ssim(x_LaNet, x_true);

PSNR_FBP = psnr(x_FBP, x_true);
PSNR_Tik = psnr(x_Tik, x_true);
PSNR_TV = psnr(x_TV, x_true);
PSNR_NN = psnr(x_NN, x_true);
PSNR_LaTrue = psnr(x_LaTrue, x_true);
PSNR_LaFBP = psnr(x_LaFBP, x_true);
PSNR_LaTik = psnr(x_LaTik, x_true);
PSNR_LaTV = psnr(x_LaTV, x_true);
PSNR_LaNet = psnr(x_LaNet, x_true);

% Print in a table
fprintf('-------------------------------\n');
fprintf('Sol.     RMSE       SSIM     PSNR\n');
fprintf('\n');
fprintf('x_FBP:   %0.4f     %0.4f     %0.4f          \n', RMSE_FBP, SSIM_FBP, PSNR_FBP);
fprintf('x_Tik:   %0.4f     %0.4f     %0.4f          \n', RMSE_Tik, SSIM_Tik, PSNR_Tik);
fprintf('x_TV:    %0.4f     %0.4f     %0.4f          \n', RMSE_TV, SSIM_TV, PSNR_TV);
fprintf('x_Net:   %0.4f     %0.4f     %0.4f          \n', RMSE_NN, SSIM_NN, PSNR_NN);
fprintf('-------------------------------\n');
fprintf('x_LaTrue:%0.4f     %0.4f     %0.4f          \n', RMSE_LaTrue, SSIM_LaTrue, PSNR_LaTrue);
fprintf('x_LaFBP: %0.4f     %0.4f     %0.4f          \n', RMSE_LaFBP, SSIM_LaFBP, PSNR_LaFBP);
fprintf('x_LaTik: %0.4f     %0.4f     %0.4f          \n', RMSE_LaTik, SSIM_LaTik, PSNR_LaTik);
fprintf('x_LaTV:  %0.4f     %0.4f     %0.4f          \n', RMSE_LaTV, SSIM_LaTV, PSNR_LaTV);
fprintf('x_LaNet: %0.4f     %0.4f     %0.4f          \n', RMSE_LaNet, SSIM_LaNet, PSNR_LaNet);
fprintf('-------------------------------\n');






