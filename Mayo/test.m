%% Clear workspace
clc
clear
close all

%% Load Mayo Data
train_set = fileDatastore('../data/Mayo_train/*.mat','ReadFcn',@load,'FileExtensions','.mat');
test_set = fileDatastore('../data/Mayo_test/*.mat','ReadFcn',@load,'FileExtensions','.mat');

train_set = transform(train_set, @(data) rearrange_datastore(data));
test_set = transform(test_set, @(data) rearrange_datastore(data));

% Resize data
train_set = transform(train_set, @(data) resize_Mayo(data));
test_set = transform(test_set, @(data) resize_Mayo(data));

%% Visualize sample image (for reference only)
x_true = read(test_set);
figure; imshow(x_true);

% Get the shape of x
n = size(x_true, 1);

%% Define the projector
% Parameters
n_theta = 180;
n_d = floor(n * sqrt(2)); % Number of rays
theta = linspace(0, 179, n_theta);

% Define the projector
options.phantomImage = x_true;
options.angles = theta;
options.p = n_d;
A = PRtomo(n, options);

% A = fanlineartomo(n, theta, n_d);

%% Compute sinogram and add noise
% Compute sinogram and add noise
y = A * x_true(:);

noise_level = 0.02;
e = randn(size(y));
e = e / norm(e) * norm(y) * noise_level;
y_delta = y + e;

%% Compute TV solution at convergence
TV         = TVclass(n,n);
epsilon    = 1e-1;
q          = 1;
tau        = 1.01;
iter       = 500;
tol        = 1e-4;
Rest       = 30;
mu         = [];
noise_norm = norm(e);

x_TV = l2lqDPRestarted(A, y_delta, TV, q, epsilon, noise_norm, ...
                      tau, iter, tol, Rest);

%% Visualize TV
figure; imshow(reshape(x_TV, n, n));
fprintf('SSIM: %0.3f\n', ssim(reshape(x_TV, n, n), x_true));

%% Compute TqV solution for the initial approximate of GraphLaplacian
R        = 5;
sigmaInt = 1e-3;
k        = 50;

x_tilde_TV = reshape(KTikhonovGenGCV(A, y_delta, k, TV), n, n);
LG         = computeL(x_tilde_TV, sigmaInt, R);

fprintf('SSIM: %0.3f\n', ssim(x_tilde_TV, x_true));

%% Compute GraphLaplacian solution
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

x_LaTV = l2lqFract(A, y_delta, options);

fprintf('SSIM: %0.3f\n', ssim(x_LaTV, x_true));

%% Compute NN solution for the initial approximate of GraphLaplacian
R        = 5;
sigmaInt = 1e-3;
k        = 50;

net   = load(".\model_weights\Mayo\unet_mae_"+n_theta+".mat").net;
x_FBP = dlarray(reshape(fbp(A, y_delta, theta), [n, n, 1, 1]), "SSCB");

% Compute prediction
x_NN = predict(net, x_FBP);

% Take x_NN and x_FBP from GPU
x_FBP = double(gather(extractdata(x_FBP)));
x_NN = double(gather(extractdata(x_NN)));

% Build Graph Laplacian
LG   = computeL(x_NN, sigmaInt, R);

fprintf('SSIM: %0.3f\n', ssim(x_NN, x_true));

%% Compute GraphLaNet solution
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
options.mu         = 7e5;

x_LaNet = l2lqFract(A, y_delta, options);

fprintf('SSIM: %0.4f\n', ssim(x_LaNet, x_true));
imshow(x_LaNet);

%% Compute LaTrue and LaFBP solutions
% x_LaTrue
LG   = computeL(x_true, sigmaInt, R);
options.LG = LG;
x_LaTrue = l2lqFract(A, y_delta, options);

% x_LaFBP
LG   = computeL(x_FBP, sigmaInt, R);
options.LG = LG;
x_LaFBP = l2lqFract(A, y_delta, options);

%% Visualize solution and compute metrics
% Reshape
x_TV = reshape(x_TV, n, n);
x_LaTV = reshape(x_LaTV, n, n);

x_NN = reshape(x_NN, n, n);
x_LaNet = reshape(x_LaNet, n, n);

figure;
subplot(3, 3, 2); imshow(x_LaTrue); %title("LaTrue");
subplot(3, 3, 4); imshow(x_FBP); %title("FBP");
subplot(3, 3, 5); imshow(x_TV); %title("TV");
subplot(3, 3, 6); imshow(x_NN); %title("NN");
subplot(3, 3, 7); imshow(x_LaFBP); %title("LaFBP");
subplot(3, 3, 8); imshow(x_LaTV); %title("LaTik");
subplot(3, 3, 9); imshow(x_LaNet); %title("LaNet");

% Metrics
RMSE_FBP = sqrt(mean((x_true - x_FBP).^2, "all"));
RMSE_LaFBP = sqrt(mean((x_true - x_LaFBP).^2, "all"));
RMSE_TV = sqrt(mean((x_true - x_TV).^2, "all"));
RMSE_NN = sqrt(mean((x_true - x_NN).^2, "all"));
RMSE_LaTV = sqrt(mean((x_true - x_LaTV).^2, "all"));
RMSE_LaNet = sqrt(mean((x_true - x_LaNet).^2, "all"));
RMSE_LaTrue = sqrt(mean((x_true - x_LaTrue).^2, "all"));

SSIM_FBP = ssim(x_LaFBP, x_true);
SSIM_LaFBP = ssim(x_LaFBP, x_true);
SSIM_TV = ssim(x_TV, x_true);
SSIM_NN = ssim(x_NN, x_true);
SSIM_LaTV = ssim(x_LaTV, x_true);
SSIM_LaNet = ssim(x_LaNet, x_true);
SSIM_LaTrue = ssim(x_LaTrue, x_true);

PSNR_FBP = psnr(x_LaFBP, x_true);
PSNR_LaFBP = psnr(x_LaFBP, x_true);
PSNR_TV = psnr(x_TV, x_true);
PSNR_NN = psnr(x_NN, x_true);
PSNR_LaTV = psnr(x_LaTV, x_true);
PSNR_LaNet = psnr(x_LaNet, x_true);
PSNR_LaTrue = psnr(x_LaTrue, x_true);

% Print in a table
fprintf('-------------------------------\n');
fprintf('Sol.     RMSE       SSIM     PSNR\n');
fprintf('\n');
fprintf('x_FBP:   %0.4f     %0.4f     %0.4f          \n', RMSE_FBP, SSIM_FBP, PSNR_FBP);
fprintf('x_TV:    %0.4f     %0.4f     %0.4f          \n', RMSE_TV, SSIM_TV, PSNR_TV);
fprintf('x_NN:    %0.4f     %0.4f     %0.4f          \n', RMSE_NN, SSIM_NN, PSNR_NN);
fprintf('x_LaFBP: %0.4f     %0.4f     %0.4f          \n', RMSE_LaFBP, SSIM_LaFBP, PSNR_LaFBP);
fprintf('x_LaTik: %0.4f     %0.4f     %0.4f          \n', RMSE_LaTV, SSIM_LaTV, PSNR_LaTV);
fprintf('x_LaNet: %0.4f     %0.4f     %0.4f          \n', RMSE_LaNet, SSIM_LaNet, PSNR_LaNet);
fprintf('x_LaTrue:%0.4f     %0.4f     %0.4f          \n', RMSE_LaTrue, SSIM_LaTrue, PSNR_LaTrue);
fprintf('-------------------------------\n');


