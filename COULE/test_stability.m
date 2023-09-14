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

noise_level_min = 0;
noise_level_max = 0.1;
noise_level_n   = 11;

noise_levels = linspace(noise_level_min, noise_level_max, noise_level_n);
noise_levels(1) = 1e-3;

% Initialize noise (normalized)
e = randn(size(y));
e = e / norm(e) * norm(y);

%% Compute TV solution at convergence (for comparison) and GraphLaTV
TV         = TVclass(n,n);
epsilon    = 1e-1;
q          = 1;
tau        = 1.01;
iter       = 500;
tol        = 1e-4;
Rest       = 30;
mu         = [];

x_TV = zeros(noise_level_n, n, n);
x_LaTV = zeros(noise_level_n, n, n);
fprintf("------ TV SOLUTION --------\n");
for k = 1:noise_level_n
    % Get noise level
    noise_level = noise_levels(k);
    fprintf("k = %i. Running noise_level = %0.3f. \n", k, noise_level);

    e_tmp = e * noise_level;
    y_delta = y + e_tmp;
    noise_norm = norm(e_tmp);

    xx_TV = reshape(l2lqDPRestarted(A, y_delta, TV, q, epsilon, noise_norm, ...
                          tau, iter, tol, Rest), n, n);
    
    % Parameters
    R        = 5;
    sigmaInt = 1e-3;
    
    LG    = computeL(xx_TV, sigmaInt, R);
    
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
    x_LaTV(k, :, :) = l2lqFract(A, y_delta, options);
    x_TV(k, :, :) = xx_TV;
end

%% Compute GraphLaTik solution
R        = 5;
sigmaInt = 1e-3;
n_it        = 50;

x_Tik = zeros(noise_level_n, n, n);
x_LaTik = zeros(noise_level_n, n, n);
fprintf("------ Tik SOLUTION --------\n");
for k = 1:noise_level_n
    % Get noise level
    noise_level = noise_levels(k);
    fprintf("k = %i. Running noise_level = %0.3f. \n", k, noise_level);

    e_tmp = e * noise_level;
    y_delta = y + e_tmp;
    noise_norm = norm(e_tmp);

    xx_Tik = reshape(KTikhonovGenGCV(A, y_delta, n_it, TV), n, n);
    LG    = computeL(xx_Tik, sigmaInt, R);
    
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
    x_LaTik(k, :, :) = l2lqFract(A, y_delta, options);
    x_Tik(k, :, :) = xx_Tik;
end

%% Compute GraphLaNet solution
R        = 5;
sigmaInt = 1e-3;

net   = load("..\model_weights\COULE\unet_mae_"+n_theta+".mat").net;

x_FBP = zeros(noise_level_n, n, n);
x_LaFBP = zeros(noise_level_n, n, n);

x_NN = zeros(noise_level_n, n, n);
x_LaNet = zeros(noise_level_n, n, n);
fprintf("------ NN and FBP SOLUTIONs --------\n");
for k = 1:noise_level_n
    % Get noise level
    noise_level = noise_levels(k);
    fprintf("k = %i. Running noise_level = %0.3f. \n", k, noise_level);

    e_tmp = e * noise_level;
    y_delta = y + e_tmp;
    noise_norm = norm(e_tmp);

    xx_FBP = dlarray(reshape(fbp(A, y_delta, theta), [n, n, 1, 1]), "SSCB");
    
    % Compute prediction
    xx_NN = predict(net, xx_FBP);

    % Take x_NN and x_FBP from GPU
    xx_FBP = double(gather(extractdata(xx_FBP)));
    xx_NN = double(gather(extractdata(xx_NN)));

    % Build Graph Laplacian
    LG   = computeL(xx_NN, sigmaInt, R);

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

    x_LaNet(k, :, :) = l2lqFract(A, y_delta, options);
    x_NN(k, :, :) = xx_NN;

    % Build Graph Laplacian for LaFBP
    LG   = computeL(xx_FBP, sigmaInt, R);

    % GraphLaFBP settings
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

    x_LaFBP(k, :, :) = l2lqFract(A, y_delta, options);
    x_FBP(k, :, :) = xx_FBP;

end

%% Compute GraphLaTrue solution
R        = 5;
sigmaInt = 1e-3;

x_LaTrue = zeros(noise_level_n, n, n);
fprintf("------ LaTrue SOLUTION --------\n");
for k = 1:noise_level_n
    % Get noise level
    noise_level = noise_levels(k);
    fprintf("k = %i. Running noise_level = %0.3f. \n", k, noise_level);

    e_tmp = e * noise_level;
    y_delta = y + e_tmp;
    noise_norm = norm(e_tmp);

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
    
    x_LaTrue(k, :, :) = l2lqFract(A, y_delta, options);
end

%% Plots

% SSIM
SSIM_FBP = zeros(noise_level_n, 1);
SSIM_Tik = zeros(noise_level_n, 1);
SSIM_TV = zeros(noise_level_n, 1);
SSIM_Net = zeros(noise_level_n, 1);
SSIM_LaTrue = zeros(noise_level_n, 1);
SSIM_LaFBP = zeros(noise_level_n, 1);
SSIM_LaTik = zeros(noise_level_n, 1);
SSIM_LaTV = zeros(noise_level_n, 1);
SSIM_LaNet = zeros(noise_level_n, 1);
for k = 1:noise_level_n
    SSIM_FBP(k) = ssim(squeeze(x_FBP(k, :, :)), x_true);
    SSIM_Tik(k) = ssim(squeeze(x_Tik(k, :, :)), x_true);
    SSIM_TV(k) = ssim(squeeze(x_TV(k, :, :)), x_true);
    SSIM_Net(k) = ssim(squeeze(x_NN(k, :, :)), x_true);
    SSIM_LaTrue(k) = ssim(squeeze(x_LaTrue(k, :, :)), x_true);
    SSIM_LaFBP(k) = ssim(squeeze(x_LaFBP(k, :, :)), x_true);
    SSIM_LaTik(k) = ssim(squeeze(x_LaTik(k, :, :)), x_true);
    SSIM_LaTV(k) = ssim(squeeze(x_LaTV(k, :, :)), x_true);
    SSIM_LaNet(k) = ssim(squeeze(x_LaNet(k, :, :)), x_true);
end

figure; hold on;
title('SSIM at increasing noise');
xlabel('noise level');
ylabel('SSIM');
%plot(noise_levels, SSIM_FBP, 'o-');
plot(noise_levels, SSIM_Tik, 'o-');
plot(noise_levels, SSIM_TV, 'o-');
plot(noise_levels, SSIM_Net, 'o-');
plot(noise_levels, SSIM_LaTrue, 'o-');
%plot(noise_levels, SSIM_LaFBP, 'o-');
plot(noise_levels, SSIM_LaTik, 'o-');
plot(noise_levels, SSIM_LaTV, 'o-');
plot(noise_levels, SSIM_LaNet, 'o-');
legend(["Tik", "TV", "Net", "LaTrue", "LaTik", "LaTV", "LaNet"]);
grid("on");
%legend(["FBP", "Tik", "TV", "Net", "LaTrue", "LaFBP", "LaTik", "LaTV", "LaNet"])


% RMSE
RMSE_FBP = zeros(noise_level_n, 1);
RMSE_Tik = zeros(noise_level_n, 1);
RMSE_TV = zeros(noise_level_n, 1);
RMSE_Net = zeros(noise_level_n, 1);
RMSE_LaTrue = zeros(noise_level_n, 1);
RMSE_LaFBP = zeros(noise_level_n, 1);
RMSE_LaTik = zeros(noise_level_n, 1);
RMSE_LaTV = zeros(noise_level_n, 1);
RMSE_LaNet = zeros(noise_level_n, 1);
for k = 1:noise_level_n
    RMSE_FBP(k) = sqrt(mean((x_true - squeeze(x_FBP(k, :, :))).^2, "all"));
    RMSE_Tik(k) = sqrt(mean((x_true - squeeze(x_Tik(k, :, :))).^2, "all"));
    RMSE_TV(k) = sqrt(mean((x_true - squeeze(x_TV(k, :, :))).^2, "all"));
    RMSE_Net(k) = sqrt(mean((x_true - squeeze(x_NN(k, :, :))).^2, "all"));
    RMSE_LaTrue(k) = sqrt(mean((x_true - squeeze(x_LaTrue(k, :, :))).^2, "all"));
    RMSE_LaFBP(k) = sqrt(mean((x_true - squeeze(x_LaFBP(k, :, :))).^2, "all"));
    RMSE_LaTik(k) = sqrt(mean((x_true - squeeze(x_LaTik(k, :, :))).^2, "all"));
    RMSE_LaTV(k) = sqrt(mean((x_true - squeeze(x_LaTV(k, :, :))).^2, "all"));
    RMSE_LaNet(k) = sqrt(mean((x_true - squeeze(x_LaNet(k, :, :))).^2, "all"));
end

figure; hold on;
title('RMSE at increasing noise');
xlabel('noise level');
ylabel('RMSE');
%plot(noise_levels, RMSE_FBP, 'o-');
plot(noise_levels, RMSE_Tik, 'o-');
plot(noise_levels, RMSE_TV, 'o-');
plot(noise_levels, RMSE_Net, 'o-');
plot(noise_levels, RMSE_LaTrue, 'o-');
%plot(noise_levels, RMSE_LaFBP, 'o-');
plot(noise_levels, RMSE_LaTik, 'o-');
plot(noise_levels, RMSE_LaTV, 'o-');
plot(noise_levels, RMSE_LaNet, 'o-');
legend(["Tik", "TV", "Net", "LaTrue", "LaTik", "LaTV", "LaNet"]);
grid("on");
%legend(["FBP", "Tik", "TV", "Net", "LaTrue", "LaFBP", "LaTik", "LaTV", "LaNet"])

% PSNR
PSNR_FBP = zeros(noise_level_n, 1);
PSNR_Tik = zeros(noise_level_n, 1);
PSNR_TV = zeros(noise_level_n, 1);
PSNR_Net = zeros(noise_level_n, 1);
PSNR_LaTrue = zeros(noise_level_n, 1);
PSNR_LaFBP = zeros(noise_level_n, 1);
PSNR_LaTik = zeros(noise_level_n, 1);
PSNR_LaTV = zeros(noise_level_n, 1);
PSNR_LaNet = zeros(noise_level_n, 1);
for k = 1:noise_level_n
    PSNR_FBP(k) = psnr(squeeze(x_FBP(k, :, :)), x_true);
    PSNR_Tik(k) = psnr(squeeze(x_Tik(k, :, :)), x_true);
    PSNR_TV(k) = psnr(squeeze(x_TV(k, :, :)), x_true);
    PSNR_Net(k) = psnr(squeeze(x_NN(k, :, :)), x_true);
    PSNR_LaTrue(k) = psnr(squeeze(x_LaTrue(k, :, :)), x_true);
    PSNR_LaFBP(k) = psnr(squeeze(x_LaFBP(k, :, :)), x_true);
    PSNR_LaTik(k) = psnr(squeeze(x_LaTik(k, :, :)), x_true);
    PSNR_LaTV(k) = psnr(squeeze(x_LaTV(k, :, :)), x_true);
    PSNR_LaNet(k) = psnr(squeeze(x_LaNet(k, :, :)), x_true);
end

figure; hold on;
title('PSNR at increasing noise');
xlabel('noise level');
ylabel('PSNR');
%plot(noise_levels, PSNR_FBP, 'o-');
plot(noise_levels, PSNR_Tik, 'o-');
plot(noise_levels, PSNR_TV, 'o-');
plot(noise_levels, PSNR_Net, 'o-');
plot(noise_levels, PSNR_LaTrue, 'o-');
%plot(noise_levels, PSNR_LaFBP, 'o-');
plot(noise_levels, PSNR_LaTik, 'o-');
plot(noise_levels, PSNR_LaTV, 'o-');
plot(noise_levels, PSNR_LaNet, 'o-');
legend(["Tik", "TV", "Net", "LaTrue", "LaTik", "LaTV", "LaNet"]);
grid("on");
%legend(["FBP", "Tik", "TV", "Net", "LaTrue", "LaFBP", "LaTik", "LaTV", "LaNet"])