%% Clear
clc
clear
close all

%% Load Mayo Data
train_set = fileDatastore('../../data/Mayo_train/*.mat','ReadFcn',@load,'FileExtensions','.mat');
test_set = fileDatastore('../../data/Mayo_test/*.mat','ReadFcn',@load,'FileExtensions','.mat');

train_set = transform(train_set, @(data) rearrange_datastore(data));
test_set = transform(test_set, @(data) rearrange_datastore(data));

% Resize data
train_set = transform(train_set, @(data) resize_Mayo(data));
test_set = transform(test_set, @(data) resize_Mayo(data));

%% Define problem parameters and get the projector
n = 256; % Number of pixels per dimension (assumed to be squared)
n_d = floor(n * sqrt(2)); % Detector size
n_theta = 180; % Number of projections equispaced in the [0, pi] domain

x_gt = read(test_set);

noise_level = 0.01; % Percentage norm of the noise compared with y

% Generate projector
theta = linspace(0, 179, n_theta); % Projection angles
% A = fanlineartomo(n, theta, n_d);

% Define the projector
options.phantomImage = x_gt;
options.angles = theta;
options.p = n_d;
A = PRtomo(n, options);

%% Training parameters
batch_size = 10;
n_epochs = 50;
learning_rate = 0.001;

N_train = 3224;
iteration_per_epoch = ceil(N_train / batch_size);
total_iterations = n_epochs * iteration_per_epoch;

% MiniBatchQueue
mbq = minibatchqueue(train_set,...
    MiniBatchSize=batch_size);

% Optimizer
averageGrad = [];
averageSqGrad = [];

%% Define the network
net = Unet_model([n, n, 1], 32);

%% Training loop
% Initialize the monitor
monitor = trainingProgressMonitor(Metrics="Loss",Info=["Epoch","Iteration","SSIM","LearnRate","ExecutionEnvironment"],XLabel="Iteration");

executionEnvironment = "auto";
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    updateInfo(monitor,ExecutionEnvironment="GPU");
else
    updateInfo(monitor,ExecutionEnvironment="CPU");
end

epoch = 0;
iteration = 0;

% Loop over epochs.
while epoch < n_epochs && ~monitor.Stop
    
    epoch = epoch + 1;

    % Shuffle data.
    shuffle(mbq);
    
    % Loop over mini-batches.
    while hasdata(mbq) && ~monitor.Stop

        iteration = iteration + 1;
        
        % Read mini-batch of data.
        X_gt = dlarray(preprocess_batch(next(mbq)), "SSCB");
        Y = compute_forward(A, X_gt, n_d, n_theta, noise_level);
        X_fbp = dlarray(compute_FBP(A, Y, theta), "SSCB");
        
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelLoss function and update the network state.
        [loss,gradients,state] = dlfeval(@modelLoss, net, X_fbp, X_gt);
        net.State = state;
        
        % Update the network parameters using the ADAM optimizer.
        [net,averageGrad,averageSqGrad] = adamupdate(net,gradients,averageGrad,averageSqGrad,iteration,learning_rate);
        
        % Compute SSIM
        ssim_val = mean(ssim(predict(net, X_fbp), X_gt));

        % Update the training progress monitor.
        recordMetrics(monitor,iteration,Loss=loss);
        updateInfo(monitor, ...
            Epoch=string(epoch) + " of " + string(n_epochs), ...
            Iteration=string(iteration) + " of " + string(total_iterations), ...
            SSIM=ssim_val, ...
            LearnRate=learning_rate);
        monitor.Progress = 100 * iteration/total_iterations;
    end
end

%% Save the model after training
save("..\..\model_weights\Mayo\unet_mse_" + n_theta + ".mat", "net")

%% Load the model and do predictions on the test set
load("..\..\model_weights\Mayo\unet_mse_" + n_theta + ".mat", "net")

% Get an x_gt from test set
% x_gt = read(test_set);

% Compute the corresponding y
y = A * x_gt(:);

% Add noise
noise_level = 0.02;
e = randn(size(y));
e = e / norm(e) * norm(y) * noise_level;
y_delta = y + e;

% Compute FBP
x_FBP = dlarray(reshape(fbp(A, y_delta, theta), [n, n, 1, 1]), "SSCB");

% Compute prediction
x_NN = predict(net, x_FBP);

% Convert dlarrays to double array
x_FBP = double(gather(extractdata(x_FBP)));
x_NN = double(gather(extractdata(x_NN)));

%% Show results
figure;
subplot(1, 3, 1); imshow(reshape(x_gt, n, n));
subplot(1, 3, 2); imshow(reshape(x_FBP, n, n));
subplot(1, 3, 3); imshow(reshape(x_NN, n, n));

ssim(x_NN, x_gt)

%%  Utility functions
function [loss,gradients,state] = modelLoss(net, X_fbp, X_gt)

% Forward data through network.
[X_pred, state] = forward(net, X_fbp);

% Calculate cross-entropy loss.
loss = mean(sum((X_pred - X_gt).^2, [1, 2, 3]));

% Calculate gradients of loss with respect to learnable parameters.
gradients = dlgradient(loss, net.Learnables);

end

function X = preprocess_batch(dataX)
% Get shape from dataX (n, n, B)
[m, n, B] = size(dataX);

% Output the right-shaped array
X = reshape(dataX, m, n, 1, B);
end

function Y = compute_forward(A, X, n_d, n_theta, noise_level)
% Parameters
batch_size = size(X, 4);

% Gather X
X = double(gather(extractdata(X)));

% Initialize Y
Y = zeros(n_d*n_theta, batch_size);

% Cycle over the batch
for i = 1:batch_size
    % Load the i-th image
    x = X(:, :, 1, i);

    % Compute the projection
    y = A * x(:);
    
    % Add noise
    e = randn(size(y));
    e = e / norm(e) * norm(y) * noise_level;
    y_delta = y + e;

    % Add to Y
    Y(:, i) = y_delta;
end
end

function X = compute_FBP(A, Y, theta)
% Parameters
n = floor(sqrt(size(A, 2)));
batch_size = size(Y, 2);

% Initialize Y
X = zeros(n, n, 1, batch_size);

% Cycle over the batch
for i = 1:batch_size
    % Load the i-th image
    y = Y(:, i);

    % Compute the projection
    x_fbp = fbp(A, y, theta);

    % Add to Y
    X(:, :, 1, i) = reshape(x_fbp, n, n);
end
X = dlarray(X);
end

