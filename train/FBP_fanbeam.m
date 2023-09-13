%% Load COULE Data
train_set = fileDatastore('./data/COULE_GT/*.mat','ReadFcn',@load,'FileExtensions','.mat');
test_set = fileDatastore('./data/COULE_test/*.mat','ReadFcn',@load,'FileExtensions','.mat');

train_set = transform(train_set, @(data) rearrange_datastore(data));
test_set = transform(test_set, @(data) rearrange_datastore(data));

%% Visualize sample image (for reference only)
x = read(train_set);
figure; imshow(x);

% Get the shape of x
n = size(x, 1);

%% Define the projector
% Parameters
theta = linspace(0, 179, 180);
p = floor(n * sqrt(2)); % Number of rays

% Define the projector
A = fanlineartomo(n,theta,p);

%% Compute sinogram, add noise and reconstruct with FBP
% Compute sinogram and add noise
y = A * x(:);

noise_level = 0.01;
e = randn(size(y));
e = e / norm(e) * norm(y) * noise_level;
y_delta = y + e;

% Compute FBP
x_FBP = fbp(A, y_delta, theta);

% Compute SIRT
x_SIRT = drop(A, y_delta, 50);
%% Visualize a sample
figure;
subplot(1, 3, 1); imshow(reshape(x, 256, 256));
subplot(1, 3, 2); imshow(reshape(x_FBP, 256, 256));
subplot(1, 3, 3); imshow(reshape(x_SIRT, 256, 256));


