%% Load COULE Data
train_set = fileDatastore('./data/COULE_GT/*.mat','ReadFcn',@load,'FileExtensions','.mat');
test_set = fileDatastore('./data/COULE_test/*.mat','ReadFcn',@load,'FileExtensions','.mat');

train_set = transform(train_set, @(data) rearrange_datastore(data));
test_set = transform(test_set, @(data) rearrange_datastore(data));

%% Visualize sample image (for reference only)
x = read(train_set);
figure; imshow(x);

%% Compute sinogram, add noise and reconstruct with FBP
theta = linspace(0, 180, 181);
y = radon(x, theta);

noise_level = 0;
e = randn(size(y));
e = e / norm(e) * norm(y) * noise_level;
y_delta = y + e;

x_FBP = iradon(y_delta, theta);

%% Visualize a sample
figure; imshow(x_FBP);



