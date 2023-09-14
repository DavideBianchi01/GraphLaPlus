%% Load COULE Data
train_set = imageDatastore('../data/Mayo_train/C077/*.png');
test_set = imageDatastore('../data/Mayo_test/C081/*.png');

%% Convert it in [0, 1] and cast it to double.
train_set = transform(train_set, @(x) rgb2gray(x));
train_set = transform(train_set, @(x) double(x));
train_set = transform(train_set, @(x) (x - min(x, [], 'all')) / (max(x, [], 'all') - min(x, [], 'all')));

test_set = transform(test_set, @(x) rgb2gray(x));
test_set = transform(test_set, @(x) double(x));
test_set = transform(test_set, @(x) (x - min(x, [], 'all')) / (max(x, [], 'all') - min(x, [], 'all')));

%% Convert to mat
for i = 1:358
    x = read(train_set);
    filename = sprintf("../data/Mayo_train/C077_%i.mat", i-1);
    save(filename, "x");
end

for i = 1:327
    x = read(test_set);
    filename = sprintf("../data/Mayo_test/C081_%i.mat", i-1);
    save(filename, "x");
end