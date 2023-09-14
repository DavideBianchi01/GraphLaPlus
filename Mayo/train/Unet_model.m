function Unet=Unet_model(inputShape, startChannels)
layers = [
    % Input
    imageInputLayer(inputShape, Normalization="none")
    
    %%% ENCODER
    % First level
    convolution2dLayer(3, startChannels, Padding="same")
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3, startChannels, Padding="same")
    batchNormalizationLayer
    reluLayer("Name", "down_1")

    % Second level
    maxPooling2dLayer(2, Padding="same", Stride=2)
    convolution2dLayer(3, startChannels*2, Padding="same")
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3, startChannels*2, Padding="same")
    batchNormalizationLayer
    reluLayer("Name", "down_2")

    % Third level
    maxPooling2dLayer(2, Padding="same", Stride=2)
    convolution2dLayer(3, startChannels*4, Padding="same")
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3, startChannels*4, Padding="same")
    batchNormalizationLayer
    reluLayer("Name", "down_3")

    % Fourth level
    maxPooling2dLayer(2, Padding="same", Stride=2)
    convolution2dLayer(3, startChannels*8, Padding="same")
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3, startChannels*8, Padding="same")
    batchNormalizationLayer
    reluLayer
   
    convolution2dLayer(3, startChannels*8, Padding="same")
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3, startChannels*8, Padding="same")
    batchNormalizationLayer
    reluLayer

    %%% DECODER
    % Third level
    transposedConv2dLayer(2, startChannels*4, Stride=2)
    additionLayer(2, Name="up_3")
    convolution2dLayer(3, startChannels*4, Padding="same")
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3, startChannels*4, Padding="same")
    batchNormalizationLayer
    reluLayer

    % Second level
    transposedConv2dLayer(2, startChannels*2, 'Stride', 2)
    additionLayer(2, Name="up_2")
    convolution2dLayer(3, startChannels*2, Padding="same")
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3, startChannels*2, Padding="same")
    batchNormalizationLayer
    reluLayer

    % First level
    transposedConv2dLayer(2, startChannels, 'Stride', 2)
    additionLayer(2, Name="up_1")
    convolution2dLayer(3, startChannels, Padding="same")
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3, startChannels, Padding="same")
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3, startChannels, Padding="same")
    batchNormalizationLayer
    reluLayer

    % Output layer
    convolution2dLayer(1, 1, Padding="same")
    reluLayer
    ];

% Define the layer graph
lgraph = layerGraph(layers);

% Add skip connections
lgraph = connectLayers(lgraph, 'down_3', 'up_3/in2');
lgraph = connectLayers(lgraph, 'down_2', 'up_2/in2');
lgraph = connectLayers(lgraph, 'down_1', 'up_1/in2');

% Create the model
Unet = dlnetwork(lgraph);