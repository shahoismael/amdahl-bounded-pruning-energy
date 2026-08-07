function net = train_baseline_1d(XTrain, YTrain, XVal, YVal, numClasses)
% TRAIN_BASELINE_1D Trains a small CNN for 1D signal data shaped
% [windowSamples x 1 x 1 x N]. Pooling/conv kernels only act along the
% signal (height) dimension, leaving the singleton width dimension alone.

    inputSize = [size(XTrain,1) size(XTrain,2) size(XTrain,3)];

    layers = [
        imageInputLayer(inputSize, 'Name', 'input')

        convolution2dLayer([5 1], 16, 'Padding', 'same', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'pool1')

        convolution2dLayer([5 1], 32, 'Padding', 'same', 'Name', 'conv2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'pool2')

        fullyConnectedLayer(numClasses, 'Name', 'fc')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 128, ...
        'ValidationData', {XVal, YVal}, ...
        'ValidationFrequency', 50, ...
        'Verbose', true, ...
        'Plots', 'none');

    net = trainNetwork(XTrain, YTrain, layers, options);
end
