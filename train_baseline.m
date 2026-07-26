function net = train_baseline(XTrain, YTrain, XVal, YVal, numClasses)
% TRAIN_BASELINE Trains a small baseline CNN classifier.
%
%   net = train_baseline(XTrain, YTrain, XVal, YVal, numClasses)

    inputSize = [size(XTrain,1) size(XTrain,2) size(XTrain,3)];

    layers = [
        imageInputLayer(inputSize, 'Name', 'input')

        convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')

        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')

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
        'Plots', 'training-progress');

    net = trainNetwork(XTrain, YTrain, layers, options);
end
