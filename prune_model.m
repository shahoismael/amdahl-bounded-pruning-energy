function prunedNet = prune_model(net, XTrain, YTrain, numPruningIterations, maxToPrunePerIteration)
% PRUNE_MODEL Manual Taylor pruning loop using taylorPrunableNetwork,
% updateScore, and updatePrunables (low-level API, confirmed available).
%
%   prunedNet = prune_model(net, XTrain, YTrain, numPruningIterations, maxToPrunePerIteration)
%   numPruningIterations   : number of prune+fine-tune cycles, e.g. 5
%   maxToPrunePerIteration : filters removed per cycle, e.g. 8 (default in MATLAB)

    if nargin < 4, numPruningIterations = 5; end
    if nargin < 5, maxToPrunePerIteration = 8; end

    % Convert trained SeriesNetwork/DAGNetwork -> dlnetwork (strip classificationLayer only)
    lgraph = layerGraph(net);
    % Auto-detect the classification output layer's name (varies: 'output',
    % 'classoutput', etc. depending on whether it was explicitly named).
    layerTypes = arrayfun(@(l) class(l), lgraph.Layers, 'UniformOutput', false);
    isClassOutput = contains(layerTypes, 'ClassificationOutputLayer');
    outputLayerName = lgraph.Layers(isClassOutput).Name;
    lgraph = removeLayers(lgraph, {outputLayerName});
    dlnetBase = dlnetwork(lgraph);

    % One-hot encode labels for crossentropy loss, oriented [numClasses x numObs]
    classes = categories(YTrain);
    TTrain = onehotencode(YTrain, 2, 'ClassNames', classes)';

    prunableNet = taylorPrunableNetwork(dlnetBase);

    miniBatchSize = 128;
    numObservations = size(XTrain, 4);
    velocity = [];

    for pruningIter = 1:numPruningIterations
        idxShuffle = randperm(numObservations);
        numBatches = floor(numObservations / miniBatchSize);

        for b = 1:numBatches
            batchIdx = idxShuffle((b-1)*miniBatchSize+1 : b*miniBatchSize);
            X = dlarray(single(XTrain(:,:,:,batchIdx)), 'SSCB');
            T = dlarray(single(TTrain(:,batchIdx)), 'CB');

            [~, pruningGradient, pruningActivations, netGradients, state] = ...
                dlfeval(@modelLossPruning, prunableNet, X, T);

            prunableNet.State = state;
            [prunableNet, velocity] = sgdmupdate(prunableNet, netGradients, velocity);
            prunableNet = updateScore(prunableNet, pruningActivations, pruningGradient);
        end

        prunableNet = updatePrunables(prunableNet, MaxToPrune=maxToPrunePerIteration);
        fprintf('Pruning iteration %d/%d complete. Remaining prunable filters: %d\n', ...
            pruningIter, numPruningIterations, prunableNet.NumPrunables);
    end

    prunedNet = dlnetwork(prunableNet);
    fprintf('Pruning complete.\n');
end

function [loss, pruningGradient, pruningActivations, netGradients, state] = modelLossPruning(prunableNet, X, T)
    [dlYPred, state, pruningActivations] = forward(prunableNet, X);
    loss = crossentropy(dlYPred, T);
    [pruningGradient, netGradients] = dlgradient(loss, pruningActivations, prunableNet.Learnables);
end
