function [outNet, diag] = prune_model_v2(net, XTrain, YTrain, opts)
%PRUNE_MODEL_V2 Taylor channel pruning with a compute-matched control arm.
%
%   [prunedNet, diag] = PRUNE_MODEL_V2(net, XTrain, YTrain, ...
%                           numIterations=2, maxToPrunePerIteration=4, ...
%                           pruneEnabled=true, seed=1)
%
%   THE CONTROL ARM -- WHY THIS MATTERS
%   -----------------------------------
%   The v1 pruning routine runs a full SGDM fine-tuning pass over the
%   training set inside every pruning iteration. With 2 iterations that is
%   2 extra epochs of gradient updates that the baseline never receives.
%
%   So "pruned" in v1 is not baseline-minus-filters. It is:
%       baseline + 2 epochs of extra training + filter removal
%
%   Reporting that the pruned model retains accuracy "within 1-2 points"
%   therefore compares two things that differ in two ways at once. The extra
%   training plausibly offsets some of the damage from pruning, which means
%   the accuracy-retention claim is not attributable to pruning alone.
%
%   Setting pruneEnabled=false runs the identical loop -- same seed, same
%   shuffle, same number of gradient steps, same updateScore calls -- but
%   never calls updatePrunables. The result is a network that has received
%   exactly the extra training the pruned arm received, with no filters
%   removed. That is the correct control:
%
%       accuracy(pruned) - accuracy(baseline_ft)   <- effect of PRUNING
%       accuracy(baseline_ft) - accuracy(baseline) <- effect of EXTRA TRAINING
%
%   ALSO FIXED: velocity reset
%   --------------------------
%   v1 carried the SGDM `velocity` across pruning iterations. updatePrunables
%   changes the shape of the learnable tensors, so the velocity accumulated
%   against the pre-prune shapes is stale (and shape-mismatched) on the next
%   iteration. Velocity is now reset whenever the network structure changes,
%   which is both safer and semantically correct.

    arguments
        net
        XTrain
        YTrain categorical
        opts.numIterations          (1,1) double = 2
        opts.maxToPrunePerIteration (1,1) double = 4
        opts.pruneEnabled           (1,1) logical = true
        opts.miniBatchSize          (1,1) double = 128
        opts.seed                   (1,1) double = 1
        opts.verbose                (1,1) logical = true
    end

    % Identical data ordering between the pruned arm and its control.
    rng(opts.seed, 'twister');

    dlnetBase = to_dlnetwork(net);

    classes = categories(YTrain);
    TTrain  = onehotencode(YTrain, 2, 'ClassNames', classes)';

    prunableNet = taylorPrunableNetwork(dlnetBase);

    numObservations = size(XTrain, 4);
    numBatches      = floor(numObservations / opts.miniBatchSize);
    velocity        = [];

    diag = struct();
    diag.pruneEnabled  = opts.pruneEnabled;
    diag.seed          = opts.seed;
    diag.gradientSteps = opts.numIterations * numBatches;
    diag.prunablesPerIteration = zeros(1, opts.numIterations);

    for it = 1:opts.numIterations
        idxShuffle = randperm(numObservations);

        for b = 1:numBatches
            batchIdx = idxShuffle((b-1)*opts.miniBatchSize + 1 : b*opts.miniBatchSize);
            X = dlarray(single(XTrain(:, :, :, batchIdx)), 'SSCB');
            Tb = dlarray(single(TTrain(:, batchIdx)), 'CB');

            [~, pruningGradient, pruningActivations, netGradients, state] = ...
                dlfeval(@modelLossPruning, prunableNet, X, Tb);

            prunableNet.State = state;
            [prunableNet, velocity] = sgdmupdate(prunableNet, netGradients, velocity);
            prunableNet = updateScore(prunableNet, pruningActivations, pruningGradient);
        end

        if opts.pruneEnabled
            prunableNet = updatePrunables(prunableNet, ...
                              MaxToPrune = opts.maxToPrunePerIteration);
            % Structure changed -> accumulated momentum is stale.
            velocity = [];
        end

        diag.prunablesPerIteration(it) = prunableNet.NumPrunables;

        if opts.verbose
            fprintf('  [%s] iteration %d/%d complete | remaining prunables: %d\n', ...
                    ternary(opts.pruneEnabled, 'PRUNE', 'CONTROL'), ...
                    it, opts.numIterations, prunableNet.NumPrunables);
        end
    end

    outNet = dlnetwork(prunableNet);

    diag.filterCounts = filter_counts(outNet);
    diag.numLearnables = sum(cellfun(@numel, outNet.Learnables.Value));

    if opts.verbose
        fprintf('  [%s] done. learnables: %d\n', ...
                ternary(opts.pruneEnabled, 'PRUNE', 'CONTROL'), diag.numLearnables);
    end
end

% -------------------------------------------------------------------------
function [loss, pruningGradient, pruningActivations, netGradients, state] = ...
            modelLossPruning(prunableNet, X, T)
    [dlYPred, state, pruningActivations] = forward(prunableNet, X);
    loss = crossentropy(dlYPred, T);
    [pruningGradient, netGradients] = ...
        dlgradient(loss, pruningActivations, prunableNet.Learnables);
end

% -------------------------------------------------------------------------
function s = filter_counts(dlnet)
%FILTER_COUNTS Per-conv-layer filter and parameter counts, read from the
%   actual weight tensors rather than from layer properties, which are not
%   always refreshed after pruning.
    s = struct('layer', {}, 'filters', {}, 'params', {});
    for i = 1:numel(dlnet.Layers)
        L = dlnet.Layers(i);
        if isa(L, 'nnet.cnn.layer.Convolution2DLayer')
            w = size(L.Weights);
            s(end+1) = struct('layer', L.Name, 'filters', w(4), ...
                              'params', numel(L.Weights) + numel(L.Bias)); %#ok<AGROW>
        elseif isa(L, 'nnet.cnn.layer.FullyConnectedLayer')
            s(end+1) = struct('layer', L.Name, 'filters', NaN, ...
                              'params', numel(L.Weights) + numel(L.Bias)); %#ok<AGROW>
        end
    end
end

% -------------------------------------------------------------------------
function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
