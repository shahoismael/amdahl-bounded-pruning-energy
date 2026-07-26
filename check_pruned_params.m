%% CHECK_PRUNED_PARAMS - Reports exact per-layer filter/parameter counts
%% before and after pruning, using the saved baseline and pruned networks.

clear; clc;

load(fullfile('..','results','baseline_net.mat'), 'baselineNet');
load(fullfile('..','results','pruned_net.mat'), 'prunedNet');

fprintf('=== BASELINE NETWORK ===\n');
baseLayers = baselineNet.Layers;
totalBaseParams = 0;
for i = 1:numel(baseLayers)
    l = baseLayers(i);
    if isprop(l, 'Weights') && ~isempty(l.Weights)
        w = numel(l.Weights);
        b = 0;
        if isprop(l, 'Bias') && ~isempty(l.Bias)
            b = numel(l.Bias);
        end
        totalBaseParams = totalBaseParams + w + b;
        if isa(l, 'nnet.cnn.layer.Convolution2DLayer')
            fprintf('%s: %d filters, weight size %s, params = %d\n', ...
                l.Name, l.NumFilters, mat2str(size(l.Weights)), w+b);
        else
            fprintf('%s: params = %d\n', l.Name, w+b);
        end
    end
end
fprintf('TOTAL baseline trainable parameters: %d\n\n', totalBaseParams);

fprintf('=== PRUNED NETWORK ===\n');
prunedLayers = prunedNet.Layers;
totalPrunedParams = 0;
for i = 1:numel(prunedLayers)
    l = prunedLayers(i);
    if isprop(l, 'Weights') && ~isempty(l.Weights)
        w = numel(l.Weights);
        b = 0;
        if isprop(l, 'Bias') && ~isempty(l.Bias)
            b = numel(l.Bias);
        end
        totalPrunedParams = totalPrunedParams + w + b;
        if isa(l, 'nnet.cnn.layer.Convolution2DLayer')
            fprintf('%s: %d filters, weight size %s, params = %d\n', ...
                l.Name, l.NumFilters, mat2str(size(l.Weights)), w+b);
        else
            fprintf('%s: params = %d\n', l.Name, w+b);
        end
    end
end
fprintf('TOTAL pruned trainable parameters: %d\n\n', totalPrunedParams);

reduction = (totalBaseParams - totalPrunedParams) / totalBaseParams * 100;
fprintf('Parameter reduction: %.2f%%\n', reduction);
