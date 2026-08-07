%% MAIN PIPELINE - Fashion-MNIST
% Same structure as main_pipeline.m (MNIST), pointed at Fashion-MNIST data.
% Fashion-MNIST repo files are typically gzipped; this script unzips them
% automatically into a local subfolder if needed.

clear; clc;

%% 1. Load dataset
dataDir = fullfile('..','data','2_Fashion_MNIST','data','fashion');

% Auto-unzip .gz files if present and not yet extracted
gzFiles = {'train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz', ...
           't10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz'};
for i = 1:numel(gzFiles)
    gzPath = fullfile(dataDir, gzFiles{i});
    unzippedPath = fullfile(dataDir, erase(gzFiles{i}, '.gz'));
    if exist(gzPath, 'file') && ~exist(unzippedPath, 'file')
        gunzip(gzPath, dataDir);
    end
end

[XTrain, YTrain] = load_idx_dataset( ...
    fullfile(dataDir, 'train-images-idx3-ubyte'), ...
    fullfile(dataDir, 'train-labels-idx1-ubyte'));
[XTest, YTest] = load_idx_dataset( ...
    fullfile(dataDir, 't10k-images-idx3-ubyte'), ...
    fullfile(dataDir, 't10k-labels-idx1-ubyte'));

% simple train/val split
splitIdx = round(0.9 * size(XTrain,4));
XVal = XTrain(:,:,:,splitIdx+1:end);
YVal = YTrain(splitIdx+1:end);
XTrain = XTrain(:,:,:,1:splitIdx);
YTrain = YTrain(1:splitIdx);

numClasses = numel(categories(YTrain));

%% 2. Train baseline
baselineNet = train_baseline(XTrain, YTrain, XVal, YVal, numClasses);
save(fullfile('..','results','baseline_net_fashion.mat'), 'baselineNet');

%% 3. Prune (same light settings validated on MNIST: 2 iterations, 4 filters/iter)
prunedNet = prune_model(baselineNet, XTrain, YTrain, 2, 4);
save(fullfile('..','results','pruned_net_fashion.mat'), 'prunedNet');

%% 3b. Check pruned model accuracy
YPredPrunedRaw = predict(prunedNet, dlarray(single(XTest), 'SSCB'));
[~, predIdx] = max(extractdata(YPredPrunedRaw), [], 1);
classNames = categories(YTest);
YPredPruned = categorical(classNames(predIdx));
accPruned = mean(YPredPruned(:) == YTest(:));
fprintf('Pruned model accuracy (Fashion-MNIST): %.2f%%\n', accPruned*100);

%% 4. Quantize
quantizedNet = quantize_model(baselineNet, XTrain(:,:,:,1:100));
save(fullfile('..','results','quantized_net_fashion.mat'), 'quantizedNet');

%% 5. Measure energy (30-second sustained runs, matched against HWiNFO CSV log)
resultsBaseline = measure_energy(baselineNet, XTest, 30, 'Baseline-Fashion');
resultsPruned   = measure_energy(prunedNet,   XTest, 30, 'Pruned-Fashion');

%% 6. Save results
runN = get_next_run_number(fullfile('..','results'), 'all_results_fashion');
allResults = struct('baseline', resultsBaseline, 'pruned', resultsPruned);
save(fullfile('..','results', sprintf('all_results_fashion_run%d.mat', runN)), 'allResults');
fprintf('Saved as run #%d.\n', runN);

disp('Fashion-MNIST pipeline complete. Extract watts from HWiNFO CSV log using printed timestamps.');
