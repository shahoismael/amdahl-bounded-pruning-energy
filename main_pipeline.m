%% MAIN PIPELINE - Green Illusion energy audit
% Run this after datasets are placed in ..\data\<dataset_folder>\

clear; clc;

%% 1. Load dataset (example: MNIST)
dataDir = fullfile('..','data','1_MNIST');
[XTrain, YTrain] = load_idx_dataset( ...
    fullfile(dataDir, 'train-images.idx3-ubyte'), ...
    fullfile(dataDir, 'train-labels.idx1-ubyte'));
[XTest, YTest] = load_idx_dataset( ...
    fullfile(dataDir, 't10k-images.idx3-ubyte'), ...
    fullfile(dataDir, 't10k-labels.idx1-ubyte'));

% simple train/val split
splitIdx = round(0.9 * size(XTrain,4));
XVal = XTrain(:,:,:,splitIdx+1:end);
YVal = YTrain(splitIdx+1:end);
XTrain = XTrain(:,:,:,1:splitIdx);
YTrain = YTrain(1:splitIdx);

numClasses = numel(categories(YTrain));

%% 2. Train baseline
baselineNet = train_baseline(XTrain, YTrain, XVal, YVal, numClasses);
save(fullfile('..','results','baseline_net.mat'), 'baselineNet');

%% 3. Prune
prunedNet = prune_model(baselineNet, XTrain, YTrain, 2, 4); % 2 iterations, 4 filters/iteration
save(fullfile('..','results','pruned_net.mat'), 'prunedNet');

%% 4. Quantize
quantizedNet = quantize_model(baselineNet, XTrain(:,:,:,1:100)); % calibration subset
save(fullfile('..','results','quantized_net.mat'), 'quantizedNet');

%% 5. Measure energy (run alongside HWiNFO CSV logging)
resultsBaseline  = measure_energy(baselineNet,  XTest, 30, 'Baseline'); % 30-second sustained run
resultsPruned    = measure_energy(prunedNet,    XTest, 30, 'Pruned');
% NOTE: quantizedNet requires validate()/predict() via dlquantizer object;
% adapt measure_energy accordingly for the quantized case.

%% 6. Save all results for the paper's Results section
runN = get_next_run_number(fullfile('..','results'), 'all_results_mnist');
allResults = struct('baseline', resultsBaseline, 'pruned', resultsPruned);
save(fullfile('..','results', sprintf('all_results_mnist_run%d.mat', runN)), 'allResults');
fprintf('Saved as run #%d.\n', runN);

disp('Pipeline complete. Fill in measuredWatts values from your power meter, then re-run energy ratio calculations.');

%% 3b. Check pruned model accuracy (sanity check before trusting energy numbers)
YPredPrunedRaw = predict(prunedNet, dlarray(single(XTest), 'SSCB'));
[~, predIdx] = max(extractdata(YPredPrunedRaw), [], 1);
classNames = categories(YTest);
YPredPruned = categorical(classNames(predIdx));
accPruned = mean(YPredPruned(:) == YTest(:));
fprintf('Pruned model accuracy: %.2f%%\n', accPruned*100);
