%% MAIN PIPELINE - MIT-BIH Arrhythmia (ECG signal classification: Normal vs Abnormal)
% Different modality from MNIST/Fashion-MNIST: 1D physiological signal.

clear; clc;

dataDir = fullfile('..','data','4_MITBIH','mit-bih-arrhythmia-database-1.0.0');

% A representative subset of records (add more record IDs as needed)
recordIDs = {'100','101','103','105','106','108','109','111', ...
             '112','113','114','115','116','117','118','119'};

windowSamples = 250; % ~0.7s window at 360 Hz, centered on each beat
XAll = [];
YAll = [];

for r = 1:numel(recordIDs)
    recPath = fullfile(dataDir, recordIDs{r});
    try
        [signal, fs, ~] = load_mitbih_record(recPath);
        [sampleIdx, typeCodes] = read_atr_annotations([recPath '.atr']);

        sig = signal(:,1); % use first channel
        half = floor(windowSamples/2);

        for k = 1:numel(sampleIdx)
            c = sampleIdx(k);
            if c-half >= 1 && c+half <= numel(sig)
                seg = sig(c-half:c+half-1);
                XAll(:,1,1,end+1) = seg; %#ok<AGROW> % [time x 1 x 1 x N]
                % Type code 1 = Normal beat 'N' in MIT-BIH annotation coding.
                % Label: 1 = Normal, 2 = Abnormal (everything else)
                if typeCodes(k) == 1
                    YAll(end+1,1) = 1; %#ok<AGROW>
                else
                    YAll(end+1,1) = 2; %#ok<AGROW>
                end
            end
        end
    catch ME
        fprintf('Skipping record %s due to error: %s\n', recordIDs{r}, ME.message);
    end
end

YAll = categorical(YAll, [1 2], {'Normal','Abnormal'});
fprintf('Total beat segments extracted: %d\n', numel(YAll));

% ---- Train/val/test split ----
numObs = numel(YAll);
idxShuffle = randperm(numObs);
splitTrain = round(0.7*numObs);
splitVal   = round(0.85*numObs);

XTrain = XAll(:,:,:, idxShuffle(1:splitTrain));
YTrain = YAll(idxShuffle(1:splitTrain));
XVal   = XAll(:,:,:, idxShuffle(splitTrain+1:splitVal));
YVal   = YAll(idxShuffle(splitTrain+1:splitVal));
XTest  = XAll(:,:,:, idxShuffle(splitVal+1:end));
YTest  = YAll(idxShuffle(splitVal+1:end));

numClasses = numel(categories(YTrain));

%% Train baseline (dedicated 1D-aware architecture for signal data)
baselineNet = train_baseline_1d(XTrain, YTrain, XVal, YVal, numClasses);
save(fullfile('..','results','baseline_net_mitbih.mat'), 'baselineNet');

%% Prune (same light settings validated on MNIST)
prunedNet = prune_model(baselineNet, XTrain, YTrain, 2, 4);
save(fullfile('..','results','pruned_net_mitbih.mat'), 'prunedNet');

%% Check pruned accuracy
YPredPrunedRaw = predict(prunedNet, dlarray(single(XTest), 'SSCB'));
[~, predIdx] = max(extractdata(YPredPrunedRaw), [], 1);
classNames = categories(YTest);
YPredPruned = categorical(classNames(predIdx));
accPruned = mean(YPredPruned(:) == YTest(:));
fprintf('Pruned model accuracy (MIT-BIH): %.2f%%\n', accPruned*100);

%% Quantize
quantizedNet = quantize_model(baselineNet, XTrain(:,:,:,1:min(100,size(XTrain,4))));
save(fullfile('..','results','quantized_net_mitbih.mat'), 'quantizedNet');

%% Measure energy
resultsBaseline = measure_energy(baselineNet, XTest, 30, 'Baseline-MITBIH');
resultsPruned   = measure_energy(prunedNet,   XTest, 30, 'Pruned-MITBIH');

%% Save results (auto-numbered)
runN = get_next_run_number(fullfile('..','results'), 'all_results_mitbih');
allResults = struct('baseline', resultsBaseline, 'pruned', resultsPruned);
save(fullfile('..','results', sprintf('all_results_mitbih_run%d.mat', runN)), 'allResults');
fprintf('Saved as run #%d.\n', runN);

disp('MIT-BIH pipeline complete. Extract watts from HWiNFO CSV log using printed timestamps.');
