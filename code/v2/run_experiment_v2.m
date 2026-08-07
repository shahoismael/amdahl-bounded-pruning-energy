function T = run_experiment_v2(datasetName, XTrain, YTrain, XVal, YVal, XTest, YTest, opts)
%RUN_EXPERIMENT_V2 End-to-end corrected audit for one dataset.
%
%   T = RUN_EXPERIMENT_V2(name, XTrain, YTrain, XVal, YVal, XTest, YTest, ...)
%
%   Three arms are measured, not two:
%
%     baseline      trained network, untouched
%     baseline_ft   baseline + the SAME extra fine-tuning the pruning loop
%                   applies, with NO filters removed  (compute-matched control)
%     pruned        baseline + that fine-tuning + Taylor filter removal
%
%   The contrast of interest is pruned vs baseline_ft. The contrast
%   baseline_ft vs baseline quantifies how much of v1's reported accuracy
%   retention was actually due to the extra training rather than to pruning.
%
%   All three are normalised to dlnetwork and measured through an identical
%   input pool, interleaved and order-randomised.
%
%   OPERATOR CHECKLIST BEFORE CALLING
%     1. HWiNFO running, sensors-only, CSV logging ON, interval 500 ms.
%     2. Power plan fixed and recorded. Mains power, battery not charging.
%     3. Close everything else. No browser, no antivirus scan, no sync client.
%     4. Let the machine idle 5 minutes so it starts from a steady thermal state.
%     5. Record an idle window with measure_idle_power before you begin.

    arguments
        datasetName (1,:) char
        XTrain
        YTrain categorical
        XVal
        YVal   categorical
        XTest
        YTest  categorical
        opts.numBlocks       (1,1) double = 6
        opts.durationSec     (1,1) double = 60
        opts.settleSec       (1,1) double = 20
        opts.batchSize       (1,1) double = 128
        opts.poolBatches     (1,1) double = 64
        opts.warmupBatches   (1,1) double = 200
        opts.pruneIterations (1,1) double = 2
        opts.maxToPrune      (1,1) double = 4
        opts.seed            (1,1) double = 1
        opts.trainFcn        (1,1) function_handle = @train_baseline
        opts.dataFormat      (1,:) char   = 'SSCB'
        opts.hwinfoCsv       (1,:) char   = ''
        opts.idleWatts       (1,1) double = NaN
        opts.outDir          (1,:) char   = fullfile('..','..','results_v2')
    end

    % v1 helpers (train_baseline, loaders) live one directory up.
    addpath(fullfile(fileparts(mfilename('fullpath')), '..'));

    if ~exist(opts.outDir, 'dir'), mkdir(opts.outDir); end
    numClasses = numel(categories(YTrain));

    % ---- 1. Baseline -----------------------------------------------------
    % trainFcn is injectable so the 1D signal architecture (train_baseline_1d)
    % can be swapped in for MIT-BIH without duplicating this whole pipeline.
    fprintf('\n[%s] training baseline (%s)...\n', datasetName, func2str(opts.trainFcn));
    baselineNet = opts.trainFcn(XTrain, YTrain, XVal, YVal, numClasses);

    % ---- 2. Two derived arms, identical seed -> identical data ordering ---
    fprintf('[%s] fine-tuning control arm (no pruning)...\n', datasetName);
    [ftNet, ftDiag] = prune_model_v2(baselineNet, XTrain, YTrain, ...
        numIterations = opts.pruneIterations, ...
        maxToPrunePerIteration = opts.maxToPrune, ...
        pruneEnabled = false, seed = opts.seed);

    fprintf('[%s] pruning arm...\n', datasetName);
    [prunedNet, prDiag] = prune_model_v2(baselineNet, XTrain, YTrain, ...
        numIterations = opts.pruneIterations, ...
        maxToPrunePerIteration = opts.maxToPrune, ...
        pruneEnabled = true, seed = opts.seed);

    % ---- 3. Normalise every arm to the same framework path ---------------
    conditions = struct( ...
        'name', {'baseline', 'baseline_ft', 'pruned'}, ...
        'net',  {to_dlnetwork(baselineNet), ftNet, prunedNet});

    % ---- 4. Theoretical metrics ------------------------------------------
    inputSize = [size(XTest,1) size(XTest,2) size(XTest,3)];
    fprintf('\n[%s] theoretical cost, baseline:\n', datasetName);
    fBase = count_flops(conditions(1).net, inputSize, opts.dataFormat);
    fprintf('[%s] theoretical cost, pruned:\n', datasetName);
    fPrun = count_flops(conditions(3).net, inputSize, opts.dataFormat);

    theoreticalMACReduction   = (fBase.totalMACs  - fPrun.totalMACs)  / fBase.totalMACs  * 100;
    theoreticalParamReduction = (fBase.learnables - fPrun.learnables) / fBase.learnables * 100;
    fprintf('\n[%s] THEORETICAL MAC reduction   : %.2f%%\n', datasetName, theoreticalMACReduction);
    fprintf('[%s] THEORETICAL param reduction : %.2f%%\n\n', datasetName, theoreticalParamReduction);

    % ---- 5. Accuracy, all three arms -------------------------------------
    [accBase, balBase] = dl_accuracy(conditions(1).net, XTest, YTest, opts.dataFormat);
    [accFt,   balFt]   = dl_accuracy(conditions(2).net, XTest, YTest, opts.dataFormat);
    [accPrun, balPrun] = dl_accuracy(conditions(3).net, XTest, YTest, opts.dataFormat);

    % Raw accuracy is uninterpretable on an imbalanced test split: a constant
    % majority-class predictor scores whatever the majority share is. Balanced
    % accuracy (mean per-class recall) is reported alongside, and the majority
    % baseline is printed so it is obvious when raw accuracy is meaningless.
    cc = countcats(YTest(:));
    majority = max(cc) / sum(cc);

    fprintf('[%s] test set: %s\n', datasetName, ...
            strjoin(compose('%s=%d', string(categories(YTest)), cc(:)), '  '));
    fprintf('[%s] majority-class baseline: %.2f%%\n', datasetName, majority*100);
    fprintf('[%s] accuracy   baseline %.2f%%  baseline_ft %.2f%%  pruned %.2f%%\n', ...
            datasetName, accBase*100, accFt*100, accPrun*100);
    fprintf('[%s] balanced   baseline %.2f%%  baseline_ft %.2f%%  pruned %.2f%%\n', ...
            datasetName, balBase*100, balFt*100, balPrun*100);
    fprintf('[%s]   effect of extra training : %+.2f pp balanced\n', datasetName, (balFt-balBase)*100);
    fprintf('[%s]   effect of pruning alone  : %+.2f pp balanced\n', datasetName, (balPrun-balFt)*100);
    if majority > 0.90
        fprintf('[%s]   WARNING: test split is %.1f%% one class. Report balanced accuracy only.\n', ...
                datasetName, majority*100);
    end
    fprintf('\n');

    % ---- 6. Shared input pool, built once ---------------------------------
    pool = prepare_input_pool(XTest, opts.batchSize, opts.poolBatches, ...
                              opts.dataFormat, opts.seed);

    % ---- 7. Interleaved measurement ---------------------------------------
    T = run_paired_measurement(conditions, pool, ...
            numBlocks = opts.numBlocks, durationSec = opts.durationSec, ...
            settleSec = opts.settleSec, warmupBatches = opts.warmupBatches, ...
            seed = opts.seed);

    % ---- 8. Power, extracted programmatically -----------------------------
    if ~isempty(opts.hwinfoCsv)
        T = attach_power_to_table(T, opts.hwinfoCsv, idleWatts = opts.idleWatts);
    else
        warning('run_experiment_v2:NoLog', ...
                'No HWiNFO CSV supplied; power columns left as NaN.');
    end

    % ---- 9. Annotate and persist -------------------------------------------
    T.dataset                   = repmat(string(datasetName), height(T), 1);
    T.theoreticalMACReduction   = repmat(theoreticalMACReduction,   height(T), 1);
    T.theoreticalParamReduction = repmat(theoreticalParamReduction, height(T), 1);
    T.accBaseline               = repmat(accBase, height(T), 1);
    T.accBaselineFt             = repmat(accFt,   height(T), 1);
    T.accPruned                 = repmat(accPrun, height(T), 1);

    stamp   = datestr(now, 'yyyymmdd_HHMMSS'); %#ok<TNOW1,DATST>
    csvOut  = fullfile(opts.outDir, sprintf('measurements_%s_%s.csv', datasetName, stamp));
    matOut  = fullfile(opts.outDir, sprintf('run_%s_%s.mat', datasetName, stamp));
    writetable(T, csvOut);
    save(matOut, 'T', 'fBase', 'fPrun', 'ftDiag', 'prDiag', ...
         'accBase', 'accFt', 'accPrun', 'opts', '-v7.3');

    fprintf('[%s] wrote %s\n', datasetName, csvOut);
end

% -------------------------------------------------------------------------
function [acc, bal] = dl_accuracy(dlnet, X, Y, fmt)
%DL_ACCURACY Top-1 accuracy and balanced accuracy, evaluated in chunks.
%
%   acc : proportion correct overall
%   bal : mean per-class recall -- the honest figure when the test split is
%         imbalanced, since a constant majority-class predictor scores 0.5
%         on a two-class problem regardless of how skewed the split is.
    classNames = categories(Y);
    n = size(X, 4);
    chunk = 1024;
    pred = zeros(n, 1);
    for i = 1:chunk:n
        j  = min(i + chunk - 1, n);
        Yp = predict(dlnet, dlarray(single(X(:,:,:,i:j)), fmt));
        [~, k] = max(extractdata(Yp), [], 1);
        pred(i:j) = k(:);
    end
    yhat = categorical(classNames(pred));
    ytrue = Y(:);
    acc = mean(yhat(:) == ytrue);

    recalls = nan(numel(classNames), 1);
    for c = 1:numel(classNames)
        m = ytrue == classNames{c};
        if any(m), recalls(c) = mean(yhat(m) == classNames{c}); end
    end
    bal = mean(recalls, 'omitnan');
end
