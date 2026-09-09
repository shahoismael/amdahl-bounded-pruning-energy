function T = run_experiment_v3(datasetName, XTrain, YTrain, XVal, YVal, XTest, YTest, opts)
%RUN_EXPERIMENT_V3 Audit for one dataset. Persists EVERYTHING needed to
%recompute any reported metric without retraining.
%
%   Supersedes run_experiment_v2. Same experiment, same protocol, same
%   numbers. The only change is what gets written to disk.
%
%   v2 saved top-1 and balanced accuracy as scalars and discarded the
%   networks. Any metric not anticipated at run time was therefore
%   unrecoverable. v3 saves the trained networks, the raw predictions, the
%   per-class recalls, the confusion matrices, the class counts and the
%   environment, so a reviewer can recompute any figure in the paper.
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
        opts.outDir          (1,:) char   = fullfile('..','..','results_v3')
        opts.burnInSec       (1,1) double = 0
        opts.saveNets        (1,1) logical = true
    end

    addpath(fullfile(fileparts(mfilename('fullpath')), '..'));
    if ~exist(opts.outDir, 'dir'), mkdir(opts.outDir); end
    numClasses = numel(categories(YTrain));

    rngStateAtStart = rng(opts.seed, 'twister');

    % ---- 1. Baseline -----------------------------------------------------
    fprintf('\n[%s] training baseline (%s)...\n', datasetName, func2str(opts.trainFcn));
    baselineNet = opts.trainFcn(XTrain, YTrain, XVal, YVal, numClasses);

    % ---- 2. Two derived arms, identical seed -----------------------------
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
    fprintf('[%s] theoretical cost, baseline_ft:\n', datasetName);
    fFt   = count_flops(conditions(2).net, inputSize, opts.dataFormat);
    fprintf('[%s] theoretical cost, pruned:\n', datasetName);
    fPrun = count_flops(conditions(3).net, inputSize, opts.dataFormat);

    theoreticalMACReduction   = (fBase.totalMACs  - fPrun.totalMACs)  / fBase.totalMACs  * 100;
    theoreticalParamReduction = (fBase.learnables - fPrun.learnables) / fBase.learnables * 100;
    fprintf('\n[%s] THEORETICAL MAC reduction   : %.2f%%\n', datasetName, theoreticalMACReduction);
    fprintf('[%s] THEORETICAL param reduction : %.2f%%\n\n', datasetName, theoreticalParamReduction);

    % ---- 5. Accuracy, all three arms, full detail retained ---------------
    eBase = dl_eval(conditions(1).net, XTest, YTest, opts.dataFormat);
    eFt   = dl_eval(conditions(2).net, XTest, YTest, opts.dataFormat);
    ePrun = dl_eval(conditions(3).net, XTest, YTest, opts.dataFormat);

    accBase = eBase.acc;  balBase = eBase.bal;
    accFt   = eFt.acc;    balFt   = eFt.bal;
    accPrun = ePrun.acc;  balPrun = ePrun.bal;

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

    % ---- 6b. Burn-in: drive the processor to a steady thermal and clock
    % state before any measurement begins. Discarded entirely. Needed for
    % heavy workloads where the first blocks otherwise catch the ramp.
    if opts.burnInSec > 0
        fprintf('[%s] burn-in %.0f s (discarded)...\n', datasetName, opts.burnInSec);
        bnet = conditions(1).net;
        t0 = tic; k = 0;
        while toc(t0) < opts.burnInSec
            k = k + 1;
            X = pool{mod(k-1, numel(pool)) + 1};
            Y = predict(bnet, X); %#ok<NASGU>
        end
        fprintf('[%s] burn-in done (%d batches).\n', datasetName, k);
    end

    % ---- 7. Interleaved measurement ---------------------------------------
    T = run_paired_measurement(conditions, pool, ...
            numBlocks = opts.numBlocks, durationSec = opts.durationSec, ...
            settleSec = opts.settleSec, warmupBatches = opts.warmupBatches, ...
            seed = opts.seed);

    % ---- 8. Power, extracted programmatically -----------------------------
    if ~isempty(opts.hwinfoCsv)
        T = attach_power_to_table(T, opts.hwinfoCsv, idleWatts = opts.idleWatts);
    else
        warning('run_experiment_v3:NoLog', ...
                'No HWiNFO CSV supplied; power columns left as NaN.');
    end

    % ---- 9. Annotate table with EVERY per-dataset scalar -------------------
    n = height(T);
    T.dataset                   = repmat(string(datasetName), n, 1);
    T.theoreticalMACReduction   = repmat(theoreticalMACReduction,   n, 1);
    T.theoreticalParamReduction = repmat(theoreticalParamReduction, n, 1);
    T.macsBaseline              = repmat(fBase.totalMACs,  n, 1);
    T.macsPruned                = repmat(fPrun.totalMACs,  n, 1);
    T.paramsBaseline            = repmat(fBase.learnables, n, 1);
    T.paramsPruned              = repmat(fPrun.learnables, n, 1);
    T.accBaseline               = repmat(accBase, n, 1);
    T.accBaselineFt             = repmat(accFt,   n, 1);
    T.accPruned                 = repmat(accPrun, n, 1);
    T.balBaseline               = repmat(balBase, n, 1);
    T.balBaselineFt             = repmat(balFt,   n, 1);
    T.balPruned                 = repmat(balPrun, n, 1);
    T.majorityClassRate         = repmat(majority, n, 1);
    T.batchSizeOpt              = repmat(opts.batchSize, n, 1);
    T.idleWattsOpt              = repmat(opts.idleWatts, n, 1);
    T.seedOpt                   = repmat(opts.seed, n, 1);

    % ---- 10. Environment provenance ---------------------------------------
    env = struct();
    env.timestamp     = datetime('now');
    env.matlabVersion = version;
    env.computer      = computer;
    try, env.toolboxes = struct2table(ver); catch, env.toolboxes = 'unavailable'; end
    try
        [st, gh] = system('git rev-parse HEAD');
        if st == 0, env.gitCommit = strtrim(gh); else, env.gitCommit = 'not a git repo'; end
    catch
        env.gitCommit = 'unavailable';
    end
    env.rngStateAtStart = rngStateAtStart;
    env.scriptPath      = mfilename('fullpath');

    % ---- 11. Persist -------------------------------------------------------
    stamp  = datestr(now, 'yyyymmdd_HHMMSS'); %#ok<TNOW1,DATST>
    csvOut = fullfile(opts.outDir, sprintf('measurements_%s_%s.csv', datasetName, stamp));
    matOut = fullfile(opts.outDir, sprintf('run_%s_%s.mat', datasetName, stamp));
    evalOut = fullfile(opts.outDir, sprintf('eval_%s_%s.csv', datasetName, stamp));

    writetable(T, csvOut);

    % Flat, human-readable accuracy summary. One row per arm.
    E = table( ...
        ["baseline";"baseline_ft";"pruned"], ...
        [eBase.acc; eFt.acc; ePrun.acc], ...
        [eBase.bal; eFt.bal; ePrun.bal], ...
        repmat(majority, 3, 1), ...
        'VariableNames', {'arm','top1','balanced','majorityClassRate'});
    writetable(E, evalOut);

    saveVars = {'T','E','env','opts', ...
                'fBase','fFt','fPrun','ftDiag','prDiag', ...
                'accBase','accFt','accPrun','balBase','balFt','balPrun', ...
                'majority','eBase','eFt','ePrun', ...
                'theoreticalMACReduction','theoreticalParamReduction'};

    if opts.saveNets
        netBaseline = conditions(1).net; %#ok<NASGU>
        netFt       = conditions(2).net; %#ok<NASGU>
        netPruned   = conditions(3).net; %#ok<NASGU>
        saveVars = [saveVars, {'netBaseline','netFt','netPruned'}];
    end

    save(matOut, saveVars{:}, '-v7.3');

    fprintf('[%s] wrote %s\n', datasetName, csvOut);
    fprintf('[%s] wrote %s\n', datasetName, evalOut);
    fprintf('[%s] wrote %s\n', datasetName, matOut);
end

% -------------------------------------------------------------------------
function e = dl_eval(dlnet, X, Y, fmt)
%DL_EVAL Top-1, balanced accuracy, per-class recall, confusion matrix and
%raw predictions. Everything is retained so any other metric can be derived
%later without re-running the network.
    classNames = categories(Y);
    K = numel(classNames);
    n = size(X, 4);
    chunk = 1024;
    pred = zeros(n, 1);
    for i = 1:chunk:n
        j  = min(i + chunk - 1, n);
        Yp = predict(dlnet, dlarray(single(X(:,:,:,i:j)), fmt));
        [~, k] = max(extractdata(Yp), [], 1);
        pred(i:j) = k(:);
    end
    yhat  = categorical(classNames(pred));
    ytrue = Y(:);

    e = struct();
    e.classNames = classNames;
    e.yhat       = yhat(:);
    e.acc        = mean(yhat(:) == ytrue);

    recalls   = nan(K, 1);
    precisions= nan(K, 1);
    support   = zeros(K, 1);
    C         = zeros(K, K);
    for a = 1:K
        m = ytrue == classNames{a};
        support(a) = sum(m);
        if any(m), recalls(a) = mean(yhat(m) == classNames{a}); end
        p = yhat(:) == classNames{a};
        if any(p), precisions(a) = mean(ytrue(p) == classNames{a}); end
        for b = 1:K
            C(a,b) = sum(m & yhat(:) == classNames{b});
        end
    end
    e.recalls    = recalls;
    e.precisions = precisions;
    e.support    = support;
    e.confusion  = C;          % rows = true, cols = predicted
    e.bal        = mean(recalls, 'omitnan');
end
