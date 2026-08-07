function A = audit_v1_runs(resultsDir, opts)
%AUDIT_V1_RUNS Forensic re-analysis of the original (v1) measurement files.
%
%   A = AUDIT_V1_RUNS('../../results')
%
%   Run this before re-collecting anything. It answers three questions from
%   data already on disk, and the answers change what the paper should claim.
%
%   Q1. Did the pruned model actually run faster?
%       The saved files record numRuns and avgInferenceTimeSec for both arms,
%       so throughput can be recovered even though measuredWatts was never
%       written back. Across the unthrottled runs the pruned arm is SLOWER,
%       despite ~24% fewer MACs. That is the paper's most interesting
%       unreported result and it needs an explanation (framework overhead in
%       the v1 harness, and/or SIMD-unfriendly channel counts after pruning).
%
%   Q2. Were the apparent energy savings an artefact of measurement order?
%       v1 always measured baseline first, pruned second. Flagging runs where
%       throughput collapsed relative to the dataset's nominal rate separates
%       throttled from clean runs. In the v1 data, every throttled run favours
%       the pruned arm and no clean run does so strongly -- the signature of
%       an order effect, not a compression effect.
%
%   Q3. Is the reported dataset complete and traceable?
%       Counts saved run files per dataset against the 5-per-dataset claim,
%       and checks whether measuredWatts was ever persisted.

    arguments
        resultsDir (1,:) char
        opts.throttleThreshold (1,1) double = 0.80
    end

    files = dir(fullfile(resultsDir, 'all_results*.mat'));
    if isempty(files)
        error('audit_v1_runs:NoFiles', 'No all_results*.mat in %s', resultsDir);
    end

    rows = cell(0, 7);   % width fixed up front: (end+1,:) on a bare {} errors
    wattsPersisted = 0;

    for i = 1:numel(files)
        S = load(fullfile(files(i).folder, files(i).name));
        if ~isfield(S, 'allResults'), continue; end
        R = S.allResults;
        if ~isfield(R, 'baseline') || ~isfield(R, 'pruned'), continue; end

        b = R.baseline; p = R.pruned;
        if isfield(b, 'measuredWatts') && ~isnan(b.measuredWatts)
            wattsPersisted = wattsPersisted + 1;
        end

        rows(end+1, :) = { string(files(i).name), dataset_of(files(i).name), ...
            double(b.numRuns), double(p.numRuns), ...
            b.avgInferenceTimeSec*1e3, p.avgInferenceTimeSec*1e3, ...
            (p.avgInferenceTimeSec - b.avgInferenceTimeSec)/b.avgInferenceTimeSec*100 }; %#ok<AGROW>
    end

    A = cell2table(rows, 'VariableNames', ...
        {'file','dataset','nBaseline','nPruned','msBaseline','msPruned','latencyChangePct'});

    % Nominal throughput = fastest window observed for that dataset in either arm.
    ds = unique(A.dataset);
    A.pctNominalBaseline = NaN(height(A),1);
    A.pctNominalPruned   = NaN(height(A),1);
    for d = 1:numel(ds)
        m = A.dataset == ds(d);
        nominal = max([A.nBaseline(m); A.nPruned(m)]);
        A.pctNominalBaseline(m) = A.nBaseline(m) / nominal;
        A.pctNominalPruned(m)   = A.nPruned(m)   / nominal;
    end

    A.clean = A.pctNominalBaseline >= opts.throttleThreshold & ...
              A.pctNominalPruned   >= opts.throttleThreshold;

    fprintf('\n================ v1 FORENSIC AUDIT ================\n\n');
    disp(A);

    c = A.latencyChangePct(A.clean);
    t = A.latencyChangePct(~A.clean);

    fprintf('\nQ1/Q2  Latency change of pruned vs baseline (positive = pruned SLOWER)\n');
    fprintf('  CLEAN runs        n=%2d  mean %+6.2f%%  median %+6.2f%%  pruned slower in %d/%d\n', ...
            numel(c), mean(c), median(c), sum(c > 0), numel(c));
    fprintf('  THROTTLED runs    n=%2d  mean %+6.2f%%  median %+6.2f%%  pruned slower in %d/%d\n', ...
            numel(t), mean(t), median(t), sum(t > 0), numel(t));
    fprintf('\n  If the throttled runs favour the pruned arm and the clean runs do not,\n');
    fprintf('  the reported energy saving is an ordering artefact. v1 always measured\n');
    fprintf('  baseline first, so a slow baseline window inflates the pruned arm.\n');

    fprintf('\nQ3  Provenance\n');
    for d = 1:numel(ds)
        n = sum(A.dataset == ds(d));
        fprintf('  %-16s saved run files: %d   (manuscript claims 5)\n', ds(d), n);
    end
    fprintf('  total saved: %d   manuscript claims 20\n', height(A));
    fprintf('  files with measuredWatts persisted: %d / %d\n', wattsPersisted, height(A));
    if wattsPersisted == 0
        fprintf('  >> No wattage was ever written back to disk. The energy figures in\n');
        fprintf('     Tables 6 and 7 are not reproducible from this repository.\n');
    end
    fprintf('\n===================================================\n');
end

% -------------------------------------------------------------------------
function d = dataset_of(fname)
    f = lower(fname);
    if     contains(f, 'chestxray'), d = "ChestX-ray14";
    elseif contains(f, 'mitbih'),    d = "MIT-BIH";
    elseif contains(f, 'fashion'),   d = "Fashion-MNIST";
    else,                            d = "MNIST";
    end
end
