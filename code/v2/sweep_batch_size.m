function S = sweep_batch_size(conditions, XTest, opts)
%SWEEP_BATCH_SIZE Locate the batch size at which the pruning effect becomes resolvable.
%
%   S = SWEEP_BATCH_SIZE(conditions, XTest, batchSizes=[1 8 32 128 512])
%
%   WHY THIS IS THE MOST VALUABLE SINGLE ADDITION TO THE PAPER
%   ----------------------------------------------------------
%   v1 measured at batch size 1 and found no energy effect. That result is
%   ambiguous between two very different explanations:
%
%     (a) pruning genuinely does not reduce energy, or
%     (b) at batch 1 the measurement is dominated by framework dispatch
%         overhead, which pruning cannot touch, so no harness at that batch
%         size could have detected an effect of any size.
%
%   A sweep distinguishes them. If the measured saving stays near zero at
%   batch 1 and grows toward the theoretical prediction as the batch size
%   rises, the honest conclusion is (b) plus a statement about the regime in
%   which compression claims are and are not meaningful. If it stays flat
%   across three orders of magnitude, (a) is supported far more strongly than
%   any single-batch-size experiment could support it.
%
%   Either outcome is publishable and both are more informative than v1.
%   This is a latency sweep only -- fast, no power logging required -- which
%   makes it cheap to run before committing to a full re-measurement.

    arguments
        conditions (1,:) struct
        XTest
        opts.batchSizes    (1,:) double = [1 8 32 128 512]
        opts.poolBatches   (1,1) double = 32
        opts.durationSec   (1,1) double = 20
        opts.warmupBatches (1,1) double = 50
        opts.repeats       (1,1) double = 3
        opts.burnInReps    (1,1) double = 2
        opts.dataFormat    (1,:) char   = 'SSCB'
        opts.referenceArm  (1,:) char   = 'baseline'
        opts.seed          (1,1) double = 7
    end

    % BURN-IN. The per-measurement warm-up inside measure_energy_v2 counts
    % BATCHES, so at batch size 1 it covers a fraction of a second -- far too
    % short to absorb MATLAB's first-call JIT and lazy buffer allocation.
    % Observed effect: the first three replicates at bs=1 reported the pruned
    % network at 4.5 ms/inference against a 1.2 ms steady state, which reads
    % as a catastrophic 236% slowdown that does not exist.
    %
    % Replicates 1..burnInReps are therefore measured and DISCARDED. They are
    % still run, so the machine reaches the same state it would have anyway.

    rows = cell(0, 5);   % width fixed up front: (end+1,:) on a bare {} errors

    for bs = opts.batchSizes
        if size(XTest, 4) < bs
            warning('sweep_batch_size:Skip', ...
                    'Skipping batch size %d: only %d test observations.', bs, size(XTest,4));
            continue;
        end

        % Smaller pool at large batch sizes to keep memory bounded.
        poolN = max(4, min(opts.poolBatches, floor(4096 / bs)));
        pool  = prepare_input_pool(XTest, bs, poolN, opts.dataFormat, opts.seed);

        % Warm-up batches are a COUNT, so at small batch sizes they cover far
        % less wall-clock time. Scale so every batch size gets a comparable
        % warm-up duration.
        warmN = max(opts.warmupBatches, ceil(opts.warmupBatches * 128 / bs));

        for rep = 1:(opts.burnInReps + opts.repeats)
            isBurnIn = rep <= opts.burnInReps;
            order = randperm(numel(conditions));   % randomise within replicate
            for c = order
                tag = sprintf('%s bs=%d', conditions(c).name, bs);
                if isBurnIn, tag = ['(burn-in) ' tag]; end %#ok<AGROW>

                r = measure_energy_v2(conditions(c).net, pool, ...
                        durationSec   = opts.durationSec, ...
                        warmupBatches = warmN, ...
                        label         = tag);

                if isBurnIn, continue; end   % measured, then discarded

                rows(end+1, :) = {bs, rep - opts.burnInReps, ...
                                  string(conditions(c).name), ...
                                  r.secPerInference, r.throughputPerSec}; %#ok<AGROW>
            end
        end
    end

    S = cell2table(rows, 'VariableNames', ...
        {'batchSize','repeat','condition','secPerInference','throughputPerSec'});

    % ---- Summary: latency change of pruned vs reference, per batch size ---
    fprintf('\n=========== BATCH SIZE SWEEP ===========\n');
    fprintf('%10s %16s %16s %14s\n', 'batchSize', 'ref us/inf', 'pruned us/inf', 'pruned change');
    fprintf('%s\n', repmat('-', 1, 60));

    bss = unique(S.batchSize);
    summary = cell(numel(bss), 1);
    for i = 1:numel(bss)
        m   = S.batchSize == bss(i);
        ref = mean(S.secPerInference(m & S.condition == string(opts.referenceArm)));
        prn = mean(S.secPerInference(m & S.condition == "pruned"));
        chg = (prn - ref) / ref * 100;
        fprintf('%10d %16.3f %16.3f %+13.2f%%\n', bss(i), ref*1e6, prn*1e6, chg);
        summary{i} = struct('batchSize', bss(i), 'refSecPerInf', ref, ...
                            'prunedSecPerInf', prn, 'latencyChangePct', chg);
    end
    fprintf('========================================\n');
    fprintf('Read this as: negative = pruning helps. If the column trends from ~0\n');
    fprintf('at batch 1 toward the theoretical MAC reduction at large batch, the v1\n');
    fprintf('null was an artefact of measuring in the overhead-bound regime.\n\n');

    S = struct('raw', S, 'summary', struct2table([summary{:}]));
end
