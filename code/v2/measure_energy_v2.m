function r = measure_energy_v2(dlnet, pool, opts)
%MEASURE_ENERGY_V2 Time a single inference condition under controlled conditions.
%
%   r = MEASURE_ENERGY_V2(dlnet, pool)
%   r = MEASURE_ENERGY_V2(dlnet, pool, opts)
%
%   Measures ONE arm. Pairing, interleaving and power-log matching are the
%   job of run_paired_measurement.m -- this function deliberately does one
%   thing so that the timed region is auditable at a glance.
%
%   WHAT CHANGED FROM v1, AND WHY
%   -----------------------------
%   1. Input must already be a dlnetwork (enforced). v1 accepted either a
%      SeriesNetwork or a dlnetwork and branched on isa(net,'dlnetwork'),
%      giving the two arms different execution paths and different per-call
%      allocation costs. Normalise with to_dlnetwork() before calling.
%
%   2. Batched inference. v1 ran batch size 1. On a ~20k-parameter network,
%      a single-sample predict() call is dominated by framework dispatch and
%      argument validation, not by convolution. Pruning cannot reduce
%      dispatch overhead, so a batch-1 harness cannot resolve the effect it
%      is trying to measure. Batch size is now explicit and should be swept
%      (see sweep_batch_size.m) so the overhead-bound and compute-bound
%      regimes can be reported separately.
%
%   3. Warm-up. The first calls pay JIT, lazy buffer allocation and cache
%      cold-start. These are excluded from the measured window.
%
%   4. Nothing is allocated inside the timed loop. Batches are pre-built.
%
%   5. The output is consumed (checksum) so the call cannot be elided and so
%      lazy evaluation cannot shift work outside the window. The checksum
%      cost is identical across arms (same number of classes, same batch).
%
%   6. Fixed iteration count is available as an alternative to fixed
%      duration. Fixed duration confounds throughput with the measured
%      window; fixed iterations makes the two arms do provably equal work.
%
%   OPTIONS
%     .mode          'duration' (default) | 'iterations'
%     .durationSec   seconds of sustained inference   (default 60)
%     .numIterations batches to run in 'iterations' mode (default 5000)
%     .warmupBatches batches discarded before timing  (default 200)
%     .label         char label for logging           (default 'condition')
%     .acceleration  'auto' | 'none'                  (default 'none')
%
%   NOTE ON acceleration: the default is 'none', not 'auto'. 'auto' caches a
%   compiled trace keyed to the network; the two arms are different networks,
%   so they would pay different, non-comparable one-off compilation costs
%   inside the measured window. 'none' is slower but symmetric. Symmetry
%   matters more than speed here.

    arguments
        dlnet (1,1) dlnetwork
        pool  (:,1) cell
        opts.mode          (1,:) char   {mustBeMember(opts.mode, {'duration','iterations'})} = 'duration'
        opts.durationSec   (1,1) double = 60
        opts.numIterations (1,1) double = 5000
        opts.warmupBatches (1,1) double = 200
        opts.label         (1,:) char   = 'condition'
        opts.acceleration  (1,:) char   {mustBeMember(opts.acceleration, {'auto','none'})} = 'none'
    end

    nPool     = numel(pool);
    batchSize = size(pool{1}, finddim(pool{1}, 'B'));

    % ---- Warm-up (excluded from measurement) -----------------------------
    for k = 1:opts.warmupBatches
        Y = predict(dlnet, pool{mod(k-1, nPool) + 1}, 'Acceleration', opts.acceleration);
    end
    clear Y;

    % ---- Timed region ----------------------------------------------------
    r = struct();
    r.label      = opts.label;
    r.batchSize  = batchSize;
    r.startTime  = datetime('now');

    checksum   = 0;
    numBatches = 0;

    t0 = tic;
    if strcmp(opts.mode, 'duration')
        while toc(t0) < opts.durationSec
            Y = predict(dlnet, pool{mod(numBatches, nPool) + 1}, ...
                        'Acceleration', opts.acceleration);
            checksum   = checksum + double(extractdata(sum(Y, 'all')));
            numBatches = numBatches + 1;
        end
    else
        for k = 1:opts.numIterations
            Y = predict(dlnet, pool{mod(k-1, nPool) + 1}, ...
                        'Acceleration', opts.acceleration);
            checksum   = checksum + double(extractdata(sum(Y, 'all')));
        end
        numBatches = opts.numIterations;
    end
    elapsed = toc(t0);
    % ---- End timed region ------------------------------------------------

    r.endTime            = datetime('now');
    r.elapsedSec         = elapsed;
    r.numBatches         = numBatches;
    r.numInferences      = numBatches * batchSize;
    r.secPerBatch        = elapsed / numBatches;
    r.secPerInference    = elapsed / r.numInferences;
    r.throughputPerSec   = r.numInferences / elapsed;
    r.checksum           = checksum;   % retained purely to defeat elision
    r.acceleration       = opts.acceleration;
    r.mode               = opts.mode;

    % Power is filled in later, programmatically, by parse_hwinfo_log.
    r.measuredWatts      = NaN;
    r.wattsSamples       = NaN;
    r.wattsSD            = NaN;
    r.energyJPerInference = NaN;

    % ---- Stall detection --------------------------------------------------
    % In duration mode the loop exits on the FIRST toc past durationSec, so a
    % single hung predict() call can stretch the window arbitrarily. Observed
    % once in practice: a 60 s measurement ran 324 s at one tenth the normal
    % throughput, and the resulting energy figure was 6x too high -- enough to
    % swing a 6-block mean from +13% to -70% on its own.
    %
    % A measurement that overran its budget was not measuring the network; it
    % was measuring whatever else grabbed the CPU. Flag it here so the
    % analysis can drop it by rule rather than by eye.
    r.overrunRatio = elapsed / opts.durationSec;
    r.stalled = strcmp(opts.mode,'duration') && r.overrunRatio > 1.20;

    fprintf('[%-14s] %6d batches x %3d = %9d inferences in %6.2f s | %8.1f inf/s | %.4f ms/batch%s\n', ...
        opts.label, numBatches, batchSize, r.numInferences, elapsed, ...
        r.throughputPerSec, r.secPerBatch * 1e3, ...
        stall_tag(r.stalled, r.overrunRatio));

    if r.stalled
        warning('measure_energy_v2:Stalled', ...
            ['Measurement "%s" ran %.1f s against a %.0f s budget (%.1fx). ' ...
             'The CPU was taken by something else. This block should be excluded.'], ...
            opts.label, elapsed, opts.durationSec, r.overrunRatio);
    end
end

% -------------------------------------------------------------------------
function s = stall_tag(stalled, ratio)
    if stalled
        s = sprintf('   <<< STALLED %.1fx', ratio);
    else
        s = '';
    end
end
