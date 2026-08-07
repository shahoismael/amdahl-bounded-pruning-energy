function T = run_paired_measurement(conditions, pool, opts)
%RUN_PAIRED_MEASUREMENT Interleaved, order-randomised measurement of N arms.
%
%   T = RUN_PAIRED_MEASUREMENT(conditions, pool, opts)
%
%   conditions : struct array with fields .name (char) and .net (dlnetwork)
%   pool       : shared, pre-built input pool (see prepare_input_pool)
%   returns    : table, one row per (block x condition) measurement
%
%   WHY THIS EXISTS -- THE SINGLE MOST IMPORTANT FIX
%   ------------------------------------------------
%   In v1 the baseline was always measured first and the pruned model second,
%   once per run. Any drift in machine state between the two measurements --
%   thermal throttling, a background process waking up, a power-state
%   transition -- is therefore perfectly confounded with the condition.
%
%   That confound is not hypothetical in this dataset. Re-analysis of the
%   saved v1 run files shows that in every run where the machine was
%   throttled (inference count far below the nominal rate for that dataset),
%   the pruned arm looked faster -- 5 runs out of 5. In the unthrottled runs
%   the pruned arm was SLOWER on average. The reported energy "savings" track
%   which arm happened to be measured while the CPU was slow, not which arm
%   had fewer filters.
%
%   The fix is standard experimental practice and costs nothing but wall
%   clock time:
%     - INTERLEAVE. Alternate arms many times instead of measuring each once.
%     - RANDOMISE ORDER within each block, so position cannot correlate with
%       condition.
%     - PAIR the analysis. Compare arms within a block, where they are
%       seconds apart, instead of across a session where they are minutes
%       apart.
%     - SYMMETRIC GAPS. The same settle time before every measurement, so no
%       arm systematically benefits from a cooler starting point.
%     - RECORD ORDER POSITION, so a residual order effect can be tested for
%       rather than assumed absent.
%
%   OPTIONS
%     .numBlocks     alternation blocks                (default 6)
%     .durationSec   seconds per measurement           (default 60)
%     .settleSec     idle gap before each measurement  (default 20)
%     .warmupBatches warm-up batches per measurement   (default 200)
%     .mode          'duration' | 'iterations'         (default 'duration')
%     .numIterations for 'iterations' mode             (default 5000)
%     .acceleration  'none' | 'auto'                   (default 'none')
%     .seed          RNG seed for the order schedule   (default 42)

    arguments
        conditions (1,:) struct
        pool       (:,1) cell
        opts.numBlocks     (1,1) double = 6
        opts.durationSec   (1,1) double = 60
        opts.settleSec     (1,1) double = 20
        opts.warmupBatches (1,1) double = 200
        opts.mode          (1,:) char   = 'duration'
        opts.numIterations (1,1) double = 5000
        opts.acceleration  (1,:) char   = 'none'
        opts.seed          (1,1) double = 42
    end

    nCond = numel(conditions);
    for c = 1:nCond
        if ~isa(conditions(c).net, 'dlnetwork')
            error('run_paired_measurement:NotDlnetwork', ...
                  ['Condition "%s" is a %s, not a dlnetwork. Every arm must ', ...
                   'be normalised with to_dlnetwork() first, or the ', ...
                   'comparison is confounded by framework path.'], ...
                  conditions(c).name, class(conditions(c).net));
        end
    end

    rng(opts.seed, 'twister');

    rows = cell(opts.numBlocks * nCond, 1);
    k    = 0;

    fprintf('\n=== Interleaved measurement: %d blocks x %d conditions ===\n', ...
            opts.numBlocks, nCond);
    fprintf('    %d s per measurement, %d s settle, order randomised per block\n\n', ...
            opts.durationSec, opts.settleSec);

    for b = 1:opts.numBlocks
        order = randperm(nCond);
        fprintf('-- block %d/%d  order: %s\n', b, opts.numBlocks, ...
                strjoin({conditions(order).name}, ' -> '));

        for pos = 1:nCond
            c = order(pos);

            % Symmetric settle before EVERY measurement, including the first
            % in a block, so no arm inherits a systematically different
            % thermal starting point.
            pause(opts.settleSec);

            r = measure_energy_v2(conditions(c).net, pool, ...
                    mode          = opts.mode, ...
                    durationSec   = opts.durationSec, ...
                    numIterations = opts.numIterations, ...
                    warmupBatches = opts.warmupBatches, ...
                    acceleration  = opts.acceleration, ...
                    label         = conditions(c).name);

            k = k + 1;
            rows{k} = struct( ...
                'block',            b, ...
                'orderPosition',    pos, ...
                'condition',        string(conditions(c).name), ...
                'startTime',        r.startTime, ...
                'endTime',          r.endTime, ...
                'elapsedSec',       r.elapsedSec, ...
                'batchSize',        r.batchSize, ...
                'numBatches',       r.numBatches, ...
                'numInferences',    r.numInferences, ...
                'secPerInference',  r.secPerInference, ...
                'throughputPerSec', r.throughputPerSec, ...
                'overrunRatio',     r.overrunRatio, ...
                'stalled',          r.stalled, ...
                'measuredWatts',    NaN, ...
                'wattsSamples',     NaN, ...
                'wattsSD',          NaN, ...
                'energyJPerInference', NaN);
        end
        fprintf('\n');
    end

    T = struct2table([rows{:}]);
end
