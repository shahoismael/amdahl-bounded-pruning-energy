function T = attach_power_to_table(T, csvPath, opts)
%ATTACH_POWER_TO_TABLE Fill measured power and energy for every measurement row.
%
%   T = ATTACH_POWER_TO_TABLE(T, csvPath)
%
%   Replaces the manual "open the CSV and average by eye" step. Every row of
%   the measurement table gets its power window extracted programmatically
%   from the same log, and the per-inference energy computed from it, so the
%   saved artefact contains the numbers the paper reports.
%
%   Energy per inference:
%
%       E = (P_avg * T_elapsed) / N_inferences        [joules]
%
%   Note this is TOTAL package draw during the window, including the idle
%   floor. Reporting it this way is defensible (it reflects realistic
%   deployment) but it systematically dilutes any compression effect, since
%   the idle floor is a constant that pruning cannot touch. Both figures are
%   therefore computed: energyJPerInference (total) and
%   energyJPerInferenceNetIdle (idle-subtracted). Report both -- the gap
%   between them is itself informative, and a reviewer will ask.

    arguments
        T table
        csvPath (1,:) char
        opts.idleWatts    (1,1) double = NaN
        opts.powerPattern (1,:) char = 'CPU Package Power'
        opts.minSamples   (1,1) double = 20
    end

    n = height(T);
    [w, wsd, wn, tmean, tmax, cmean] = deal(NaN(n,1));

    for i = 1:n
        s = parse_hwinfo_log(csvPath, T.startTime(i), T.endTime(i), ...
                powerPattern = opts.powerPattern, minSamples = opts.minSamples);
        w(i)   = s.power.mean;
        wsd(i) = s.power.sd;
        wn(i)  = s.power.n;
        tmean(i) = s.temp.mean;
        tmax(i)  = s.temp.max;
        cmean(i) = s.clock.mean;
    end

    T.measuredWatts = w;
    T.wattsSD       = wsd;
    T.wattsSamples  = wn;
    T.tempMeanC     = tmean;
    T.tempMaxC      = tmax;
    T.clockMeanMHz  = cmean;

    T.energyJPerInference = (T.measuredWatts .* T.elapsedSec) ./ T.numInferences;

    if ~isnan(opts.idleWatts)
        netW = max(T.measuredWatts - opts.idleWatts, 0);
        T.energyJPerInferenceNetIdle = (netW .* T.elapsedSec) ./ T.numInferences;
    else
        T.energyJPerInferenceNetIdle = NaN(n,1);
    end
end
