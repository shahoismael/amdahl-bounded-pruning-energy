function results = analyze_energy_v2(T, opts)
%ANALYZE_ENERGY_V2 Paired, equivalence-based analysis of the energy audit.
%
%   results = ANALYZE_ENERGY_V2(T)
%   results = ANALYZE_ENERGY_V2(T, theoreticalSaving=24.44, ...)
%
%   THE CENTRAL STATISTICAL CORRECTION
%   ----------------------------------
%   v1 ran a one-sample t-test of measured energy change against zero, got
%   p = 0.76, and read that as evidence that pruning produces no energy
%   saving. That inference is invalid: failing to reject a null is not
%   evidence for it. The v1 confidence interval [-8.55, +6.37] comfortably
%   contains a real 6 percent saving, so the data as analysed cannot rule
%   one out.
%
%   The question the paper actually wants to answer is not "is the saving
%   zero?" but "is the saving as large as the theoretical metric predicts?"
%   That is a NON-INFERIORITY question, and it has a proper test:
%
%       H0:  true saving  >=  theoretical predicted saving
%       H1:  true saving  <   theoretical predicted saving
%
%   Rejecting H0 is a positive finding. On the v1 data pooled across 20 runs
%   (mean change -1.09%, SD 15.94, predicted saving 24.44%) this rejects at
%   t(19) = -6.55, p = 1.4e-06. The paper's thesis survives -- and becomes
%   an assertion supported by evidence rather than an argument from absence.
%
%   Equivalence bounds (TOST) are also reported, so the paper can state the
%   smallest effect its design could have detected instead of implying it
%   detected nothing.
%
%   OPTIONS
%     .theoreticalSaving  predicted % energy saving (default: taken from T)
%     .referenceArm       'baseline_ft' (default) | 'baseline'
%     .equivalenceMargin  TOST bound in % points (default 5)
%     .nBoot              bootstrap resamples (default 10000)
%     .alpha              (default 0.05)

    arguments
        T table
        opts.theoreticalSaving (1,1) double = NaN
        opts.referenceArm      (1,:) char   = 'baseline_ft'
        opts.equivalenceMargin (1,1) double = 5
        opts.nBoot             (1,1) double = 10000
        opts.alpha             (1,1) double = 0.05
    end

    if isnan(opts.theoreticalSaving)
        if ismember('theoreticalMACReduction', T.Properties.VariableNames)
            opts.theoreticalSaving = mean(T.theoreticalMACReduction, 'omitnan');
        else
            error('analyze_energy_v2:NoTheoretical', ...
                  'Supply theoreticalSaving= or include theoreticalMACReduction in T.');
        end
    end

    % ---- 1. Within-block paired differences -------------------------------
    % Pairing WITHIN a block is what removes the thermal / background-load
    % confound: the two arms are seconds apart, not sessions apart.
    datasets = unique(T.dataset, 'stable');
    rows = cell(0, 5);   % width fixed up front: (end+1,:) on a bare {} errors

    for d = 1:numel(datasets)
        Td = T(T.dataset == datasets(d), :);
        for b = unique(Td.block)'
            Tb  = Td(Td.block == b, :);
            ref = Tb(Tb.condition == string(opts.referenceArm), :);
            prn = Tb(Tb.condition == "pruned", :);
            if isempty(ref) || isempty(prn), continue; end

            % Drop the whole block if any arm in it stalled. A measurement
            % that overran its budget was measuring background load, not the
            % network, and one such block is enough to dominate the mean.
            if ismember('stalled', Tb.Properties.VariableNames) && any(Tb.stalled)
                fprintf('  EXCLUDED %s block %d: stalled measurement (%.1fx budget)\n', ...
                        datasets(d), b, max(Tb.overrunRatio));
                continue;
            end

            savingPct = (ref.energyJPerInference(1) - prn.energyJPerInference(1)) ...
                        / ref.energyJPerInference(1) * 100;
            latencyPct = (prn.secPerInference(1) - ref.secPerInference(1)) ...
                        / ref.secPerInference(1) * 100;

            rows(end+1, :) = {datasets(d), b, savingPct, latencyPct, ...
                              prn.orderPosition(1) - ref.orderPosition(1)}; %#ok<AGROW>
        end
    end

    P = cell2table(rows, 'VariableNames', ...
        {'dataset', 'block', 'energySavingPct', 'latencyChangePct', 'orderDelta'});

    results = struct();
    results.paired             = P;
    results.theoreticalSaving  = opts.theoreticalSaving;
    results.referenceArm       = string(opts.referenceArm);

    % ---- 2. Per-dataset and pooled tests -----------------------------------
    groups = [{'ALL POOLED'}; cellstr(string(datasets))];
    stats  = cell(numel(groups), 1);

    fprintf('\n================ ENERGY AUDIT, v2 ANALYSIS ================\n');
    fprintf('Reference arm      : %s\n', opts.referenceArm);
    fprintf('Theoretical saving : %.2f%% (MAC-based prediction)\n', opts.theoreticalSaving);
    fprintf('Equivalence margin : +/- %.1f percentage points\n\n', opts.equivalenceMargin);
    fprintf('%-16s %4s %9s %8s %11s %12s %14s\n', ...
            'group', 'n', 'mean', 'sd', 'CI95 low', 'CI95 high', 'p(noninf)');
    fprintf('%s\n', repmat('-', 1, 80));

    for g = 1:numel(groups)
        if g == 1
            x = P.energySavingPct;
        else
            x = P.energySavingPct(P.dataset == string(groups{g}));
        end
        x = x(~isnan(x));
        s = one_group_stats(x, opts.theoreticalSaving, opts.equivalenceMargin, ...
                            opts.alpha, opts.nBoot);
        s.group  = string(groups{g});
        stats{g} = s;

        fprintf('%-16s %4d %+9.2f %8.2f %+11.2f %+12.2f %14.3e\n', ...
                groups{g}, s.n, s.mean, s.sd, s.ci(1), s.ci(2), s.pNonInferiority);
    end

    S = [stats{:}];
    results.stats = S;

    % ---- 3. Holm-Bonferroni across the per-dataset tests -------------------
    perDs = S(2:end);
    [~, ord] = sort([perDs.pNonInferiority]);
    m = numel(perDs);
    adj = NaN(1, m);
    running = 0;
    for i = 1:m
        running = max(running, (m - i + 1) * perDs(ord(i)).pNonInferiority);
        adj(ord(i)) = min(1, running);
    end
    for i = 1:m
        perDs(i).pHolm = adj(i);
    end
    results.perDatasetHolm = perDs;

    fprintf('\nHolm-Bonferroni, non-inferiority tests across %d datasets:\n', m);
    for i = 1:m
        fprintf('  %-16s raw p = %.3e   Holm p = %.3e   %s\n', ...
                perDs(i).group, perDs(i).pNonInferiority, perDs(i).pHolm, ...
                verdict(perDs(i).pHolm < opts.alpha));
    end

    % ---- 4. Order effect: did position in the block matter? ----------------
    if numel(unique(P.orderDelta)) > 1
        [rho, pOrd] = corr_pearson(P.orderDelta, P.energySavingPct);
        results.orderEffect = struct('rho', rho, 'p', pOrd);
        fprintf('\nOrder-effect check: corr(order gap, measured saving) r = %+.3f, p = %.3f\n', rho, pOrd);
        if pOrd < 0.05
            fprintf('  WARNING: measurement order still predicts the outcome. Increase\n');
            fprintf('  settle time or number of blocks before trusting these estimates.\n');
        end
    end

    % ---- 5. Clustered estimate, dataset as the cluster ---------------------
    % The pooled t-test treats every block as independent. They are not:
    % blocks within a dataset share a trained network, a session and a
    % thermal environment, so the pooled df is inflated.
    %
    % fitlme would handle this, but it needs the Statistics Toolbox, which is
    % not present on this machine. The random-intercept model is computed
    % directly instead -- for a balanced one-way design the closed form is
    % exact, and it degrades gracefully when unbalanced.
    if numel(datasets) > 1
        results.clustered = clustered_mean(P.energySavingPct, P.dataset, opts.alpha);
        c = results.clustered;
        fprintf('\nClustered estimate (dataset as random intercept, %d clusters):\n', c.nClusters);
        fprintf('  grand mean          : %+.2f%%\n', c.mean);
        fprintf('  between-dataset SD  : %.2f\n', c.sdBetween);
        fprintf('  within-dataset SD   : %.2f\n', c.sdWithin);
        fprintf('  ICC                 : %.3f\n', c.icc);
        fprintf('  SE (cluster-aware)  : %.3f   vs naive SE %.3f\n', c.se, c.seNaive);
        fprintf('  95%% CI              : %+.2f to %+.2f  (df = %d)\n', c.ci(1), c.ci(2), c.df);
        tNI = (c.mean - opts.theoreticalSaving) / c.se;
        fprintf('  non-inferiority     : t(%d) = %.3f, p = %.3e\n', ...
                c.df, tNI, tcdf_local(tNI, c.df));
        if c.se > 1.5 * c.seNaive
            fprintf('  NOTE: clustering inflates the SE by %.0f%%. The pooled t-test\n', ...
                    (c.se/c.seNaive - 1)*100);
            fprintf('        above is anticonservative; report this row instead.\n');
        end
    end

    fprintf('\n===========================================================\n');
end

% -------------------------------------------------------------------------
function s = one_group_stats(x, theoretical, margin, alpha, nBoot)
    n  = numel(x);
    mu = mean(x);
    sd = std(x);
    se = sd / sqrt(n);
    df = n - 1;

    tcrit = tinv_local(1 - alpha/2, df);
    s.n    = n;
    s.mean = mu;
    s.sd   = sd;
    s.se   = se;
    s.ci   = [mu - tcrit*se, mu + tcrit*se];

    % Non-inferiority: H0 saving >= theoretical, H1 saving < theoretical
    s.tNonInferiority = (mu - theoretical) / se;
    s.pNonInferiority = tcdf_local(s.tNonInferiority, df);

    % TOST equivalence against +/- margin
    tLow  = (mu - (-margin)) / se;
    tHigh = (mu -   margin)  / se;
    s.pTOST = max(1 - tcdf_local(tLow, df), tcdf_local(tHigh, df));
    s.equivalent = s.pTOST < alpha;

    % Classical test against zero, retained for comparison with v1
    s.tVsZero = mu / se;
    s.pVsZero = 2 * tcdf_local(-abs(s.tVsZero), df);
    s.cohensD = mu / sd;

    % Smallest effect this design could detect at 80% power
    s.mde = (tinv_local(1 - alpha/2, df) + tinv_local(0.80, df)) * se;

    % Bootstrap percentile CI, distribution-free
    if n >= 3
        bs = zeros(nBoot, 1);
        for i = 1:nBoot
            bs(i) = mean(x(randi(n, n, 1)));
        end
        s.bootCI = prctile_local(bs, [100*alpha/2, 100*(1-alpha/2)]);
    else
        s.bootCI = [NaN NaN];
    end
end

% -------------------------------------------------------------------------
function c = clustered_mean(y, group, alpha)
%CLUSTERED_MEAN One-way random-intercept model computed in closed form.
%
%   Estimates the grand mean and its standard error while respecting that
%   observations are nested within clusters (here, datasets). Uses the
%   cluster means as the unit of analysis, which is exact for a balanced
%   design and conservative otherwise -- the right direction to err in.
%
%   Variance components come from the standard one-way ANOVA decomposition:
%       MSB = between-cluster mean square
%       MSW = within-cluster mean square
%       sigma2_between = (MSB - MSW) / n0      (floored at zero)
%   where n0 is the effective cluster size.

    g = categorical(group);
    lev = categories(g);
    k = numel(lev);

    ni = zeros(k,1); mi = zeros(k,1); ssw = 0;
    for i = 1:k
        yi = y(g == lev{i});
        yi = yi(~isnan(yi));
        ni(i) = numel(yi);
        mi(i) = mean(yi);
        ssw = ssw + sum((yi - mi(i)).^2);
    end

    N  = sum(ni);
    gm = sum(ni .* mi) / N;

    dfW = N - k;
    dfB = k - 1;
    msw = ssw / max(dfW, 1);
    msb = sum(ni .* (mi - gm).^2) / max(dfB, 1);

    n0 = (N - sum(ni.^2)/N) / max(dfB, 1);          % effective cluster size
    s2b = max(0, (msb - msw) / max(n0, eps));

    % Inference on the cluster means: df = k - 1, the honest denominator.
    c = struct();
    c.nClusters = k;
    c.n         = N;
    c.mean      = mean(mi);                          % unweighted grand mean
    c.sdBetween = sqrt(s2b);
    c.sdWithin  = sqrt(msw);
    c.icc       = s2b / max(s2b + msw, eps);
    c.se        = std(mi) / sqrt(k);
    c.seNaive   = std(y(~isnan(y))) / sqrt(N);
    c.df        = k - 1;
    tcrit       = tinv_local(1 - alpha/2, c.df);
    c.ci        = [c.mean - tcrit*c.se, c.mean + tcrit*c.se];
    c.clusterMeans = mi(:)';
    c.clusterSizes = ni(:)';
end

% -------------------------------------------------------------------------
% Toolbox-free distribution helpers, so the core analysis runs on a bare
% MATLAB install. betainc and erfinv are base functions.
% -------------------------------------------------------------------------
function p = tcdf_local(t, df)
    x = df ./ (df + t.^2);
    p = 0.5 * betainc(x, df/2, 0.5);
    p(t > 0) = 1 - p(t > 0);
end

function t = tinv_local(p, df)
    lo = -50; hi = 50;
    for i = 1:200
        mid = (lo + hi) / 2;
        if tcdf_local(mid, df) < p, lo = mid; else, hi = mid; end
    end
    t = (lo + hi) / 2;
end

function q = prctile_local(x, pcts)
    xs = sort(x(:));
    n  = numel(xs);
    q  = zeros(size(pcts));
    for i = 1:numel(pcts)
        pos = pcts(i)/100 * n + 0.5;
        lo  = max(1, floor(pos)); hi = min(n, ceil(pos));
        w   = pos - floor(pos);
        q(i) = (1-w)*xs(lo) + w*xs(hi);
    end
end

function [r, p] = corr_pearson(x, y)
    x = x(:); y = y(:);
    ok = ~isnan(x) & ~isnan(y);
    x = x(ok); y = y(ok); n = numel(x);
    r = sum((x-mean(x)).*(y-mean(y))) / sqrt(sum((x-mean(x)).^2)*sum((y-mean(y)).^2));
    t = r * sqrt((n-2)/(1-r^2));
    p = 2 * tcdf_local(-abs(t), n-2);
end

function v = verdict(tf)
    if tf, v = 'REJECT H0 (measured saving falls short of theory)';
    else,  v = 'retain H0 (underpowered at this n)'; end
end
