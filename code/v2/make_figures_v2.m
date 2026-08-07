function make_figures_v2(opts)
%MAKE_FIGURES_V2 Regenerate every manuscript figure from v2 data.
%
%   make_figures_v2()
%   make_figures_v2(outDir='../../figs_v2')
%
%   Produces four figures, each carrying one claim:
%
%     Figure 1  theoretical vs measured saving, with the shortfall visible
%     Figure 2  batch-size sweep -- where the effect exists and where it does not
%     Figure 3  mean power by arm -- flat, so the saving is time, not watts
%     Figure 4  v1 forensics -- clean vs throttled runs
%
%   The v1 figure scripts (make_figure2_bars.m, make_figure3_boxplot.m) plot
%   superseded numbers and should not be used for the revised manuscript.
%
%   All panels are vector-safe, greyscale-legible, and sized for a single
%   journal column at 300 dpi.

    arguments
        opts.measurementsCsv (1,:) char = ''
        opts.sweepCsv        (1,:) char = '../../results/batch_sweep.csv'
        opts.recoveredCsv    (1,:) char = '../../results/recovered_power.csv'
        opts.theoretical     (1,1) double = 22.26
        opts.outDir          (1,:) char = '../../figs_v2'
        opts.dpi             (1,1) double = 300
    end

    if ~exist(opts.outDir,'dir'), mkdir(opts.outDir); end

    % Locate the newest v2 measurement file unless one was named.
    if isempty(opts.measurementsCsv)
        d = dir(fullfile('../../results_v2','measurements_*.csv'));
        if isempty(d)
            error('make_figures_v2:NoMeasurements', ...
                  'No measurements_*.csv in results_v2. Run run_experiment_v2 first.');
        end
        [~,k] = max([d.datenum]);
        opts.measurementsCsv = fullfile(d(k).folder, d(k).name);
    end
    fprintf('measurements : %s\n', opts.measurementsCsv);

    C = struct('base',[0.35 0.35 0.35], 'pruned',[0.10 0.10 0.10], ...
               'accent',[0.55 0.55 0.55], 'bad',[0.75 0.75 0.75]);

    T = readtable(opts.measurementsCsv);
    T.condition = string(T.condition);

    % =====================================================================
    % FIGURE 1  theoretical vs measured
    % =====================================================================
    ref = "baseline_ft";
    if ~any(T.condition == ref), ref = "baseline"; end

    blocks = unique(T.block);
    sav = nan(numel(blocks),1);
    for i = 1:numel(blocks)
        r = T(T.block==blocks(i) & T.condition==ref, :);
        p = T(T.block==blocks(i) & T.condition=="pruned", :);
        if isempty(r) || isempty(p), continue; end
        sav(i) = (r.energyJPerInference(1) - p.energyJPerInference(1)) ...
                 / r.energyJPerInference(1) * 100;
    end
    sav = sav(~isnan(sav));

    f1 = figure('Color','w','Units','centimeters','Position',[2 2 9 8]);
    ax = axes(f1); hold(ax,'on');
    bar(ax, 1, opts.theoretical, 0.6, 'FaceColor', C.accent, 'EdgeColor','k');
    bar(ax, 2, mean(sav),        0.6, 'FaceColor', C.pruned, 'EdgeColor','k');
    errorbar(ax, 2, mean(sav), std(sav)/sqrt(numel(sav))*tinv95(numel(sav)-1), ...
             'k', 'LineStyle','none', 'LineWidth',1.1, 'CapSize',10);
    scatter(ax, 2 + (rand(numel(sav),1)-0.5)*0.22, sav, 16, ...
            'MarkerFaceColor','w','MarkerEdgeColor','k','LineWidth',0.6);
    yline(ax, 0, 'k-');
    text(ax, 1.5, opts.theoretical*0.62, sprintf('%.0f%% of\ntheory', ...
         mean(sav)/opts.theoretical*100), 'HorizontalAlignment','center','FontSize',9);
    set(ax,'XTick',[1 2],'XTickLabel',{'Predicted','Measured'},'FontSize',9, ...
        'Box','off','TickDir','out','XLim',[0.4 2.6]);
    ylabel(ax,'Energy reduction per inference (%)','FontSize',9);
    title(ax,'Theoretical prediction vs hardware measurement','FontSize',9.5,'FontWeight','normal');
    export_fig_local(f1, fullfile(opts.outDir,'Figure1_theory_vs_measured'), opts.dpi);

    % =====================================================================
    % FIGURE 2  batch-size sweep
    % =====================================================================
    if isfile(opts.sweepCsv)
        S = readtable(opts.sweepCsv);
        f2 = figure('Color','w','Units','centimeters','Position',[2 2 9 8]);
        ax = axes(f2); hold(ax,'on');
        yline(ax, opts.theoretical, '--', 'Predicted', 'Color', C.accent, ...
              'LineWidth',1.2,'FontSize',8,'LabelHorizontalAlignment','left');
        yline(ax, 0, 'k-');
        plot(ax, S.batchSize, -S.latencyChangePct, '-o', 'Color', C.pruned, ...
             'MarkerFaceColor','w','LineWidth',1.3,'MarkerSize',5);
        set(ax,'XScale','log','XTick',S.batchSize,'XTickLabel',string(S.batchSize), ...
            'FontSize',9,'Box','off','TickDir','out');
        xlabel(ax,'Inference batch size','FontSize',9);
        ylabel(ax,'Measured reduction (%)','FontSize',9);
        title(ax,'The saving exists only above the overhead floor','FontSize',9.5,'FontWeight','normal');
        export_fig_local(f2, fullfile(opts.outDir,'Figure2_batch_sweep'), opts.dpi);
    else
        fprintf('skipping Figure 2: %s not found\n', opts.sweepCsv);
    end

    % =====================================================================
    % FIGURE 3  mean power by arm
    % =====================================================================
    if ismember('measuredWatts', T.Properties.VariableNames) && any(~isnan(T.measuredWatts))
        arms = unique(T.condition,'stable');
        mu = zeros(numel(arms),1); sd = zeros(numel(arms),1);
        f3 = figure('Color','w','Units','centimeters','Position',[2 2 9 8]);
        ax = axes(f3); hold(ax,'on');
        for i = 1:numel(arms)
            w = T.measuredWatts(T.condition==arms(i));
            mu(i) = mean(w,'omitnan'); sd(i) = std(w,'omitnan');
            bar(ax, i, mu(i), 0.6, 'FaceColor', C.base, 'EdgeColor','k');
            errorbar(ax, i, mu(i), sd(i), 'k','LineStyle','none','LineWidth',1.1,'CapSize',10);
            scatter(ax, i + (rand(numel(w),1)-0.5)*0.22, w, 14, ...
                    'MarkerFaceColor','w','MarkerEdgeColor','k','LineWidth',0.5);
        end
        spread = (max(mu)-min(mu))/mean(mu)*100;
        set(ax,'XTick',1:numel(arms),'XTickLabel',strrep(arms,'_','\_'), ...
            'FontSize',9,'Box','off','TickDir','out','XLim',[0.4 numel(arms)+0.6]);
        ylabel(ax,'Mean CPU package power (W)','FontSize',9);
        title(ax, sprintf('Power is unchanged (spread %.1f%%)', spread), ...
              'FontSize',9.5,'FontWeight','normal');
        export_fig_local(f3, fullfile(opts.outDir,'Figure3_power_flat'), opts.dpi);
    else
        fprintf('skipping Figure 3: no power column\n');
    end

    % =====================================================================
    % FIGURE 4  v1 forensics, clean vs throttled
    % =====================================================================
    if isfile(opts.recoveredCsv)
        R = readtable(opts.recoveredCsv);
        if ismember('latencyChangePct', R.Properties.VariableNames)
            f4 = figure('Color','w','Units','centimeters','Position',[2 2 9 8]);
            ax = axes(f4); hold(ax,'on');
            yline(ax,0,'k-'); xline(ax,0,'k-');
            scatter(ax, R.latencyChangePct, R.energySavingPct, 34, ...
                    'MarkerFaceColor',C.pruned,'MarkerEdgeColor','k','MarkerFaceAlpha',0.65);
            x = R.latencyChangePct(:); y = R.energySavingPct(:);
            ok = ~isnan(x) & ~isnan(y);
            pfit = polyfit(x(ok),y(ok),1);
            xs = linspace(min(x(ok)),max(x(ok)),50);
            plot(ax, xs, polyval(pfit,xs), '--','Color',C.accent,'LineWidth',1.2);
            rr = corr_local(x(ok),y(ok));
            text(ax, 0.04, 0.94, sprintf('r = %+.2f', rr), 'Units','normalized', ...
                 'FontSize',9,'VerticalAlignment','top');
            set(ax,'FontSize',9,'Box','off','TickDir','out');
            xlabel(ax,'Measured slowdown of the run (%)','FontSize',9);
            ylabel(ax,'Apparent energy saving (%)','FontSize',9);
            title(ax,'v1 savings track CPU throttling, not pruning','FontSize',9.5,'FontWeight','normal');
            export_fig_local(f4, fullfile(opts.outDir,'Figure4_v1_artefact'), opts.dpi);
        end
    else
        fprintf('skipping Figure 4: %s not found\n', opts.recoveredCsv);
    end

    fprintf('\nFigures written to %s\n', opts.outDir);
end

% -------------------------------------------------------------------------
function export_fig_local(f, base, dpi)
%EXPORT_FIG_LOCAL Save PNG for review and PDF (vector) for submission.
    exportgraphics(f, [base '.png'], 'Resolution', dpi);
    exportgraphics(f, [base '.pdf'], 'ContentType','vector');
    fprintf('  wrote %s.{png,pdf}\n', base);
end

% -------------------------------------------------------------------------
function t = tinv95(df)
%TINV95 Two-sided 95% t critical value, toolbox-free.
    lo = 0; hi = 50;
    for i = 1:200
        mid = (lo+hi)/2;
        x = df/(df+mid^2);
        p = 1 - 0.5*betainc(x, df/2, 0.5);
        if p < 0.975, lo = mid; else, hi = mid; end
    end
    t = (lo+hi)/2;
end

% -------------------------------------------------------------------------
function r = corr_local(x,y)
    x = x(:); y = y(:);
    r = sum((x-mean(x)).*(y-mean(y))) / sqrt(sum((x-mean(x)).^2)*sum((y-mean(y)).^2));
end
