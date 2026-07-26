%% MAKE_FIGURE3_BOXPLOT - Run-to-run variability box plot with jittered points
%% (Manual construction - no Statistics and Machine Learning Toolbox required)

clear; clc; close all;

datasets = {'MNIST','Fashion-MNIST','MIT-BIH','ChestX-ray14'};
data = { [5.7 12.8 -3.9 19.5 13.3], ...
         [-1.6 0.05 -29.0 24.2 -12.8], ...
         [2.1 7.1 -40.7 5.5 -24.4], ...
         [5.4 -3.1 5.6 -8.9 1.3] };

colors = [0.506 0.447 0.698; 0.800 0.725 0.455; 0.392 0.710 0.803; 0.333 0.659 0.408];

f = figure('Units','inches','Position',[1 1 6 4], 'Color','w');
ax = axes(f); hold(ax,'on');

boxWidth = 0.35;
for i = 1:numel(data)
    yv = sort(data{i});
    q1 = median(yv(yv <= median(yv)));       % simple quartile approx for n=5
    q3 = median(yv(yv >= median(yv)));
    med = median(yv);
    mn = min(yv);
    mx = max(yv);
    meanVal = mean(yv);

    % Box (Q1 to Q3)
    rectangle(ax, 'Position', [i-boxWidth/2, q1, boxWidth, q3-q1], ...
        'FaceColor', colors(i,:), 'FaceAlpha', 0.6, 'EdgeColor','k', 'LineWidth', 0.8);

    % Median line
    plot(ax, [i-boxWidth/2, i+boxWidth/2], [med med], 'k-', 'LineWidth', 1.2);

    % Whiskers
    plot(ax, [i i], [q3 mx], 'k-', 'LineWidth', 0.8);
    plot(ax, [i i], [mn q1], 'k-', 'LineWidth', 0.8);
    plot(ax, [i-0.08 i+0.08], [mx mx], 'k-', 'LineWidth', 0.8);
    plot(ax, [i-0.08 i+0.08], [mn mn], 'k-', 'LineWidth', 0.8);

    % Mean marker (white diamond)
    plot(ax, i, meanVal, 'd', 'MarkerFaceColor','w', 'MarkerEdgeColor','k', 'MarkerSize',6, 'LineWidth',0.8);

    % Individual jittered points
    xv = i + (rand(size(yv))-0.5)*0.15;
    scatter(ax, xv, yv, 20, 'k', 'filled', 'MarkerFaceAlpha', 0.7);
end

yline(ax, 0, 'k--', 'LineWidth', 0.7);
set(ax, 'XTick', 1:4, 'XTickLabel', datasets, 'FontName','Times New Roman', 'FontSize',9);
xlim(ax, [0.5 4.5]);
xlabel(ax, 'Dataset', 'FontName','Times New Roman', 'FontSize',10);
ylabel(ax, 'Measured energy change vs. baseline (%)', 'FontName','Times New Roman', 'FontSize',10);
box(ax,'off');

exportgraphics(f, fullfile('..','results','Figure3_variability_boxplot.png'), 'Resolution', 350);
fprintf('Saved Figure3_variability_boxplot.png\n');
