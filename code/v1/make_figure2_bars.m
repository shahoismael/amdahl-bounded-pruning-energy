%% MAKE_FIGURE2_BARS - Theoretical vs measured energy change, grouped bar chart

clear; clc; close all;

datasets = {'MNIST','Fashion-MNIST','MIT-BIH','ChestX-ray14'};
theoretical = [20.5 20.5 20.5 20.5];
means = [9.48 -3.83 -10.08 0.06];
sds   = [8.93 19.49 21.38 6.14];

f = figure('Units','inches','Position',[1 1 6 4], 'Color','w');
ax = axes(f); hold(ax,'on');

x = 1:4;
w = 0.32;
b1 = bar(ax, x-w/2, theoretical, w, 'FaceColor',[0.298 0.447 0.690], 'EdgeColor','k','LineWidth',0.6);
b2 = bar(ax, x+w/2, means, w, 'FaceColor',[0.769 0.306 0.322], 'EdgeColor','k','LineWidth',0.6);
errorbar(ax, x+w/2, means, sds, 'k', 'LineStyle','none', 'LineWidth',1, 'CapSize',4);

yline(ax, 0, 'k-', 'LineWidth',0.8);
set(ax, 'XTick', x, 'XTickLabel', datasets, 'FontName','Times New Roman', 'FontSize',9);
ylabel(ax, 'Change relative to baseline (%)', 'FontName','Times New Roman', 'FontSize',10);
ylim(ax, [-45 45]);
box(ax,'off');
legend(ax, [b1 b2], {'Theoretical filter reduction (fixed)','Measured energy change (mean \pm SD)'}, ...
    'FontName','Times New Roman','FontSize',8, 'Box','off', 'Location','northeast');

exportgraphics(f, fullfile('..','results','Figure2_theoretical_vs_measured.png'), 'Resolution', 350);
fprintf('Saved Figure2_theoretical_vs_measured.png\n');
