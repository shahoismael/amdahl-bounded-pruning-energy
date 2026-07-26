%% MAKE_FIGURE1_PIPELINE - Vertical flowchart of the experimental pipeline

clear; clc; close all;

steps = {'1. Dataset Loading', '2. Baseline Training', ...
         sprintf('3. Taylor Pruning\n(2 iterations, 4 filters/iteration)'), ...
         '4. INT8 Quantization', ...
         sprintf('5. Hardware-Timed Inference\n(30 s sustained run, HWiNFO power log)'), ...
         sprintf('6. Compare Theoretical vs.\nMeasured Energy Change')};

n = numel(steps);
boxW = 5.2; boxH = 0.95; gapY = 0.35;
figHeight = n*boxH + (n-1)*gapY + 0.6;

f = figure('Units','inches','Position',[1 1 6 figHeight], 'Color','w');
ax = axes('Position',[0 0 1 1]);
axis(ax, [0 boxW+1 0 figHeight]);
axis(ax, 'off'); hold(ax, 'on');

y = figHeight - 0.4 - boxH;
centersY = zeros(1,n);
for i = 1:n
    rectangle('Position',[0.4, y, boxW, boxH], 'Curvature', 0.15, ...
        'FaceColor', [0.933 0.945 0.961], 'EdgeColor','k', 'LineWidth',1.1);
    text(0.4+boxW/2, y+boxH/2, steps{i}, 'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', 'FontName','Times New Roman', 'FontSize',10.5);
    centersY(i) = y;
    y = y - (boxH+gapY);
end

for i = 1:n-1
    y0 = centersY(i);
    y1 = centersY(i+1) + boxH;
    annotation('arrow', [0.4+boxW/2, 0.4+boxW/2]/(boxW+1), [y0 y1]/figHeight, ...
        'Color','k','LineWidth',1.2, 'HeadWidth',8, 'HeadLength',8);
end

exportgraphics(f, fullfile('..','results','Figure1_pipeline.png'), 'Resolution', 350);
fprintf('Saved Figure1_pipeline.png\n');
