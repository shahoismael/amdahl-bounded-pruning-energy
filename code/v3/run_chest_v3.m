%RUN_CHEST_V3  ChestX-ray14 only. Stop HWiNFO logging when this finishes.
clear; clc;

ROOT   = 'D:\claude_projects\ICIEAHS 2026\research_v4';
DATA   = fullfile(ROOT, 'datasets');
OUTDIR = 'D:\claude_projects\ICIEAHS 2026\final_submission\results';
HWDIR  = 'D:\claude_projects\ICIEAHS 2026\final_submission\hwinfo_results';

addpath(fullfile(ROOT, 'code_v1'));
addpath(fullfile(ROOT, 'code_v1', 'v2'));
if ~exist(OUTDIR, 'dir'), mkdir(OUTDIR); end

IDLE_W = 4.221;

diary(fullfile(OUTDIR, sprintf('console_Chest_%s.txt', datestr(now,'yyyymmdd_HHMMSS')))); %#ok<TNOW1,DATST>
diary on
fprintf('ChestX-ray14 v3 started %s\n', datestr(now)); %#ok<TNOW1,DATST>

d = fullfile(DATA, '3_NIH_ChestX-ray14');
[XTr,YTr,XVa,YVa,XTe,YTe,infoChest] = load_chestxray_dataset(d, ...
    splitMode = "patient", subsetSize = 2000);

T_chest = run_experiment_v3('ChestX-ray14', XTr,YTr, XVa,YVa, XTe,YTe, ...
    trainFcn = @train_baseline, dataFormat = 'SSCB', batchSize = 64, ...
    numBlocks = 10, settleSec = 60, burnInSec = 300, ...
    hwinfoCsv = fullfile(HWDIR,'hwinfo_v3_chest_run2.csv'), ...
    idleWatts = IDLE_W, outDir = OUTDIR);

save(fullfile(OUTDIR,'info_chestxray.mat'), 'infoChest');
fprintf('ChestX-ray14 v3 finished %s\n', datestr(now)); %#ok<TNOW1,DATST>
diary off
