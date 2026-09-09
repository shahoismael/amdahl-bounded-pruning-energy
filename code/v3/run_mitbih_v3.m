%RUN_MITBIH_V3  MIT-BIH only. Stop HWiNFO logging when this finishes.
clear; clc;

ROOT   = 'D:\claude_projects\ICIEAHS 2026\research_v4';
DATA   = fullfile(ROOT, 'datasets');
OUTDIR = 'D:\claude_projects\ICIEAHS 2026\final_submission\results';
HWDIR  = 'D:\claude_projects\ICIEAHS 2026\final_submission\hwinfo_results';

addpath(fullfile(ROOT, 'code_v1'));
addpath(fullfile(ROOT, 'code_v1', 'v2'));
if ~exist(OUTDIR, 'dir'), mkdir(OUTDIR); end

IDLE_W = 4.221;

diary(fullfile(OUTDIR, sprintf('console_MITBIH_%s.txt', datestr(now,'yyyymmdd_HHMMSS')))); %#ok<TNOW1,DATST>
diary on
fprintf('MIT-BIH v3 started %s\n', datestr(now)); %#ok<TNOW1,DATST>

d = fullfile(DATA, '4_MITBIH', 'mit-bih-arrhythmia-database-1.0.0');
[XTr,YTr,XVa,YVa,XTe,YTe,infoMit] = load_mitbih_dataset(d, splitMode = "record");

T_mitbih = run_experiment_v3('MIT-BIH', XTr,YTr, XVa,YVa, XTe,YTe, ...
    trainFcn = @train_baseline_1d, dataFormat = 'SSCB', batchSize = 128, ...
    numBlocks = 8, settleSec = 60, ...
    hwinfoCsv = fullfile(HWDIR,'hwinfo_v3_mitbih_run4.csv'), ...
    idleWatts = IDLE_W, outDir = OUTDIR);

save(fullfile(OUTDIR,'info_mitbih.mat'), 'infoMit');
fprintf('MIT-BIH v3 finished %s\n', datestr(now)); %#ok<TNOW1,DATST>
diary off
