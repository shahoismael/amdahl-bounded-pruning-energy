%RUN_SEEDREP_V3  Seed replication, MNIST. Three independent draws.
clear; clc;

ROOT   = 'D:\claude_projects\ICIEAHS 2026\research_v4';
DATA   = fullfile(ROOT, 'datasets');
OUTDIR = 'D:\claude_projects\ICIEAHS 2026\final_submission\results';
HWDIR  = 'D:\claude_projects\ICIEAHS 2026\final_submission\hwinfo_results';

addpath(fullfile(ROOT, 'code_v1'));
addpath(fullfile(ROOT, 'code_v1', 'v2'));

IDLE_W = 4.221;

diary(fullfile(OUTDIR, sprintf('console_SEEDREP_%s.txt', datestr(now,'yyyymmdd_HHMMSS'))));
diary on
fprintf('Seed replication v3 started %s\n', datestr(now));

d = fullfile(DATA, '1_MNIST');
[XTr,YTr] = load_idx_dataset(fullfile(d,'train-images.idx3-ubyte'), ...
                             fullfile(d,'train-labels.idx1-ubyte'));
[XTe,YTe] = load_idx_dataset(fullfile(d,'t10k-images.idx3-ubyte'), ...
                             fullfile(d,'t10k-labels.idx1-ubyte'));
s = round(0.9*size(XTr,4));
XVa = XTr(:,:,:,s+1:end); YVa = YTr(s+1:end);
XTr = XTr(:,:,:,1:s);     YTr = YTr(1:s);

fprintf('\n########## SEED 2 ##########\n');
run_experiment_v3('MNIST-seed2', XTr,YTr, XVa,YVa, XTe,YTe, ...
    trainFcn = @train_baseline, dataFormat = 'SSCB', batchSize = 128, ...
    numBlocks = 6, settleSec = 60, seed = 2, ...
    hwinfoCsv = fullfile(HWDIR,'hwinfo_v3_seed2.csv'), ...
    idleWatts = IDLE_W, outDir = OUTDIR);

fprintf('Seed replication v3 finished %s\n', datestr(now));
diary off