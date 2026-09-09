%RUN_B1_SEEDS  Reviewer B1: independent training runs, not repeated measurement.
%  Seeds 3,4,5,6 on MNIST. Each seed retrains the baseline from scratch and
%  reprunes, so between-seed spread estimates MODEL variability. Combined with
%  the existing seed-1 and seed-2 runs this gives six independent draws.
%
%  ONE HWiNFO log for the whole script: hwinfo_b1_seeds.csv
%  Runtime approx 3 h. Unattended.

clear; clc;
ROOT   = 'D:\claude_projects\ICIEAHS 2026\research_v4';
DATA   = fullfile(ROOT, 'datasets');
OUTDIR = 'D:\claude_projects\ICIEAHS 2026\final_submission\results';
HWDIR  = 'D:\claude_projects\ICIEAHS 2026\final_submission\hwinfo_results';
addpath(fullfile(ROOT,'code_v1')); addpath(fullfile(ROOT,'code_v1','v2'));
IDLE_W = 4.221;
HW = fullfile(HWDIR,'hwinfo_b1_seeds.csv');

diary(fullfile(OUTDIR, sprintf('console_B1SEEDS_%s.txt', datestr(now,'yyyymmdd_HHMMSS')))); %#ok<TNOW1,DATST>
diary on
fprintf('B1 seed study started %s\n', datestr(now)); %#ok<TNOW1,DATST>

d = fullfile(DATA,'1_MNIST');
[XTr,YTr] = load_idx_dataset(fullfile(d,'train-images.idx3-ubyte'), ...
                             fullfile(d,'train-labels.idx1-ubyte'));
[XTe,YTe] = load_idx_dataset(fullfile(d,'t10k-images.idx3-ubyte'), ...
                             fullfile(d,'t10k-labels.idx1-ubyte'));
s = round(0.9*size(XTr,4));
XVa = XTr(:,:,:,s+1:end); YVa = YTr(s+1:end);
XTr = XTr(:,:,:,1:s);     YTr = YTr(1:s);

for sd = [3 4 5 6]
    fprintf('\n########## SEED %d ##########\n', sd);
    run_experiment_v3(sprintf('MNIST-seed%d', sd), XTr,YTr, XVa,YVa, XTe,YTe, ...
        trainFcn = @train_baseline, dataFormat = 'SSCB', batchSize = 128, ...
        numBlocks = 3, settleSec = 60, seed = sd, ...
        hwinfoCsv = HW, idleWatts = IDLE_W, outDir = OUTDIR);
end

fprintf('B1 seed study finished %s\n', datestr(now)); %#ok<TNOW1,DATST>
diary off
