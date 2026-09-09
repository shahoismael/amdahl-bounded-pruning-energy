%RUN_A22_RATIOS  Reviewer A22: multiple pruning ratios.
%
%  The submitted paper reports one pruning ratio per dataset. This script
%  sweeps four ratios on all four datasets so that the effective reachable
%  fraction can be plotted against compression ratio, and so that the phrase
%  "upper bound" is either licensed or retracted on evidence.
%
%  Ratio is controlled by maxToPrune, the number of channels removed per
%  iteration, at a fixed two iterations. The baseline network has 48 filters.
%    maxToPrune 2  ->  4 of 48 removed   ~10 percent MAC
%    maxToPrune 4  ->  8 of 48 removed   ~22 percent MAC   (the submitted setting)
%    maxToPrune 7  -> 14 of 48 removed   ~35 percent MAC
%    maxToPrune 10 -> 20 of 48 removed   ~48 percent MAC
%  Realised MAC reduction is recorded per run and is not assumed.
%
%  Blocks are reduced to 3 to keep the sweep to one night. The ratio trend is
%  the quantity of interest here, not a precise per-point interval.
%
%  ONE HWiNFO log for the whole script: hwinfo_a22_ratios.csv
%  Runtime approx 9 h. Unattended. Disable sleep before starting.

clear; clc;
ROOT   = 'D:\claude_projects\ICIEAHS 2026\research_v4';
DATA   = fullfile(ROOT, 'datasets');
OUTDIR = 'D:\claude_projects\ICIEAHS 2026\final_submission\results';
HWDIR  = 'D:\claude_projects\ICIEAHS 2026\final_submission\hwinfo_results';
addpath(fullfile(ROOT,'code_v1')); addpath(fullfile(ROOT,'code_v1','v2'));
if ~exist(OUTDIR,'dir'), mkdir(OUTDIR); end

IDLE_W  = 4.221;
HW      = fullfile(HWDIR,'hwinfo_a22_ratios.csv');
RATIOS  = [2 4 7 10];
NBLOCKS = 3;

diary(fullfile(OUTDIR, sprintf('console_A22_%s.txt', datestr(now,'yyyymmdd_HHMMSS')))); %#ok<TNOW1,DATST>
diary on
fprintf('A22 ratio sweep started %s\n', datestr(now)); %#ok<TNOW1,DATST>
fprintf('ratios (maxToPrune per iteration): %s\n', mat2str(RATIOS));

%% ---- 1. MNIST -----------------------------------------------------------
d = fullfile(DATA,'1_MNIST');
[XTr,YTr] = load_idx_dataset(fullfile(d,'train-images.idx3-ubyte'), ...
                             fullfile(d,'train-labels.idx1-ubyte'));
[XTe,YTe] = load_idx_dataset(fullfile(d,'t10k-images.idx3-ubyte'), ...
                             fullfile(d,'t10k-labels.idx1-ubyte'));
s = round(0.9*size(XTr,4));
XVa = XTr(:,:,:,s+1:end); YVa = YTr(s+1:end);
XTr = XTr(:,:,:,1:s);     YTr = YTr(1:s);
for r = RATIOS
    fprintf('\n##### MNIST  maxToPrune = %d #####\n', r);
    run_experiment_v3(sprintf('MNIST-r%d', r), XTr,YTr, XVa,YVa, XTe,YTe, ...
        trainFcn=@train_baseline, dataFormat='SSCB', batchSize=128, ...
        numBlocks=NBLOCKS, settleSec=60, seed=1, maxToPrune=r, ...
        hwinfoCsv=HW, idleWatts=IDLE_W, outDir=OUTDIR);
end
clear XTr YTr XVa YVa XTe YTe d s

%% ---- 2. Fashion-MNIST ---------------------------------------------------
d = fullfile(DATA,'2_Fashion_MNIST','data','fashion');
[XTr,YTr] = load_idx_dataset(fullfile(d,'train-images-idx3-ubyte'), ...
                             fullfile(d,'train-labels-idx1-ubyte'));
[XTe,YTe] = load_idx_dataset(fullfile(d,'t10k-images-idx3-ubyte'), ...
                             fullfile(d,'t10k-labels-idx1-ubyte'));
s = round(0.9*size(XTr,4));
XVa = XTr(:,:,:,s+1:end); YVa = YTr(s+1:end);
XTr = XTr(:,:,:,1:s);     YTr = YTr(1:s);
for r = RATIOS
    fprintf('\n##### Fashion-MNIST  maxToPrune = %d #####\n', r);
    run_experiment_v3(sprintf('Fashion-r%d', r), XTr,YTr, XVa,YVa, XTe,YTe, ...
        trainFcn=@train_baseline, dataFormat='SSCB', batchSize=128, ...
        numBlocks=NBLOCKS, settleSec=60, seed=1, maxToPrune=r, ...
        hwinfoCsv=HW, idleWatts=IDLE_W, outDir=OUTDIR);
end
clear XTr YTr XVa YVa XTe YTe d s

%% ---- 3. MIT-BIH ---------------------------------------------------------
d = fullfile(DATA,'4_MITBIH','mit-bih-arrhythmia-database-1.0.0');
[XTr,YTr,XVa,YVa,XTe,YTe,infoMit] = load_mitbih_dataset(d, splitMode="record");
for r = RATIOS
    fprintf('\n##### MIT-BIH  maxToPrune = %d #####\n', r);
    run_experiment_v3(sprintf('MITBIH-r%d', r), XTr,YTr, XVa,YVa, XTe,YTe, ...
        trainFcn=@train_baseline_1d, dataFormat='SSCB', batchSize=128, ...
        numBlocks=NBLOCKS, settleSec=60, seed=1, maxToPrune=r, ...
        hwinfoCsv=HW, idleWatts=IDLE_W, outDir=OUTDIR);
end
clear XTr YTr XVa YVa XTe YTe d

%% ---- 4. ChestX-ray14 ----------------------------------------------------
d = fullfile(DATA,'3_NIH_ChestX-ray14');
[XTr,YTr,XVa,YVa,XTe,YTe,infoChest] = load_chestxray_dataset(d, ...
        splitMode="patient", subsetSize=2000);
for r = RATIOS
    fprintf('\n##### ChestX-ray14  maxToPrune = %d #####\n', r);
    run_experiment_v3(sprintf('Chest-r%d', r), XTr,YTr, XVa,YVa, XTe,YTe, ...
        trainFcn=@train_baseline, dataFormat='SSCB', batchSize=64, ...
        numBlocks=NBLOCKS, settleSec=60, burnInSec=300, seed=1, maxToPrune=r, ...
        hwinfoCsv=HW, idleWatts=IDLE_W, outDir=OUTDIR);
end

fprintf('\nA22 ratio sweep finished %s\n', datestr(now)); %#ok<TNOW1,DATST>

%% ---- 5. B1 EXTENSION: independent seeds on MIT-BIH ----------------------
%  Reviewer B asked whether repeated blocks estimate model variability.
%  Seeds 2,3,4,5 on MIT-BIH extend that answer to a second modality, so the
%  claim rests on two datasets rather than one. Seed 1 already exists from
%  the main run. Approx 2 h.

d = fullfile(DATA,'4_MITBIH','mit-bih-arrhythmia-database-1.0.0');
[XTr,YTr,XVa,YVa,XTe,YTe,~] = load_mitbih_dataset(d, splitMode="record");
for sd = [2 3 4 5]
    fprintf('\n##### MIT-BIH  SEED %d #####\n', sd);
    run_experiment_v3(sprintf('MITBIH-seed%d', sd), XTr,YTr, XVa,YVa, XTe,YTe, ...
        trainFcn=@train_baseline_1d, dataFormat='SSCB', batchSize=128, ...
        numBlocks=3, settleSec=60, seed=sd, ...
        hwinfoCsv=HW, idleWatts=IDLE_W, outDir=OUTDIR);
end
clear XTr YTr XVa YVa XTe YTe d

%% ---- 6. A13: 500 ms sampling check on MNIST -----------------------------
%  Reviewer A13 asked whether the 2000 ms logging interval is adequate.
%  This block re-runs MNIST under the same protocol. Set the HWiNFO interval
%  to 500 ms BEFORE starting the script if you want this comparison to be
%  meaningful; otherwise it simply reproduces the main run.

d = fullfile(DATA,'1_MNIST');
[XTr,YTr] = load_idx_dataset(fullfile(d,'train-images.idx3-ubyte'), ...
                             fullfile(d,'train-labels.idx1-ubyte'));
[XTe,YTe] = load_idx_dataset(fullfile(d,'t10k-images.idx3-ubyte'), ...
                             fullfile(d,'t10k-labels.idx1-ubyte'));
s = round(0.9*size(XTr,4));
XVa = XTr(:,:,:,s+1:end); YVa = YTr(s+1:end);
XTr = XTr(:,:,:,1:s);     YTr = YTr(1:s);
fprintf('\n##### MNIST  500 ms sampling check #####\n');
run_experiment_v3('MNIST-fs500', XTr,YTr, XVa,YVa, XTe,YTe, ...
    trainFcn=@train_baseline, dataFormat='SSCB', batchSize=128, ...
    numBlocks=6, settleSec=60, seed=1, ...
    hwinfoCsv=HW, idleWatts=IDLE_W, outDir=OUTDIR);

fprintf('\nALL RUNS FINISHED %s\n', datestr(now)); %#ok<TNOW1,DATST>
diary off
