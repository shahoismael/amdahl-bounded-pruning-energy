%RUN_ALL_V3  Driver: four datasets through run_experiment_v3.
%
%   Same protocol and same options as the v2 run that produced
%   results_v2/all_datasets_final.csv. Only the persistence layer differs.
%
%   BEFORE RUNNING
%     1. HWiNFO: sensors-only, CSV logging ON, interval 500 ms.
%     2. One CSV per dataset. Set the paths in HW below.
%     3. Mains power. Battery not charging. Power plan Balanced.
%     4. Close everything else.
%     5. Idle 5 minutes before starting.
%     6. Run measure_idle_power and put the figure in IDLE_W below.
%
%   Runtime is about 25 minutes per dataset (6 blocks x 3 arms x 60 s plus
%   20 s settles) on top of training.

clear; clc;

ROOT   = 'D:\claude_projects\ICIEAHS 2026\research_v4';
DATA   = fullfile(ROOT, 'datasets');
OUTDIR = 'D:\claude_projects\ICIEAHS 2026\final_submission\results';

addpath(fullfile(ROOT, 'code_v1'));
addpath(fullfile(ROOT, 'code_v1', 'v2'));
if ~exist(OUTDIR, 'dir'), mkdir(OUTDIR); end

IDLE_W = 4.221;                       % <-- from measure_idle_power

HW = struct( ...
    'mnist',   fullfile(OUTDIR, 'hwinfo_v3_mnist.csv'), ...
    'fashion', fullfile(OUTDIR, 'hwinfo_v3_fashion.csv'), ...
    'mitbih',  fullfile(OUTDIR, 'hwinfo_v3_mitbih.csv'), ...
    'chest',   fullfile(OUTDIR, 'hwinfo_v3_chest.csv'));

diary(fullfile(OUTDIR, sprintf('console_%s.txt', datestr(now,'yyyymmdd_HHMMSS')))); %#ok<TNOW1,DATST>
diary on
fprintf('run_all_v3 started %s\n', datestr(now)); %#ok<TNOW1,DATST>

%% ---- 1. MNIST -----------------------------------------------------------
d = fullfile(DATA, '1_MNIST');
[XTr,YTr] = load_idx_dataset(fullfile(d,'train-images.idx3-ubyte'), ...
                             fullfile(d,'train-labels.idx1-ubyte'));
[XTe,YTe] = load_idx_dataset(fullfile(d,'t10k-images.idx3-ubyte'), ...
                             fullfile(d,'t10k-labels.idx1-ubyte'));
s = round(0.9*size(XTr,4));
XVa = XTr(:,:,:,s+1:end); YVa = YTr(s+1:end);
XTr = XTr(:,:,:,1:s);     YTr = YTr(1:s);

T_mnist = run_experiment_v3('MNIST', XTr,YTr, XVa,YVa, XTe,YTe, ...
    trainFcn = @train_baseline, dataFormat = 'SSCB', batchSize = 128, ...
    hwinfoCsv = HW.mnist, idleWatts = IDLE_W, outDir = OUTDIR);

clear XTr YTr XVa YVa XTe XTe YTe s d

%% ---- 2. Fashion-MNIST ---------------------------------------------------
d = fullfile(DATA, '2_Fashion_MNIST', 'data', 'fashion');
[XTr,YTr] = load_idx_dataset(fullfile(d,'train-images-idx3-ubyte'), ...
                             fullfile(d,'train-labels-idx1-ubyte'));
[XTe,YTe] = load_idx_dataset(fullfile(d,'t10k-images-idx3-ubyte'), ...
                             fullfile(d,'t10k-labels-idx1-ubyte'));
s = round(0.9*size(XTr,4));
XVa = XTr(:,:,:,s+1:end); YVa = YTr(s+1:end);
XTr = XTr(:,:,:,1:s);     YTr = YTr(1:s);

T_fashion = run_experiment_v3('Fashion-MNIST', XTr,YTr, XVa,YVa, XTe,YTe, ...
    trainFcn = @train_baseline, dataFormat = 'SSCB', batchSize = 128, ...
    hwinfoCsv = HW.fashion, idleWatts = IDLE_W, outDir = OUTDIR);

clear XTr YTr XVa YVa XTe YTe s d

%% ---- 3. MIT-BIH ---------------------------------------------------------
d = fullfile(DATA, '4_MITBIH', 'mit-bih-arrhythmia-database-1.0.0');
[XTr,YTr,XVa,YVa,XTe,YTe,infoMit] = load_mitbih_dataset(d, splitMode = "record");

T_mitbih = run_experiment_v3('MIT-BIH', XTr,YTr, XVa,YVa, XTe,YTe, ...
    trainFcn = @train_baseline_1d, dataFormat = 'SSCB', batchSize = 128, ...
    hwinfoCsv = HW.mitbih, idleWatts = IDLE_W, outDir = OUTDIR);

save(fullfile(OUTDIR,'info_mitbih.mat'), 'infoMit');
clear XTr YTr XVa YVa XTe YTe d

%% ---- 4. ChestX-ray14 ----------------------------------------------------
d = fullfile(DATA, '3_NIH_ChestX-ray14');
[XTr,YTr,XVa,YVa,XTe,YTe,infoChest] = load_chestxray_dataset(d, ...
    splitMode = "patient", subsetSize = 2000);

T_chest = run_experiment_v3('ChestX-ray14', XTr,YTr, XVa,YVa, XTe,YTe, ...
    trainFcn = @train_baseline, dataFormat = 'SSCB', batchSize = 64, ...
    hwinfoCsv = HW.chest, idleWatts = IDLE_W, outDir = OUTDIR);

save(fullfile(OUTDIR,'info_chestxray.mat'), 'infoChest');
clear XTr YTr XVa YVa XTe YTe d

%% ---- 5. Pool ------------------------------------------------------------
allT = [T_mnist; T_fashion; T_mitbih; T_chest];
writetable(allT, fullfile(OUTDIR, 'all_datasets_v3.csv'));
fprintf('\nWrote %s\n', fullfile(OUTDIR, 'all_datasets_v3.csv'));
fprintf('run_all_v3 finished %s\n', datestr(now)); %#ok<TNOW1,DATST>
diary off
