%% MAIN PIPELINE - NIH ChestX-ray14 (simplified: "No Finding" vs "Any Finding")
% Full 14-label multi-label classification is out of scope for this
% energy-audit study; we use a simplified binary task consistent with
% several published Green-AI benchmarking papers.

clear; clc;
warning('off', 'MATLAB:imagesci:png:libraryWarning');

dataDir = fullfile('..','data','3_NIH_ChestX-ray14');
labelsCsv = fullfile(dataDir, 'Data_Entry_2017.csv');

labelsTable = readtable(labelsCsv, 'ReadVariableNames', false, 'HeaderLines', 1);
labelsTable.Properties.VariableNames = {'ImageIndex','FindingLabels','FollowUpNum', ...
    'PatientID','PatientAge','PatientGender','ViewPosition', ...
    'OriginalImageWidth','OriginalImageHeight','OriginalImagePixelSpacingX','OriginalImagePixelSpacingY'};

findingColName = 'FindingLabels';
imageIndexColName = 'ImageIndex';
fprintf('Using column: %s\n', findingColName);

isNoFinding = strcmp(labelsTable.(findingColName), 'No Finding');
labelsTable.BinaryLabel = categorical(isNoFinding, [true false], {'NoFinding','Finding'});

% ---- Build a manageable subset for the energy audit (adjust N as needed) ----
N = 2000; % subset size; increase later if time allows
rng(1);
subsetIdx = randperm(height(labelsTable), min(N, height(labelsTable)));
subsetTable = labelsTable(subsetIdx, :);

imageFolders = dir(fullfile(dataDir, 'images_*'));
imagePaths = strings(height(subsetTable),1);
validMask = false(height(subsetTable),1);

for i = 1:height(subsetTable)
    fname = subsetTable.(imageIndexColName){i};
    found = false;
    for f = 1:numel(imageFolders)
        candidate = fullfile(dataDir, imageFolders(f).name, 'images', fname);
        if exist(candidate, 'file')
            imagePaths(i) = candidate;
            found = true;
            break;
        end
    end
    validMask(i) = found;
end

imagePaths = imagePaths(validMask);
labelsSubset = subsetTable.BinaryLabel(validMask);
fprintf('Found %d/%d images on disk.\n', numel(imagePaths), height(subsetTable));

imds = imageDatastore(imagePaths, 'Labels', labelsSubset);
imds.ReadFcn = @read_and_resize_gray;

[imdsTrain, imdsVal, imdsTest] = splitEachLabel(imds, 0.7, 0.15, 0.15, 'randomized');

numClasses = numel(categories(labelsSubset));

layers = [
    imageInputLayer([128 128 1], 'Name', 'input')
    convolution2dLayer(3, 16, 'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3, 32, 'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 32, ...
    'ValidationData', imdsVal, ...
    'ValidationFrequency', 20, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

baselineNet = trainNetwork(imdsTrain, layers, options);
save(fullfile('..','results','baseline_net_chestxray.mat'), 'baselineNet');

% ---- For pruning/quantization/energy, load a fixed-size test array ----
XTest = zeros(128,128,1,numel(imdsTest.Files));
for i = 1:numel(imdsTest.Files)
    XTest(:,:,1,i) = read_and_resize_gray(imdsTest.Files{i});
end
YTest = imdsTest.Labels;

% Need a small XTrain array too for pruning's fine-tuning step
XTrainArr = zeros(128,128,1,numel(imdsTrain.Files));
for i = 1:numel(imdsTrain.Files)
    XTrainArr(:,:,1,i) = read_and_resize_gray(imdsTrain.Files{i});
end
YTrainArr = imdsTrain.Labels;

prunedNet = prune_model(baselineNet, XTrainArr, YTrainArr, 2, 4);
save(fullfile('..','results','pruned_net_chestxray.mat'), 'prunedNet');

YPredPrunedRaw = predict(prunedNet, dlarray(single(XTest), 'SSCB'));
[~, predIdx] = max(extractdata(YPredPrunedRaw), [], 1);
classNames = categories(YTest);
YPredPruned = categorical(classNames(predIdx));
accPruned = mean(YPredPruned(:) == YTest(:));
fprintf('Pruned model accuracy (ChestX-ray14): %.2f%%\n', accPruned*100);

quantizedNet = quantize_model(baselineNet, XTrainArr(:,:,:,1:min(100,size(XTrainArr,4))));
save(fullfile('..','results','quantized_net_chestxray.mat'), 'quantizedNet');

resultsBaseline = measure_energy(baselineNet, XTest, 30, 'Baseline-ChestXray');
resultsPruned   = measure_energy(prunedNet,   XTest, 30, 'Pruned-ChestXray');

runN = get_next_run_number(fullfile('..','results'), 'all_results_chestxray');
allResults = struct('baseline', resultsBaseline, 'pruned', resultsPruned);
save(fullfile('..','results', sprintf('all_results_chestxray_run%d.mat', runN)), 'allResults');
fprintf('Saved as run #%d.\n', runN);

disp('ChestX-ray14 pipeline complete. Extract watts from HWiNFO CSV log using printed timestamps.');

function imgOut = read_and_resize_gray(filepath)
% Forces any input (RGB or grayscale, with or without odd ICC profiles)
% into a single-channel [128 128] image.
    img = imread(filepath);
    if size(img,3) == 3
        img = rgb2gray(img);
    elseif size(img,3) > 3
        img = img(:,:,1); % fallback: take first channel
    end
    imgOut = imresize(img, [128 128]);
end
