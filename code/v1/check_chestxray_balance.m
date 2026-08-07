%% CHECK_CHESTXRAY_BALANCE - Reproduces the exact random subset (rng(1))
%% used in main_pipeline_chestxray.m and reports the real class balance.

clear; clc;

dataDir = fullfile('..','data','3_NIH_ChestX-ray14');
labelsCsv = fullfile(dataDir, 'Data_Entry_2017.csv');

labelsTable = readtable(labelsCsv, 'ReadVariableNames', false, 'HeaderLines', 1);
labelsTable.Properties.VariableNames = {'ImageIndex','FindingLabels','FollowUpNum', ...
    'PatientID','PatientAge','PatientGender','ViewPosition', ...
    'OriginalImageWidth','OriginalImageHeight','OriginalImagePixelSpacingX','OriginalImagePixelSpacingY'};

isNoFinding = strcmp(labelsTable.FindingLabels, 'No Finding');
labelsTable.BinaryLabel = categorical(isNoFinding, [true false], {'NoFinding','Finding'});

N = 2000;
rng(1); % identical seed used in the original pipeline
subsetIdx = randperm(height(labelsTable), min(N, height(labelsTable)));
subsetTable = labelsTable(subsetIdx, :);

counts = countcats(subsetTable.BinaryLabel);
catNames = categories(subsetTable.BinaryLabel);

fprintf('=== ChestX-ray14 subset class balance (N=%d, rng(1)) ===\n', height(subsetTable));
for i = 1:numel(catNames)
    fprintf('%s: %d (%.2f%%)\n', catNames{i}, counts(i), 100*counts(i)/height(subsetTable));
end
