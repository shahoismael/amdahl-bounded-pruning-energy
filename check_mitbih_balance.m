%% CHECK_MITBIH_BALANCE - Tallies real Normal vs Abnormal beat counts
%% across the same 16 records used in main_pipeline_mitbih.m.

clear; clc;

dataDir = fullfile('..','data','4_MITBIH','mit-bih-arrhythmia-database-1.0.0');
recordIDs = {'100','101','103','105','106','108','109','111', ...
             '112','113','114','115','116','117','118','119'};

windowSamples = 250;
half = floor(windowSamples/2);
normalCount = 0;
abnormalCount = 0;
totalSegments = 0;

for r = 1:numel(recordIDs)
    recPath = fullfile(dataDir, recordIDs{r});
    try
        [signal, ~, ~] = load_mitbih_record(recPath);
        [sampleIdx, typeCodes] = read_atr_annotations([recPath '.atr']);
        sig = signal(:,1);

        for k = 1:numel(sampleIdx)
            c = sampleIdx(k);
            if c-half >= 1 && c+half <= numel(sig)
                totalSegments = totalSegments + 1;
                if typeCodes(k) == 1
                    normalCount = normalCount + 1;
                else
                    abnormalCount = abnormalCount + 1;
                end
            end
        end
    catch ME
        fprintf('Skipping record %s: %s\n', recordIDs{r}, ME.message);
    end
end

fprintf('=== MIT-BIH class balance across 16 records ===\n');
fprintf('Total segments: %d\n', totalSegments);
fprintf('Normal: %d (%.2f%%)\n', normalCount, 100*normalCount/totalSegments);
fprintf('Abnormal: %d (%.2f%%)\n', abnormalCount, 100*abnormalCount/totalSegments);
