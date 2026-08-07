function results = measure_energy(net, XTest, durationSeconds, modelName)
% MEASURE_ENERGY Runs continuous inference for a fixed duration and prints
% exact start/end timestamps so they can be matched against an HWiNFO CSV
% power log (logged independently, e.g. at 1-second intervals).
%
%   results = measure_energy(net, XTest, durationSeconds, modelName)
%   durationSeconds : how long to run continuous inference, e.g. 30
%
%   After running, open the HWiNFO CSV log, find rows between
%   results.startTime and results.endTime, and average the
%   "CPU Package Power [W]" column over that window. Enter that value
%   into results.measuredWatts, then recompute energyJoulesPerInference.

    results = struct();
    results.modelName = modelName;

    isDl = isa(net, 'dlnetwork');

    results.startTime = datetime('now');
    fprintf('[%s] START: %s\n', modelName, datestr(results.startTime, 'HH:MM:SS.FFF'));

    numRuns = 0;
    tic;
    while toc < durationSeconds
        idx = randi(size(XTest,4));
        Xsample = XTest(:,:,:,idx);
        if isDl
            Xsample = dlarray(single(Xsample), 'SSCB');
        end
        predict(net, Xsample);
        numRuns = numRuns + 1;
    end
    elapsedSeconds = toc;

    results.endTime = datetime('now');
    fprintf('[%s] END:   %s\n', modelName, datestr(results.endTime, 'HH:MM:SS.FFF'));

    results.numRuns = numRuns;
    results.avgInferenceTimeSec = elapsedSeconds / numRuns;

    fprintf('[%s] Ran %d inferences in %.2f s | Avg time/inference: %.6f s\n', ...
        modelName, numRuns, elapsedSeconds, results.avgInferenceTimeSec);
    fprintf('[%s] --> Now go to the HWiNFO CSV log, average CPU Package Power [W] between %s and %s\n', ...
        modelName, datestr(results.startTime,'HH:MM:SS'), datestr(results.endTime,'HH:MM:SS'));

    % ---- Fill in after reading the averaged watts from the CSV log ----
    results.measuredWatts = NaN;  % <-- replace with averaged watts from CSV
    results.energyJoulesPerInference = (results.measuredWatts * elapsedSeconds) / numRuns;
end
