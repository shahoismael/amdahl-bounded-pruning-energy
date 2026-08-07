%% MEASURE_IDLE_POWER - Measures true no-load CPU package power for comparison
%% against the active-inference measurements reported elsewhere.
%% Run this with MATLAB otherwise idle (no other scripts running).

clear; clc;

fprintf('Starting idle power measurement.\n');
fprintf('Ensure HWiNFO CSV logging is running, then wait...\n');

startTime = datetime('now');
fprintf('START: %s\n', datestr(startTime, 'HH:MM:SS.FFF'));

pause(30); % 30-second idle window, matching the active measurement duration

endTime = datetime('now');
fprintf('END:   %s\n', datestr(endTime, 'HH:MM:SS.FFF'));
fprintf('--> Now go to the HWiNFO CSV log, average CPU Package Power [W] between %s and %s\n', ...
    datestr(startTime,'HH:MM:SS'), datestr(endTime,'HH:MM:SS'));
