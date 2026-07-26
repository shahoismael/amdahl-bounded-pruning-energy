function n = get_next_run_number(resultsDir, prefix)
% GET_NEXT_RUN_NUMBER Finds the next available run index for a given
% results prefix, so repeated runs don't overwrite each other.
%
%   n = get_next_run_number(resultsDir, prefix)
%   e.g. prefix = 'all_results_mnist' -> looks for all_results_mnist_run*.mat

    if ~exist(resultsDir, 'dir')
        mkdir(resultsDir);
    end
    existing = dir(fullfile(resultsDir, [prefix '_run*.mat']));
    maxN = 0;
    for i = 1:numel(existing)
        tok = regexp(existing(i).name, '_run(\d+)\.mat', 'tokens');
        if ~isempty(tok)
            maxN = max(maxN, str2double(tok{1}{1}));
        end
    end
    n = maxN + 1;
end
