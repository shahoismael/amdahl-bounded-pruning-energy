function s = parse_hwinfo_log(csvPath, startTime, endTime, opts)
%PARSE_HWINFO_LOG Average sensor readings over a measurement window.
%
%   s = PARSE_HWINFO_LOG(csvPath, startTime, endTime)
%
%   Replaces the manual "open the CSV and average by eye" step that left
%   measuredWatts = NaN in every v1 result file. Returns the full
%   distribution plus temperature and clock covariates, so a throttling
%   event can be detected rather than inferred.
%
%   TIMESTAMP HANDLING
%   ------------------
%   Matches on full date+time (HWiNFO logs a Date column), so windows that
%   cross midnight work. An earlier version of this function built the
%   timestamps with a hand-written regex and a try/catch loop over candidate
%   date formats; that silently dropped rows on this log (14 samples where a
%   direct parse found 15) and rejected two run windows entirely. It is now
%   a single explicit datetime + duration parse, and any row that fails to
%   parse is COUNTED and reported rather than quietly discarded.
%
%   The log is also read once and cached, since a 12-run recovery otherwise
%   re-parses a 5000-row, 344-column CSV twenty-four times.
%
%   OPTIONS
%     .powerPattern  regexp for the power column (default 'CPU Package Power')
%     .tempPattern   regexp for temperature
%     .clockPattern  regexp for clock
%     .minSamples    warn below this many samples (default 20)
%     .dateFormat    override the date format (default: auto-detect)
%     .timeFormat    override the time format (default 'hh:mm:ss.SSS')
%     .reload        force re-read of the CSV, bypassing the cache

    arguments
        csvPath   (1,:) char
        startTime (1,1) datetime
        endTime   (1,1) datetime
        opts.powerPattern (1,:) char = 'CPU Package Power'
        opts.tempPattern  (1,:) char = 'CPU Package.*Temp|Core Temp|CPU \(Tctl'
        opts.clockPattern (1,:) char = 'Core Clock|Effective Clock'
        opts.minSamples   (1,1) double = 20
        opts.dateFormat   (1,:) char = ''
        opts.timeFormat   (1,:) char = 'hh:mm:ss.SSS'
        opts.reload       (1,1) logical = false
    end

    [stamp, T, vars, parseReport] = load_log_cached(csvPath, ...
        opts.dateFormat, opts.timeFormat, opts.reload);

    valid    = ~isnat(stamp);
    inWindow = valid & (stamp >= startTime) & (stamp <= endTime);
    n = sum(inWindow);

    s = struct();
    s.csvPath     = csvPath;
    s.startTime   = startTime;
    s.endTime     = endTime;
    s.windowSec   = seconds(endTime - startTime);
    s.numSamples  = n;
    s.logSpan     = [min(stamp(valid)) max(stamp(valid))];
    s.parseReport = parseReport;

    if n == 0
        error('parse_hwinfo_log:EmptyWindow', ...
              ['No log rows between %s and %s.\n' ...
               'Log spans %s to %s (%d of %d rows parsed).\n' ...
               'If the window IS inside that span, the timestamp parse is at ' ...
               'fault -- check s.parseReport.'], ...
              string(startTime), string(endTime), ...
              string(s.logSpan(1)), string(s.logSpan(2)), ...
              parseReport.nParsed, parseReport.nRows);
    end
    if n < opts.minSamples
        warning('parse_hwinfo_log:FewSamples', ...
                ['Only %d samples in a %.1f s window. Reduce the HWiNFO logging ' ...
                 'interval (500 ms recommended) before trusting the variance.'], ...
                n, s.windowSec);
    end

    s.power = summarise_column(T, vars, inWindow, opts.powerPattern, 'power');
    s.temp  = summarise_column(T, vars, inWindow, opts.tempPattern,  'temperature');
    s.clock = summarise_column(T, vars, inWindow, opts.clockPattern, 'clock');

    if isnan(s.power.mean)
        error('parse_hwinfo_log:NoPowerColumn', ...
              'No column matched power pattern "%s".', opts.powerPattern);
    end
end

% -------------------------------------------------------------------------
function [stamp, T, vars, rep] = load_log_cached(csvPath, dateFmt, timeFmt, reload)
%LOAD_LOG_CACHED Read and timestamp the log once per file, then reuse.
    persistent cachePath cacheStamp cacheT cacheVars cacheRep cacheDir

    info = dir(csvPath);
    key  = sprintf('%s|%d|%s', csvPath, info.bytes, info.date);

    if ~reload && ~isempty(cachePath) && strcmp(cachePath, key)
        stamp = cacheStamp; T = cacheT; vars = cacheVars; rep = cacheRep;
        return;
    end

    impOpts = detectImportOptions(csvPath, 'VariableNamingRule', 'preserve', ...
                                  'Encoding', 'ISO-8859-1');
    impOpts.VariableNamesLine = 1;
    impOpts.DataLines         = [2 Inf];
    impOpts = setvartype(impOpts, 'char');
    T = readtable(csvPath, impOpts);

    vars = strtrim(erase(string(T.Properties.VariableNames), '"'));

    timeCol = find(strcmpi(vars, 'Time'), 1);
    dateCol = find(strcmpi(vars, 'Date'), 1);
    if isempty(timeCol)
        error('parse_hwinfo_log:NoTimeColumn', ...
              'No Time column. First 5 columns: %s', strjoin(vars(1:min(5,end)), ', '));
    end

    rawTime = strtrim(erase(string(T{:, timeCol}), '"'));
    tod = to_duration(rawTime, timeFmt);

    if isempty(dateCol)
        d = repmat(dateshift(datetime('now'), 'start', 'day'), size(rawTime));
    else
        rawDate = strtrim(erase(string(T{:, dateCol}), '"'));
        d = to_date(rawDate, dateFmt);
    end

    stamp = d + tod;

    rep = struct('nRows', numel(stamp), 'nParsed', sum(~isnat(stamp)), ...
                 'nFailed', sum(isnat(stamp)));
    if rep.nFailed > 0
        warning('parse_hwinfo_log:UnparsedRows', ...
                '%d of %d log rows had unparseable timestamps and were excluded.', ...
                rep.nFailed, rep.nRows);
    end

    cachePath = key; cacheStamp = stamp; cacheT = T;
    cacheVars = vars; cacheRep = rep;  cacheDir = info; %#ok<NASGU>
end

% -------------------------------------------------------------------------
function tod = to_duration(raw, fmt)
%TO_DURATION Parse the Time column, trying the stated format then variants.
    candidates = [{fmt}, {'hh:mm:ss.SSS', 'hh:mm:ss'}];
    for k = 1:numel(candidates)
        if isempty(candidates{k}), continue; end
        try
            tod = duration(raw, 'InputFormat', candidates{k});
            if sum(~isnan(tod)) > 0.5 * numel(raw), return; end
        catch
        end
    end
    error('parse_hwinfo_log:TimeParseFailed', ...
          'Could not parse the Time column. First value was "%s".', raw(1));
end

% -------------------------------------------------------------------------
function d = to_date(raw, fmt)
%TO_DATE Parse the Date column. Picks the format that parses the MOST rows
%   rather than the first that does not throw -- a lenient-but-wrong format
%   would otherwise win and corrupt every timestamp.
    if ~isempty(fmt)
        d = datetime(raw, 'InputFormat', fmt);
        return;
    end

    candidates = {'d.M.yyyy', 'dd.MM.yyyy', 'd/M/yyyy', 'dd/MM/yyyy', ...
                  'M/d/yyyy', 'MM/dd/yyyy', 'yyyy-MM-dd', 'd-M-yyyy'};
    best = NaT(size(raw)); bestN = -1;
    for k = 1:numel(candidates)
        try
            trial = datetime(raw, 'InputFormat', candidates{k});
        catch
            continue;
        end
        nOk = sum(~isnat(trial));
        if nOk > bestN
            bestN = nOk; best = trial;
        end
        if nOk == numel(raw), break; end
    end

    if bestN <= 0
        error('parse_hwinfo_log:DateParseFailed', ...
              'Could not parse the Date column. First value was "%s".', raw(1));
    end
    d = best;
end

% -------------------------------------------------------------------------
function c = summarise_column(T, vars, mask, pattern, label)
    c = struct('name', "", 'mean', NaN, 'sd', NaN, 'min', NaN, 'max', NaN, ...
               'n', 0, 'values', []);
    idx = find(~cellfun(@isempty, regexpi(cellstr(vars), pattern, 'once')), 1);
    if isempty(idx)
        if ~strcmp(label, 'power')
            warning('parse_hwinfo_log:NoColumn', ...
                    'No %s column matched "%s"; continuing without it.', label, pattern);
        end
        return;
    end
    v = str2double(erase(strtrim(string(T{mask, idx})), '"'));
    nDropped = sum(isnan(v));
    v = v(~isnan(v));
    if isempty(v), return; end
    if nDropped > 0
        warning('parse_hwinfo_log:NonNumeric', ...
                '%d non-numeric %s readings in window were dropped.', nDropped, label);
    end
    c.name   = vars(idx);
    c.values = v;
    c.n      = numel(v);
    c.mean   = mean(v);
    c.sd     = std(v);
    c.min    = min(v);
    c.max    = max(v);
end
