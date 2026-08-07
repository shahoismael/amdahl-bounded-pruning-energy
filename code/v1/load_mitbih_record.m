function [signal, fs, gain] = load_mitbih_record(recordPath)
% LOAD_MITBIH_RECORD Reads a single MIT-BIH record (.hea + .dat, format 212).
%
%   [signal, fs, gain] = load_mitbih_record(recordPath)
%   recordPath: path WITHOUT extension, e.g. '.../mit-bih.../100'
%
%   signal: [numSamples x numSignals] double, in physical units
%   fs: sampling frequency (Hz)
%   gain: ADC gain used for the first signal (units/mV scaling)

    heaFile = [recordPath '.hea'];
    datFile = [recordPath '.dat'];

    fid = fopen(heaFile, 'r');
    if fid == -1
        error('Could not open header file: %s', heaFile);
    end
    headerLine = fgetl(fid);
    parts = strsplit(strtrim(headerLine));
    numSignals = str2double(parts{2});
    fs = str2double(parts{3});

    gain = 200; % MIT-BIH default gain (adc units per mV) if not parsed below
    for i = 1:numSignals
        sigLine = fgetl(fid);
        sigParts = strsplit(strtrim(sigLine));
        if i == 1 && numel(sigParts) >= 3
            gainStr = sigParts{3};
            gainNum = str2double(strtok(gainStr, '('));
            if ~isnan(gainNum), gain = gainNum; end
        end
    end
    fclose(fid);

    % ---- Read format-212 binary signal data ----
    fid = fopen(datFile, 'r');
    if fid == -1
        error('Could not open data file: %s', datFile);
    end
    raw = fread(fid, Inf, 'uint8=>double');
    fclose(fid);

    numTriplets = floor(numel(raw) / 3);
    raw = raw(1:numTriplets*3);
    A = reshape(raw, 3, numTriplets)';

    M2H = bitshift(bitand(240, A(:,2)), -4); % high nibble of byte2
    M1H = bitand(15, A(:,2));                % low nibble of byte2
    M1 = bitshift(M1H,8) + A(:,1);
    M2 = bitshift(M2H,8) + A(:,3);
    M1(M1>=2048) = M1(M1>=2048) - 4096;
    M2(M2>=2048) = M2(M2>=2048) - 4096;

    interleaved = [M1, M2]';
    allSamples = interleaved(:);

    if numSignals == 2
        signal = reshape(allSamples(1:2*floor(numel(allSamples)/2)), 2, [])';
    else
        signal = allSamples; % single-channel fallback
    end

    signal = signal / gain; % convert ADC units to physical units (mV)
    fprintf('Loaded record %s: %d samples/signal, fs=%d Hz\n', recordPath, size(signal,1), fs);
end
