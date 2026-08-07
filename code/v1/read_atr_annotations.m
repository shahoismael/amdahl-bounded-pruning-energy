function [sampleIdx, typeCodes] = read_atr_annotations(atrPath)
% READ_ATR_ANNOTATIONS Parses a MIT-BIH .atr binary annotation file.
%
%   [sampleIdx, typeCodes] = read_atr_annotations(atrPath)
%   sampleIdx : sample index (in the signal) of each annotated beat
%   typeCodes : MIT-BIH annotation type code for each beat (1 = Normal 'N', etc.)
%
%   NOTE: this implements the standard/common annotation entries. Rare
%   extension codes are skipped as best-effort; if beat counts look off,
%   cross-check against the record's documented annotation count.

    fid = fopen(atrPath, 'r');
    if fid == -1
        error('Could not open annotation file: %s', atrPath);
    end
    bytes = fread(fid, Inf, 'uint8=>double');
    fclose(fid);

    numWords = floor(numel(bytes)/2);
    words = bytes(1:2:2*numWords) + bitshift(bytes(2:2:2*numWords), 8);

    sampleIdx = [];
    typeCodes = [];
    t = 0;
    i = 1;
    while i <= numel(words)
        w = words(i);
        typeCode = bitshift(w, -10);
        timeInc  = bitand(w, 1023);

        if typeCode == 0 && timeInc == 0
            break; % EOF marker
        elseif typeCode == 59 % SKIP: next 2 words form a 32-bit time increment
            if i+2 <= numel(words)
                highW = words(i+1);
                lowW  = words(i+2);
                skipAmount = bitshift(highW,16) + lowW;
                t = t + skipAmount;
            end
            i = i + 3;
            continue;
        elseif typeCode >= 60 && typeCode <= 63
            % NUM/SUB/CHAN/AUX: auxiliary info, not a beat label.
            % AUX (63) has a length byte + string payload; skip words needed.
            if typeCode == 63
                strLen = timeInc;
                extraWords = ceil(strLen/2);
                i = i + 1 + extraWords;
            else
                i = i + 1;
            end
            continue;
        else
            t = t + timeInc;
            sampleIdx(end+1,1) = t; %#ok<AGROW>
            typeCodes(end+1,1) = typeCode; %#ok<AGROW>
            i = i + 1;
        end
    end
    fprintf('Parsed %d annotations from %s\n', numel(sampleIdx), atrPath);
end
