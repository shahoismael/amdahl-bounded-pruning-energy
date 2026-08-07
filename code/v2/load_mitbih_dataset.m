function [XTr,YTr,XVa,YVa,XTe,YTe,info] = load_mitbih_dataset(dataDir, opts)
%LOAD_MITBIH_DATASET Build the MIT-BIH beat-classification dataset.
%
%   [XTr,YTr,XVa,YVa,XTe,YTe,info] = LOAD_MITBIH_DATASET(dataDir)
%   ... = LOAD_MITBIH_DATASET(dataDir, splitMode="record")
%
%   TWO FIXES OVER THE v1 INLINE LOADER
%   -----------------------------------
%   1. SPLIT LEAKAGE. v1 used splitMode "beat": every extracted beat was
%      shuffled independently, so beats from the same patient appeared in both
%      the training and the test set. Consecutive beats from one ECG record
%      are highly correlated -- same electrode placement, same morphology,
%      same noise -- so a classifier can score well by recognising the
%      patient rather than the arrhythmia. Inter-patient (record-level)
%      splitting is the standard protocol in the ECG literature precisely
%      because of this.
%
%      DEFAULT IS "record". This will report LOWER accuracy than v1. That
%      lower number is the honest one. splitMode="beat" reproduces the v1
%      behaviour if the old figures need to be regenerated.
%
%      Note: this affects the ACCURACY column only. The energy audit is
%      unaffected -- energy depends on the network's shape, not on whether
%      its labels generalise.
%
%   2. PREALLOCATION. v1 grew the array one beat at a time with
%      XAll(:,1,1,end+1) = seg, which reallocates and copies the whole array
%      on every one of ~34,000 iterations. Beats are now collected per record
%      and concatenated once.
%
%   OPTIONS
%     .recordIDs      cellstr of record numbers
%     .windowSamples  samples per beat window (default 250)
%     .splitMode      "record" (default) | "beat"
%     .fractions      [train val test] (default [0.7 0.15 0.15])
%     .seed           RNG seed (default 1)

    arguments
        dataDir (1,:) char
        opts.recordIDs (1,:) cell = {'100','101','103','105','106','108','109','111', ...
                                     '112','113','114','115','116','117','118','119'}
        opts.windowSamples (1,1) double = 250
        opts.splitMode (1,:) char {mustBeMember(opts.splitMode,{'record','beat'})} = 'record'
        opts.fractions (1,3) double = [0.7 0.15 0.15]
        opts.seed (1,1) double = 1
    end

    rng(opts.seed, 'twister');
    half = floor(opts.windowSamples/2);

    segsByRecord  = cell(numel(opts.recordIDs),1);
    labelByRecord = cell(numel(opts.recordIDs),1);
    kept = false(numel(opts.recordIDs),1);

    for r = 1:numel(opts.recordIDs)
        recPath = fullfile(dataDir, opts.recordIDs{r});
        try
            [signal, ~, ~] = load_mitbih_record(recPath);
            [sampleIdx, typeCodes] = read_atr_annotations([recPath '.atr']);
        catch ME
            fprintf('  skipping record %s: %s\n', opts.recordIDs{r}, ME.message);
            continue;
        end

        sig = signal(:,1);
        ok  = (sampleIdx - half >= 1) & (sampleIdx + half <= numel(sig));
        c   = sampleIdx(ok);
        tc  = typeCodes(ok);
        if isempty(c), continue; end

        % Vectorised window extraction: one index matrix, one gather.
        offsets = (-half : half-1)';                 % [W x 1]
        idxMat  = offsets + c(:)';                   % [W x nBeats]
        segs    = sig(idxMat);                       % [W x nBeats]

        segsByRecord{r}  = reshape(segs, [opts.windowSamples, 1, 1, numel(c)]);
        labelByRecord{r} = double(tc(:) ~= 1) + 1;   % 1 = Normal, 2 = Abnormal
        kept(r) = true;
    end

    segsByRecord  = segsByRecord(kept);
    labelByRecord = labelByRecord(kept);
    recIDs        = opts.recordIDs(kept);

    counts = cellfun(@(s) size(s,4), segsByRecord);
    XAll   = cat(4, segsByRecord{:});
    yAll   = vertcat(labelByRecord{:});
    recOf  = repelem((1:numel(recIDs))', counts);    % record index per beat

    YAll = categorical(yAll, [1 2], {'Normal','Abnormal'});

    info = struct();
    info.splitMode     = string(opts.splitMode);
    info.numRecords    = numel(recIDs);
    info.recordIDs     = string(recIDs);
    info.beatsPerRecord= counts(:)';
    info.numBeats      = numel(YAll);
    info.classCounts   = countcats(YAll);
    info.classNames    = categories(YAll);

    fprintf('MIT-BIH: %d beats from %d records | Normal %d (%.2f%%), Abnormal %d (%.2f%%)\n', ...
        info.numBeats, info.numRecords, info.classCounts(1), ...
        info.classCounts(1)/info.numBeats*100, info.classCounts(2), ...
        info.classCounts(2)/info.numBeats*100);

    switch opts.splitMode
        case 'beat'
            % v1 behaviour, retained only for reproducing old results.
            n  = numel(YAll);
            sh = randperm(n);
            a  = round(opts.fractions(1)*n);
            b  = round((opts.fractions(1)+opts.fractions(2))*n);
            trIdx = sh(1:a); vaIdx = sh(a+1:b); teIdx = sh(b+1:end);
            warning('load_mitbih_dataset:BeatLevelSplit', ...
                ['splitMode="beat" puts beats from the same patient in both ' ...
                 'train and test. Accuracy from this split is optimistically ' ...
                 'biased and should not be reported as a generalisation estimate.']);

        case 'record'
            % STRATIFY BY ABNORMAL FRACTION.
            % A plain random record split is unusable here. The 16 records
            % differ enormously in composition -- 109, 111 and 118 are almost
            % entirely abnormal (LBBB/RBBB), while 101, 113 and 117 are almost
            % entirely normal. A random draw put records 101 and 117 in the
            % test set, giving 3392 Normal against 17 Abnormal (99.5%), so a
            % constant "Normal" predictor scored 99.09% and the pruned model
            % appeared to gain 34.79 pp over its baseline.
            %
            % Records are therefore ordered by abnormal fraction and dealt
            % round-robin, so every split spans the full range of morphology.
            nRec = numel(recIDs);
            abnFrac = zeros(nRec,1);
            for i = 1:nRec
                abnFrac(i) = mean(labelByRecord{i} == 2);
            end
            [~, ord] = sort(abnFrac);

            targets = opts.fractions / sum(opts.fractions);
            bins = {[],[],[]};
            load_ = zeros(1,3);
            for i = 1:nRec
                % Deal each record to whichever split is furthest below quota.
                deficit = targets*i - load_;
                [~, w] = max(deficit);
                bins{w}(end+1) = ord(i); %#ok<AGROW>
                load_(w) = load_(w) + 1;
            end
            trRec = bins{1}; vaRec = bins{2}; teRec = bins{3};

            fprintf('  stratified by abnormal fraction (range %.1f%%-%.1f%%)\n', ...
                    min(abnFrac)*100, max(abnFrac)*100);
            trIdx = find(ismember(recOf, trRec));
            vaIdx = find(ismember(recOf, vaRec));
            teIdx = find(ismember(recOf, teRec));
            info.trainRecords = info.recordIDs(trRec);
            info.valRecords   = info.recordIDs(vaRec);
            info.testRecords  = info.recordIDs(teRec);
            fprintf('  record-level split: %d train / %d val / %d test records\n', ...
                    numel(trRec), numel(vaRec), numel(teRec));
    end

    XTr = XAll(:,:,:,trIdx);  YTr = YAll(trIdx);
    XVa = XAll(:,:,:,vaIdx);  YVa = YAll(vaIdx);
    XTe = XAll(:,:,:,teIdx);  YTe = YAll(teIdx);

    info.splitSizes = [numel(YTr) numel(YVa) numel(YTe)];
    fprintf('  beats: %d train / %d val / %d test\n', info.splitSizes);

    % Every class must be present in every split, or training silently
    % degenerates and accuracy becomes uninterpretable.
    if any(countcats(YTr)==0) || any(countcats(YVa)==0) || any(countcats(YTe)==0)
        warning('load_mitbih_dataset:MissingClass', ...
                ['A class is absent from at least one split. Re-run with a ' ...
                 'different seed, or use more records.']);
    end
end
