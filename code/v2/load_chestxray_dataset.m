function [XTr,YTr,XVa,YVa,XTe,YTe,info] = load_chestxray_dataset(dataDir, opts)
%LOAD_CHESTXRAY_DATASET Build the ChestX-ray14 binary subtask as in-memory arrays.
%
%   [XTr,YTr,XVa,YVa,XTe,YTe,info] = LOAD_CHESTXRAY_DATASET(dataDir)
%   ... = LOAD_CHESTXRAY_DATASET(dataDir, splitMode="patient", subsetSize=2000)
%
%   THREE FIXES OVER THE v1 INLINE PIPELINE
%   ---------------------------------------
%   1. SPLIT LEAKAGE. v1 used splitEachLabel(imds, 0.7, 0.15, 0.15,
%      'randomized'), which splits at IMAGE level. ChestX-ray14 contains
%      multiple images per patient (that is what the FollowUpNum column is),
%      so the same patient's radiographs land in both train and test.
%      DEFAULT IS "patient": all images from a patient go to one split.
%      splitMode="image" reproduces v1.
%
%   2. IN-MEMORY ARRAYS. v1 fed an imageDatastore with a ReadFcn into
%      trainNetwork. That is fine for training, but it means every energy
%      measurement re-reads and re-resizes PNGs from disk inside the timed
%      window -- so a large share of the "inference energy" was actually
%      file I/O and JPEG/PNG decode, which pruning cannot affect. Images are
%      now decoded once, up front, into a uint8 array.
%
%      This alone may change the ChestX-ray energy result substantially.
%
%   3. FILE LOOKUP. v1 searched every images_* folder for every filename,
%      an O(N x folders) scan with an exist() call per candidate. The folder
%      contents are now indexed once into a map.
%
%   OPTIONS
%     .subsetSize  images to sample (default 2000)
%     .imageSize   [H W] (default [128 128])
%     .splitMode   "patient" (default) | "image"
%     .fractions   [train val test] (default [0.7 0.15 0.15])
%     .seed        RNG seed (default 1)

    arguments
        dataDir (1,:) char
        opts.subsetSize (1,1) double = 2000
        opts.imageSize  (1,2) double = [128 128]
        opts.splitMode  (1,:) char {mustBeMember(opts.splitMode,{'patient','image'})} = 'patient'
        opts.fractions  (1,3) double = [0.7 0.15 0.15]
        opts.seed       (1,1) double = 1
    end

    labelsCsv = fullfile(dataDir, 'Data_Entry_2017.csv');
    T = readtable(labelsCsv, 'ReadVariableNames', false, 'HeaderLines', 1);
    T.Properties.VariableNames = {'ImageIndex','FindingLabels','FollowUpNum', ...
        'PatientID','PatientAge','PatientGender','ViewPosition', ...
        'OriginalImageWidth','OriginalImageHeight', ...
        'OriginalImagePixelSpacingX','OriginalImagePixelSpacingY'};

    isNoFinding = strcmp(T.FindingLabels, 'No Finding');
    T.BinaryLabel = categorical(isNoFinding, [true false], {'NoFinding','Finding'});

    % ---- Index every image file once -------------------------------------
    folders = dir(fullfile(dataDir, 'images_*'));
    pathOf  = containers.Map('KeyType','char','ValueType','char');
    for f = 1:numel(folders)
        d = dir(fullfile(dataDir, folders(f).name, 'images', '*.png'));
        for i = 1:numel(d)
            if ~isKey(pathOf, d(i).name)
                pathOf(d(i).name) = fullfile(d(i).folder, d(i).name);
            end
        end
    end
    fprintf('ChestX-ray14: indexed %d image files\n', pathOf.Count);

    rng(opts.seed, 'twister');

    % ---- Sample the subset, respecting the split unit --------------------
    if strcmp(opts.splitMode, 'patient')
        pats = unique(T.PatientID);
        pats = pats(randperm(numel(pats)));
        % Take whole patients until the subset size is reached.
        keep = false(height(T),1); total = 0; p = 0;
        while total < opts.subsetSize && p < numel(pats)
            p = p + 1;
            m = T.PatientID == pats(p);
            keep = keep | m;
            total = total + sum(m);
        end
        sub = T(keep, :);
    else
        idx = randperm(height(T), min(opts.subsetSize, height(T)));
        sub = T(idx, :);
        warning('load_chestxray_dataset:ImageLevelSplit', ...
            ['splitMode="image" puts images from the same patient in both ' ...
             'train and test. Accuracy from this split is optimistically biased.']);
    end

    % ---- Resolve paths and decode once -----------------------------------
    n = height(sub);
    ok = false(n,1); paths = strings(n,1);
    for i = 1:n
        nm = sub.ImageIndex{i};
        if isKey(pathOf, nm), paths(i) = pathOf(nm); ok(i) = true; end
    end
    sub = sub(ok,:); paths = paths(ok); n = height(sub);
    fprintf('  resolved %d images on disk\n', n);
    if n == 0
        error('load_chestxray_dataset:NoImages', 'No image files found under %s', dataDir);
    end

    X = zeros([opts.imageSize 1 n], 'uint8');
    for i = 1:n
        im = imread(paths(i));
        if size(im,3) > 1, im = rgb2gray(im); end
        X(:,:,1,i) = imresize(im, opts.imageSize);
        if mod(i, 250) == 0, fprintf('  decoded %d/%d\n', i, n); end
    end
    Y = sub.BinaryLabel;

    % ---- Split ------------------------------------------------------------
    if strcmp(opts.splitMode, 'patient')
        pats  = unique(sub.PatientID);
        shP   = pats(randperm(numel(pats)));
        a = max(1, round(opts.fractions(1)*numel(shP)));
        b = max(a+1, round((opts.fractions(1)+opts.fractions(2))*numel(shP)));
        b = min(b, numel(shP)-1);
        trIdx = find(ismember(sub.PatientID, shP(1:a)));
        vaIdx = find(ismember(sub.PatientID, shP(a+1:b)));
        teIdx = find(ismember(sub.PatientID, shP(b+1:end)));
        info.numPatients = numel(pats);
    else
        sh = randperm(n);
        a = round(opts.fractions(1)*n);
        b = round((opts.fractions(1)+opts.fractions(2))*n);
        trIdx = sh(1:a); vaIdx = sh(a+1:b); teIdx = sh(b+1:end);
        info.numPatients = numel(unique(sub.PatientID));
    end

    XTr = X(:,:,:,trIdx); YTr = Y(trIdx);
    XVa = X(:,:,:,vaIdx); YVa = Y(vaIdx);
    XTe = X(:,:,:,teIdx); YTe = Y(teIdx);

    info.splitMode   = string(opts.splitMode);
    info.numImages   = n;
    info.classCounts = countcats(Y);
    info.classNames  = categories(Y);
    info.splitSizes  = [numel(YTr) numel(YVa) numel(YTe)];

    fprintf('  class balance: %s %d (%.2f%%), %s %d (%.2f%%)\n', ...
        info.classNames{1}, info.classCounts(1), info.classCounts(1)/n*100, ...
        info.classNames{2}, info.classCounts(2), info.classCounts(2)/n*100);
    fprintf('  images: %d train / %d val / %d test  (%d patients)\n', ...
        info.splitSizes, info.numPatients);

    if any(countcats(YTr)==0) || any(countcats(YVa)==0) || any(countcats(YTe)==0)
        warning('load_chestxray_dataset:MissingClass', ...
                'A class is absent from at least one split. Try another seed.');
    end
end
