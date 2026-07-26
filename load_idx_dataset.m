function [images, labels] = load_idx_dataset(imagesFile, labelsFile)
% LOAD_IDX_DATASET Reads IDX-format image/label files (MNIST / Fashion-MNIST).
%
%   [images, labels] = load_idx_dataset(imagesFile, labelsFile)
%
%   imagesFile : path to the *-images-idx3-ubyte file (unzipped)
%   labelsFile : path to the *-labels-idx1-ubyte file (unzipped)
%
%   images : [H W 1 N] uint8 array, ready for imageDatastore-style use
%   labels : categorical vector, length N

    % ---- Read images ----
    fid = fopen(imagesFile, 'r', 'b'); % big-endian
    if fid == -1
        error('Could not open images file: %s', imagesFile);
    end
    magic = fread(fid, 1, 'int32');
    if magic ~= 2051
        error('Invalid magic number in images file (expected 2051, got %d)', magic);
    end
    numImages = fread(fid, 1, 'int32');
    numRows   = fread(fid, 1, 'int32');
    numCols   = fread(fid, 1, 'int32');
    raw = fread(fid, numRows * numCols * numImages, 'uint8=>uint8');
    fclose(fid);

    images = reshape(raw, [numCols, numRows, numImages]);
    images = permute(images, [2 1 3]); % fix row/col order
    images = reshape(images, [numRows, numCols, 1, numImages]);

    % ---- Read labels ----
    fid = fopen(labelsFile, 'r', 'b');
    if fid == -1
        error('Could not open labels file: %s', labelsFile);
    end
    magic = fread(fid, 1, 'int32');
    if magic ~= 2049
        error('Invalid magic number in labels file (expected 2049, got %d)', magic);
    end
    numLabels = fread(fid, 1, 'int32');
    rawLabels = fread(fid, numLabels, 'uint8=>uint8');
    fclose(fid);

    labels = categorical(double(rawLabels));

    fprintf('Loaded %d images (%dx%d) and %d labels.\n', numImages, numRows, numCols, numLabels);
end
