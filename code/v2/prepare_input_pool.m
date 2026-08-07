function pool = prepare_input_pool(X, batchSize, numBatches, dataFormat, seed)
%PREPARE_INPUT_POOL Pre-build a fixed set of dlarray minibatches.
%
%   pool = PREPARE_INPUT_POOL(X, batchSize, numBatches, dataFormat, seed)
%
%   WHY THIS EXISTS
%   ---------------
%   The v1 timing loop did three things per iteration that had nothing to do
%   with the network under test:
%       idx     = randi(size(XTest,4));      % RNG call
%       Xsample = XTest(:,:,:,idx);          % strided copy out of a big array
%       Xsample = dlarray(single(Xsample));  % allocation + type conversion
%                                            % (pruned arm ONLY)
%   All three are charged to the measured window. The third was charged to
%   only one of the two arms, which is a systematic bias, not noise.
%
%   Building every batch once, up front, means the timed region contains
%   exactly one thing: predict(). The same pool object is then handed to
%   every arm, so the input path is byte-identical across conditions.
%
%   The pool is deliberately finite and cycled rather than regenerated, so
%   that cache behaviour is also identical across arms.

    arguments
        X
        batchSize  (1,1) double {mustBePositive, mustBeInteger}
        numBatches (1,1) double {mustBePositive, mustBeInteger}
        dataFormat (1,:) char = 'SSCB'
        seed       (1,1) double = 0
    end

    rng(seed, 'twister');   % pool contents fixed across arms and across runs

    nObs = size(X, 4);
    if nObs < batchSize
        error('prepare_input_pool:TooFewObservations', ...
              'Need at least batchSize=%d observations, have %d.', batchSize, nObs);
    end

    pool = cell(numBatches, 1);
    for b = 1:numBatches
        idx = randperm(nObs, batchSize);
        pool{b} = dlarray(single(X(:, :, :, idx)), dataFormat);
    end
end
