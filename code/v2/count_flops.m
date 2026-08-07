function report = count_flops(dlnet, inputSize, dataFormat)
%COUNT_FLOPS Multiply-accumulate and FLOP counts for a dlnetwork.
%
%   report = COUNT_FLOPS(dlnet, inputSize)
%   report = COUNT_FLOPS(dlnet, inputSize, dataFormat)
%
%   WHY THIS EXISTS
%   ---------------
%   The Green AI literature this study engages with (Schwartz et al. 2020 and
%   everything downstream) reports FLOPs as *the* hardware-independent
%   efficiency metric. v1 of this study reported only filter and parameter
%   counts and explicitly declined to compute FLOPs. That is an open
%   invitation for a reviewer to ask why the field's standard metric was
%   omitted from a paper whose entire argument is about that metric.
%
%   For a two-conv-layer network the computation is trivial, so there is no
%   defensible reason not to report it.
%
%   METHOD
%   ------
%   Layer output sizes are obtained empirically by a single forward pass with
%   all layers marked as outputs, rather than by re-deriving padding and
%   stride arithmetic by hand. This is robust to any architecture change.
%
%   Convention: MACs are counted for convolution and fully connected layers
%   (the field's usual "FLOPs" figure is 2 x MACs). Elementwise layers
%   (BN, ReLU) and pooling are counted separately and reported but excluded
%   from the headline number, since practice varies and mixing them makes
%   cross-study comparison harder.

    arguments
        dlnet      (1,1) dlnetwork
        inputSize  (1,:) double
        dataFormat (1,:) char = 'SSCB'
    end

    % ---- Empirically measure every layer's output size -------------------
    dummy = dlarray(zeros([inputSize 1], 'single'), dataFormat);

    layerNames = {dlnet.Layers.Name};
    outSizes   = containers.Map('KeyType', 'char', 'ValueType', 'any');

    for i = 1:numel(layerNames)
        nm = layerNames{i};
        try
            Y = predict(dlnet, dummy, 'Outputs', nm);
            outSizes(nm) = size(extractdata(Y));
        catch
            % Layers that are not valid output points (e.g. input layers on
            % some releases) are skipped; they contribute no MACs anyway.
        end
    end

    % ---- Walk layers and accumulate --------------------------------------
    convMACs = 0; fcMACs = 0; elementwiseOps = 0; poolOps = 0;
    rows = cell(0, 4);   % width fixed up front: (end+1,:) on a bare {} errors

    for i = 1:numel(dlnet.Layers)
        L  = dlnet.Layers(i);
        nm = L.Name;
        if ~isKey(outSizes, nm), continue; end
        osz = outSizes(nm);

        macs = 0; kind = class(L);

        if isa(L, 'nnet.cnn.layer.Convolution2DLayer')
            % Weights: [FH FW Cin Cout]. Using the actual weight tensor makes
            % this correct after pruning, when NumFilters/NumChannels
            % properties may not have been refreshed.
            w        = size(L.Weights);
            spatial  = prod(osz(1:2));
            macs     = spatial * w(4) * w(3) * w(1) * w(2);
            convMACs = convMACs + macs;
            kind     = 'conv2d';

        elseif isa(L, 'nnet.cnn.layer.FullyConnectedLayer')
            w      = size(L.Weights);          % [out in]
            macs   = w(1) * w(2);
            fcMACs = fcMACs + macs;
            kind   = 'fc';

        elseif isa(L, 'nnet.cnn.layer.BatchNormalizationLayer')
            elementwiseOps = elementwiseOps + 2 * prod(osz);
            kind = 'batchnorm';

        elseif isa(L, 'nnet.cnn.layer.ReLULayer')
            elementwiseOps = elementwiseOps + prod(osz);
            kind = 'relu';

        elseif isa(L, 'nnet.cnn.layer.MaxPooling2DLayer')
            poolOps = poolOps + prod(osz) * prod(L.PoolSize);
            kind = 'maxpool';
        end

        rows(end+1, :) = {nm, kind, mat2str(osz), macs}; %#ok<AGROW>
    end

    totalMACs = convMACs + fcMACs;

    report = struct();
    report.perLayer = cell2table(rows, ...
        'VariableNames', {'Layer', 'Type', 'OutputSize', 'MACs'});
    report.convMACs       = convMACs;
    report.fcMACs         = fcMACs;
    report.totalMACs      = totalMACs;
    report.totalFLOPs     = 2 * totalMACs;
    report.elementwiseOps = elementwiseOps;
    report.poolOps        = poolOps;
    report.learnables     = sum(cellfun(@numel, dlnet.Learnables.Value));

    fprintf('--- FLOP report -------------------------------------------\n');
    disp(report.perLayer);
    fprintf('  conv MACs         : %12d\n', convMACs);
    fprintf('  fc   MACs         : %12d\n', fcMACs);
    fprintf('  TOTAL MACs        : %12d\n', totalMACs);
    fprintf('  TOTAL FLOPs (2xM) : %12d\n', report.totalFLOPs);
    fprintf('  learnable params  : %12d\n', report.learnables);
    fprintf('-----------------------------------------------------------\n');
end
