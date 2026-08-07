function [dlnet, info] = to_dlnetwork(net)
%TO_DLNETWORK Normalise any trained network to a bare dlnetwork.
%
%   [dlnet, info] = TO_DLNETWORK(net)
%
%   WHY THIS EXISTS
%   ---------------
%   In the v1 harness the baseline arm was a SeriesNetwork (returned by
%   trainNetwork) while the pruned arm was a dlnetwork (returned by
%   taylorPrunableNetwork -> dlnetwork). These two object types dispatch
%   predict() through completely different execution paths, and only the
%   dlnetwork arm paid the cost of wrapping its input in a dlarray on every
%   call. The measured difference between the arms therefore confounded
%   "effect of pruning" with "effect of changing framework path".
%
%   Every arm of an energy comparison must be the same class, consuming the
%   same input type, before a single joule is attributed to compression.
%   This function enforces that.
%
%   Accepts SeriesNetwork, DAGNetwork, LayerGraph, or dlnetwork.
%   Output layers (classification / regression) are stripped, because
%   dlnetwork does not carry them and their presence would otherwise be an
%   additional asymmetry between arms.
%
%   INFO fields:
%     .originalClass   class of the input network
%     .removedLayers   names of output layers stripped
%     .wasConverted    true if a conversion actually happened

    arguments
        net
    end

    info = struct('originalClass', class(net), 'removedLayers', {{}}, ...
                  'wasConverted', false);

    if isa(net, 'dlnetwork')
        dlnet = net;
        if ~dlnet.Initialized
            error('to_dlnetwork:Uninitialized', ...
                  'dlnetwork is not initialized; cannot be measured.');
        end
        return;
    end

    if isa(net, 'nnet.cnn.LayerGraph')
        lgraph = net;
    elseif isa(net, 'SeriesNetwork') || isa(net, 'DAGNetwork')
        lgraph = layerGraph(net);
    else
        error('to_dlnetwork:UnsupportedType', ...
              'Unsupported network class "%s".', class(net));
    end

    % Strip any terminal output layer. Match on class name rather than a
    % hard-coded layer name, since the name varies ('output', 'classoutput',
    % ...) depending on whether it was explicitly set at construction.
    layerClasses = arrayfun(@class, lgraph.Layers, 'UniformOutput', false);
    isOutputLayer = contains(layerClasses, 'OutputLayer');

    if any(isOutputLayer)
        names = {lgraph.Layers(isOutputLayer).Name};
        lgraph = removeLayers(lgraph, names);
        info.removedLayers = names;
    end

    dlnet = dlnetwork(lgraph);
    info.wasConverted = true;
end
