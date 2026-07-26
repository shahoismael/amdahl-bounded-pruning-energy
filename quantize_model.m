function quantizedNet = quantize_model(net, XCalibration)
% QUANTIZE_MODEL Applies post-training INT8 quantization using the
% Deep Learning Toolbox Quantization support.
%
%   quantizedNet = quantize_model(net, XCalibration)
%   XCalibration: representative data sample used to calibrate ranges

    quantObj = dlquantizer(net, 'ExecutionEnvironment', 'CPU');
    calibrate(quantObj, XCalibration);
    quantizedNet = quantObj; % use validate()/predict() via this object

    fprintf('Quantization (INT8) complete.\n');
end
