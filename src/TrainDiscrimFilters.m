function result = TrainDiscrimFilters(ImageDepthSet, params) %(Im_seq, KernelSet, params)
% train discriminative filters using different loss function and energy
% functions. This implementation doesn't assume convolution by default.
% I.e. instead of using convolution matrix this requires that the whole
% patch is used in the inner product with the filter. Therefore the patch
% size needs to be the same as the filter size.
% Also this version does not assume any specific loss function like the
% TrainDiscrimFiltersQuadLoss which assumes quadratic loss.

Res = EstDefocusFilterIterUnc(ImageDepthSet, params);
result.W = Res.W; %Filters;
if(isfield(params, 'bOptA') && params.bOptA)
    result.A0 = Res.A0;
    result.A1 = Res.A1;
    result.A1init = Res.A1init;
else
    result.A = Res.A;
end
result.Res = Res;

