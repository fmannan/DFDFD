function result = ComputeErrorStat(Depth, TrueDepth, crop_boundary_size)

MaskWindow = nan(size(Depth));
MaskWindow(crop_boundary_size:end-crop_boundary_size, ...
           crop_boundary_size:end-crop_boundary_size) = 1;

Depth = Depth .* MaskWindow;
result.MeanDepth = nanmean(Depth(:));
result.StdDepth = nanstd(Depth(:));
result.DepthRMSE = sqrt(nanmean((Depth(:) - TrueDepth).^2));

InvDepth = 1./Depth;
result.MeanInvDepth = nanmean(InvDepth(:));
result.StdInvDepth = nanstd(InvDepth(:));
result.InvDepthRMSE = sqrt(nanmean((InvDepth(:) - 1/TrueDepth).^2));