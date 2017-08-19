function result = EvalGaussRelBlurDFDVar(im1, im2, crop_boundary_size, params)
%%%% WARNING: TO WORK WITH REAL IMAGES THE CURRENT FORMAT IS IM1 BEING FAR
%%%% IMAGE AND IM2 BEING NEAR IMAGE. THIS CODE ASSUMES IM1 IS NEAR
%%%% AND IM2 IS FAR. SO THE CALLER NEEDS TO SET IM1 AND IM2 APPROPRIATELY.
% continuous Gaussian relative blur
% take a pair of large images for a specific depth, partition and evaluate
% gaussian relative blur. In the end we want to estimate the variance and
% mean of the relative blur estimates.

boundary_cond = 'symmetric';
EPS = 0; %1e-6;
% LB = params.LB;
% UB = params.UB;
% estimate upper and lower bound of relative blur
zrange = [params.z0, params.z1];
r1_px = depth2blur(zrange, params.camera1, params.zNear);
r2_px = depth2blur(zrange, params.camera2, params.zFar);
RelBlurSqrRange = r2_px.^2 - r1_px.^2;
RelBlurRange = sign(RelBlurSqrRange) .* sqrt(abs(RelBlurSqrRange)); %/ sqrt(2); % TO CONVERT FROM DISK RAD TO GAUSS SPREAD
LB = min(RelBlurRange) - EPS;
UB = max(RelBlurRange) + EPS;
%%%%
PatchSize = params.PatchSize; %[201, 201];
Stride = params.Stride; %[50, 50];
SigmaC = params.SigmaC;

im1_seq = PartitionImage(im1, PatchSize, Stride);
im2_seq = PartitionImage(im2, PatchSize, Stride);
NPartitions = size(im1_seq, 3);
SigmaSeq = nan(1, NPartitions);

%if(~exist('crop_boundary_size', 'var'))
    crop_boundary_size = max(UB, LB); %100;
%end

MaskWindow = zeros(size(im1_seq(:,:,1)));
MaskWindow(crop_boundary_size+1:end-crop_boundary_size, ...
           crop_boundary_size+1:end-crop_boundary_size) = 1;
FVAL = nan(size(SigmaSeq));
for idxSeq = 1:NPartitions
    im1N = im1_seq(:,:,idxSeq) * 1e6;
    im2N = im2_seq(:,:,idxSeq) * 1e6;

    [~, tmp1, res1] = EstGaussianPSF(im2N, im1N, -EPS, -LB, boundary_cond, ...
                                     MaskWindow, SigmaC, params.GradObj);
    [~, tmp2, res2] = EstGaussianPSF(im1N, im2N, -EPS, UB, boundary_cond, ...
                                     MaskWindow, SigmaC, params.GradObj);
    SigmaSeq(idxSeq) = -tmp1;
    FVAL(idxSeq) = res1.FVAL;
    if(res2.FVAL < res1.FVAL)
        SigmaSeq(idxSeq) = tmp2;
        FVAL(idxSeq) = res2.FVAL;
    end
    %[~, SigmaSeq(idxSeq)] = EstGaussianPSF(im1N, im2N, LB, UB, boundary_cond, MaskWindow, SigmaC);
end
SigmaSeq = SigmaSeq * sqrt(2); % sqrt(2) to account for gaussian spread and pillbox radius
result.SigREstAll = SigmaSeq;
result.SigREstMean = mean(SigmaSeq);
result.SigREstStd = std(SigmaSeq);
result.NPartitions = NPartitions;

% Estimate all the other quantities from the relative blur
[sigma1, sigma2, alpha, beta, sigma1All, sigma2All, mask_ambiguity] = RelBlur2Blur(SigmaSeq, params)
result.Sig1EstAll = sigma1;
result.Sig2EstAll = sigma2;
result.Sig1EstMean = mean(sigma1);
result.Sig2EstMean = mean(sigma2);
result.Sig1EstStd = std(sigma1);
result.Sig2EstStd = std(sigma2);
result.alpha = alpha;
result.beta = beta;

Depth = Radius2Depth(params.camera1, params.zNear, sigma1);
Depth = Depth / 1e3;
InvDepth = 1./Depth;

result.Depth = Depth;
result.DepthMean = mean(Depth);
result.DepthStd = std(Depth);
result.DepthRMSE = sqrt(nanmean((Depth(:) - params.DepthGT).^2));

result.InvDepth = InvDepth;
result.InvDepthMean = mean(InvDepth);
result.InvDepthStd = std(InvDepth);
result.InvDepthRMSE = sqrt(nanmean((InvDepth(:) - 1/params.DepthGT).^2));
result.UB = UB;
result.LB = LB;
result.FVAL = FVAL;
% % compute mean depth for rel blur
% [~, I] = nanmin(Cost, [], 3);
% Depth = Z(I)  .* MaskWindow;
% result.LabelIdx = I .* MaskWindow;
% result.Depth = Depth;
% 
% InvDepth = 1./Depth;
% result.InvDepth = InvDepth;
% result.Cost = Cost;
