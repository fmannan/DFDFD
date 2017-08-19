function result = EstDeconvWnrGenDFD(im1, im2, PSFSet, expK, Z, BK, crop_boundary_size)
nDepths = size(PSFSet, 1);

Cost = nan(size(im1, 1), size(im1, 2), nDepths);

%boundary_cond = 'symmetric';

if(~exist('crop_boundary_size', 'var'))
    crop_boundary_size = 0; %100;
end
if(~exist('BK', 'var')) % cost smoothing kernel
    BK = 1;
end
MaskWindow = nan(size(im1));
MaskWindow(crop_boundary_size+1:end-crop_boundary_size, ...
           crop_boundary_size+1:end-crop_boundary_size) = 1;
%%%%
[H, W] = size(im1);
I1 = fft2(im2double(im1));
I2 = fft2(im2double(im2));
lambda = 1e-6;
Image = nan(H, W, nDepths);
for idx = 1:nDepths
    K1 = fft2(PSFSet{idx, 1}, H, W);
    K2 = fft2(PSFSet{idx, 2}, H, W);
    [im_est, diff_cost] = DeconvWnrGenFFT(I1, I2, K1, K2, lambda);
    Cost(:,:,idx) = conv2(sum(abs(diff_cost(:,:,1:2)).^expK, 3), BK, 'same') .* MaskWindow;
    Image(:,:,idx) = im_est;
end

[~, I] = nanmin(Cost, [], 3);
Depth = Z(I) .* MaskWindow;
result.LabelIdx = I .* MaskWindow;
result.Depth = Depth;
InvDepth = 1./Depth;
result.InvDepth = InvDepth;

%%%%
result.Cost = Cost;
result.Image = Image;