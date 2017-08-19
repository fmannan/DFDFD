function result = EstBETDFD(im1, im2, PSFSet, expK, Z, BK, crop_boundary_size)
nDepths = size(PSFSet, 1);

Cost = nan(size(im1, 1), size(im1, 2), nDepths);

boundary_cond = 'symmetric';

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
for idx = 1:nDepths
   Cost(:,:,idx) = conv2(abs(imfilter(im1, PSFSet{idx, 2}, boundary_cond) ...
                - imfilter(im2, PSFSet{idx, 1}, boundary_cond)).^expK, BK, 'same') .* MaskWindow;
end

[~, I] = nanmin(Cost, [], 3);
Depth = Z(I) .* MaskWindow;
result.LabelIdx = I .* MaskWindow;
result.Depth = Depth;
InvDepth = 1./Depth;
result.InvDepth = InvDepth;

%%%%
result.Cost = Cost;
