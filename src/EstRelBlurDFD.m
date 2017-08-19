function result = EstRelBlurDFD(im1, im2, PSFSet, SGN, expK, Z, BK, crop_boundary_size)
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
    if(SGN(idx) < 0) % s1 > s2
        HRel = PSFSet{idx, 4};
        imest = imfilter(im2, HRel, boundary_cond);
        diff = im1 - imest;
    else
        HRel = PSFSet{idx, 3};
        imest = imfilter(im1, HRel, boundary_cond);
        diff = im2 - imest;
    end
    
   Cost(:,:,idx) = conv2(abs(diff).^expK, BK, 'same') .* MaskWindow;
end
% compute mean depth for rel blur
[~, I] = nanmin(Cost, [], 3);
Depth = Z(I)  .* MaskWindow;
result.LabelIdx = I .* MaskWindow;
result.Depth = Depth;

InvDepth = 1./Depth;
result.InvDepth = InvDepth;
result.Cost = Cost;
