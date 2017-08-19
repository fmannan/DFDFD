function res = EstDepthDiscrimFilterConv(im1, im2, params)
% estimate depth at every pixel from discriminative filters

K = params.KernelSize;

%im1seq = PartVecNormImage(im1, K, params.fnNormalize);
%im2seq = PartVecNormImage(im2, K, params.fnNormalize);

%imseq = params.fnCombine(im1seq, im2seq);

% preprocess the filters so that the energy function can work with it
W = params.W;
%A = params.A;
%Cost = params.fnEnergy(imseq, W, A);
NDepths = size(W, 3);
NFilters = size(W, 1);
Cost = zeros(size(im1, 1), size(im1, 2), NDepths);
HSmooth = [];
if(isfield(params, 'HSmooth'))
    HSmooth = params.HSmooth;
end
crop_boundary_size = 0;
if(isfield(params, 'crop_boundary_size'))
    crop_boundary_size = params.crop_boundary_size;
end

MaskWindow = nan(size(im1));
MaskWindow(crop_boundary_size+1:end-crop_boundary_size, ...
           crop_boundary_size+1:end-crop_boundary_size) = 1;

for idx = 1:NDepths
    for filterIdx = 1:NFilters
        F1 = rot90(reshape(W(filterIdx, 1:K*K, idx), K, K), 2);
        cost1 = conv2( im1, F1, 'same' );
        cost2 = 0;
        if(~isempty(im2))
            F2 = rot90(reshape(W(filterIdx, K*K+1:end, idx), K, K), 2);
            cost2 = conv2(im2, F2, 'same');
        end
        Cost(:,:,idx) = Cost(:,:,idx) + (cost1 + cost2).^2;
    end
    Cost(:,:,idx) = log(Cost(:,:,idx));
    if(~isempty(HSmooth))
       Cost(:,:,idx) = imfilter(Cost(:,:,idx), HSmooth, 'symmetric', 'same'); 
    end
    Cost(:,:,idx) = Cost(:,:,idx) .* MaskWindow;
end
[~, LabelIdx] = nanmin(Cost, [], 3);

LabelIdx = reshape(LabelIdx, size(im1));
res.LabelIdx = LabelIdx;
res.Cost = Cost;

if(isfield(params, 'Z'))
    res.Depth = params.Z(LabelIdx) .* MaskWindow;
    res.InvDepth = 1./res.Depth;
end


