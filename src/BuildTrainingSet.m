function ImageDepthSet = BuildTrainingSet(Im_seq, KernelSet, params)
% train discriminative filters using different loss function and energy
% functions. This implementation doesn't assume convolution by default.
% I.e. instead of using convolution matrix this requires that the whole
% patch is used in the inner product with the filter. Therefore the patch
% size needs to be the same as the filter size.
% Also this version does not assume any specific loss function like the
% TrainDiscrimFiltersQuadLoss which assumes quadratic loss.

NDepth = size(KernelSet, 1);
NMaxTraining = inf;

if(isfield(params, 'NMaxTraining'))
    NMaxTraining = params.NMaxTraining;
end

if(isfield(params, 'KernelSetLabel'))
    KernelSetLabel = params.KernelSetLabel % label per kernel
else
    KernelSetLabel = 1:NDepth;
end

ImageDepthSet = cell(1, max(KernelSetLabel));
    
KR = params.KR;
KC = params.KC;

BlockSize = [KR, KC]; % block size for patches
BlockStep = ceil(min(KR, KC)/2);

if(isfield(params, 'BlockStep'))
    BlockStep = params.BlockStep;
end

fnPatchNormalization = @(x) (bsxfun(@rdivide, bsxfun(@minus, x, mean(x)), std(x)));
if(isfield(params, 'fnPatchNormalization') && ~isempty(params.fnPatchNormalization))
    fnPatchNormalization = params.fnPatchNormalization;
end

NKernelsPerDepth = size(KernelSet, 2);
fnTform = @(x) x;
if(isfield(params, 'fnTform') && ~isempty(params.fnTform))
    fnTform = params.fnTform;
end

fnCombine = @(x, y) ([x;y]);
if(isfield(params, 'fnCombine') && ~isempty(params.fnCombine))
    fnCombine = params.fnCombine;
end

BlurType = 'PSF';
if(isfield(params, 'BlurType') && ~isempty(params.BlurType))
    BlurType = params.BlurType;
end

fnBlur = @(image, kernel) blurImg(image, kernel, BlurType, 'symmetric');
if(isfield(params, 'fnBlur') && ~isempty(params.fnBlur))
    fnBlur = params.fnBlur;
end

% the following loop is for training using synthetic images
for itImg = 1:length(Im_seq)
    im0 = Im_seq{itImg};
    for idx = 1:NDepth
        % create defocus pair
        im1 = fnBlur(im0, KernelSet{idx, 1}); %blurImg(im0, KernelSet{idx, 1}, BlurType, 'symmetric');

        im1_seq = PartitionImage(im1, BlockSize, BlockStep);
        
        % take the top patches in terms of std
        im1_seq_v = reshape(im1_seq, prod(BlockSize), []);
        
        if(NKernelsPerDepth == 1)
            im_seq_v = std(im1_seq_v);
        else
           im2 = fnBlur(im0, KernelSet{idx, 2}); %blurImg(im0, KernelSet{idx, 2}, BlurType, 'symmetric'); 
           im2_seq = PartitionImage(im2, BlockSize, BlockStep);
           im2_seq_v = reshape(im2_seq, prod(BlockSize), []);
           im_seq_v = min([std(im1_seq_v); std(im2_seq_v)]);
           im2_seq_v = fnPatchNormalization(im2_seq_v);
        end
        
        
        im1_seq_v = fnPatchNormalization(im1_seq_v);

        [~, I] = sort(im_seq_v, 2, 'descend');
        if(NKernelsPerDepth == 1)
            AA = fnTform(im1_seq_v(:,I(1:NMaxTraining)));
        else
            AA = fnCombine(im1_seq_v(:,I(1:NMaxTraining)), im2_seq_v(:,I(1:NMaxTraining)));
        end

        if(itImg == 1)
            ImageDepthSet{KernelSetLabel(idx)} = AA;
        else
            ImageDepthSet{KernelSetLabel(idx)} = [ImageDepthSet{KernelSetLabel(idx)}, AA];
        end

    end
end

%result.ImageDepthSet = ImageDepthSet;