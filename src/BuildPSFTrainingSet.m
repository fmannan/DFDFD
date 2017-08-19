function result = BuildPSFTrainingSet(KernelSet, params)
% Similar to BuildTrainingSet but input is just the PSFs and output is the
% convolution matrix for size (KR x KC) x (KR x KC) which is used for
% training.

% train discriminative filters using different loss function and energy
% functions. This implementation doesn't assume convolution by default.
% I.e. instead of using convolution matrix this requires that the whole
% patch is used in the inner product with the filter. Therefore the patch
% size needs to be the same as the filter size.
% Also this version does not assume any specific loss function like the
% TrainDiscrimFiltersQuadLoss which assumes quadratic loss.

NDepth = size(KernelSet, 1);

KernelMatSet = cell(1, NDepth);
    
KR = params.KR;
KC = params.KC;

% the following loop is for training using synthetic images
for itImg = 1:length(Im_seq)
    im0 = Im_seq{itImg};
    for idx = 1:NDepth
        % create defocus pair
        im1 = blurImg(im0, KernelSet{idx, 1}, 'PSF', 'symmetric');

        im1_seq = PartitionImage(im1, BlockSize, BlockStep);
        
        % take the top patches in terms of std
        im1_seq_v = reshape(im1_seq, prod(BlockSize), []);
        
        if(NKernelsPerDepth == 1)
            im_seq_v = std(im1_seq_v);
        else
           im2 = blurImg(im0, KernelSet{idx, 2}, 'PSF', 'symmetric'); 
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
            ImageDepthSet{idx} = AA;
        else
            ImageDepthSet{idx} = [ImageDepthSet{idx}, AA];
        end

    end
end

result.ImageDepthSet = ImageDepthSet;