function result = TrainDiscrimFiltersQuadLoss(Im_seq, KernelSet, params)

NDepth = size(KernelSet, 1);
NMaxTraining = inf;

if(isfield(params, 'NMaxTraining'))
    NMaxTraining = params.NMaxTraining;
end

ImageDepthSet = cell(1, NDepth);
    
KR = 13;
KC = 13;
if(isfield(params, 'KR'))
    KR = params.KR;
end
if(isfield(params, 'KC'))
    KC = params.KC;
end

BlockSize = [101, 101]; %KR * 5 * ones(1, 2); %
BlockStep = 50;
if(isfield(params, 'BlockSize'))
    BlockSize = params.BlockSize;
end

if(isfield(params, 'BlockStep'))
    BlockStep = params.BlockStep;
end
for itImg = 1:length(Im_seq)
    im0 = Im_seq{itImg};
    for idx = 1:NDepth
        % create defocus pair
        im1 = blurImg(im0, KernelSet{idx, 1}, 'PSF', 'symmetric');
        im2 = blurImg(im0, KernelSet{idx, 2}, 'PSF', 'symmetric');

        im1_seq = PartitionImage(im1, BlockSize, BlockStep);
        im2_seq = PartitionImage(im2, BlockSize, BlockStep);
        AA = [];
        NSeq = size(im1_seq, 3);
        for seqIdx = 1:min(NMaxTraining, NSeq)
            im1c = im1_seq(:,:,seqIdx);
            im2c = im2_seq(:,:,seqIdx);

%             im1c = im1c/mean(im1c(:)) - 1;
%             im1c = im1c / std(im1c(:));
%             im2c = im2c / mean(im2c(:)) - 1;
%             im2c = im2c / std(im2c(:));
            im1c = (im1c - mean(im1c(:))) / var(im1c(:));
            im2c = (im2c - mean(im2c(:))) / var(im2c(:));
            
            A1 = im2convmtx(im1c, KR, KC);
            A2 = im2convmtx(im2c, KR, KC);
            A = [A1, -A2];
            AAp = A'*A; % only for quadratic distance
            
            if(seqIdx == 1)
                AA = AAp;
            else
                AA = AA + AAp;
            end
        end
        if(itImg == 1)
            ImageDepthSet{idx} = AA;
        else
            ImageDepthSet{idx} = ImageDepthSet{idx} + AA;
        end

    end
end
%
%Res = EstDefocusFilterPairs(ImageDepthSet, {})

%params.X0 = X0(:,end); %(17*338+1:22*338,end);
%
params.ERR_CONST = 1; %1e4;
params.DerivCheck = 'off'; %'on';
params.useVersion = 2;
Res = EstDefocusFilterPairsCOpt(ImageDepthSet, params)

%Res = EstDefocusFilterPairsBasicQCQP(ImageDepthSet, params)

%     params = {}
%     params.ERR_CONST = 10;
%     ImageDepthSetSingleFilterSmall = ImageDepthSetSingleFilter(12:2:20);
%     Res = EstDefocusFilterPairsBasicQCQP(ImageDepthSetSingleFilterSmall, params, 1)    
%     Res1 = EstDefocusFilterPairsCOpt(ImageDepthSetSingleFilterSmall, params, 1)
%%
result.Filters = Res.Filters;
result.Res = Res;