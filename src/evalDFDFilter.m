close all
clear
clc

addpath ../
addpath ../DFD

imgName = 'D42_512x512.png'; %'cameraman.tif'; %'brown_noise_pattern_512x512.png'
im = mean(im2double(imread(imgName)), 3);

figure
imshow(im)

%load('PSFSet_Disk_DFilter7x7_Scale10.mat')
load('PSFSet_Disk_DFilter9x9_Scale5.mat')
nDepth = length(Z)


Cost0 = nan(nDepth, nDepth);

for idx = 1:nDepth %
    im1 = blurImg(im, PSFSet{idx, 1}, 'PSF', 'symmetric');
    im2 = blurImg(im, PSFSet{idx, 2}, 'PSF', 'symmetric');
    im1 = (im1 - mean(im1(:))) / var(im1(:));
    im2 = (im2 - mean(im2(:))) / var(im2(:));
%     figure
%     subplot(2, 2, 1); imshow(im1, []);
%     subplot(2, 2, 2); imshow(im2, []);
%     subplot(2, 2, 3:4); plot(im1(256,:), 'r'); hold on; plot(im2(256, :), 'g')
    for idx2 = 1:nDepth
        Cost0(idx, idx2) = sum(reshape(abs(blurImg(im1, PSFSet{idx2, 3}, 'PSF', 'symmetric') - blurImg(im2, PSFSet{idx2, 4}, 'PSF', 'symmetric')), 1, []));
    end
end
    
figure
imagesc(Cost0) 

%%
BlockSize = [101, 101]; %KR * 5 * ones(1, 2); %
BlockStep = 50;

NMaxTraining = 10;

CostAll = nan(nDepth, nDepth, NMaxTraining);
CostBETAll = nan(nDepth, nDepth, NMaxTraining);

for idx = 1:nDepth %
    im1 = blurImg(im, PSFSet{idx, 1}, 'PSF', 'symmetric');
    im2 = blurImg(im, PSFSet{idx, 2}, 'PSF', 'symmetric');

    im1_seq = PartitionImage(im1, BlockSize, BlockStep);
    im2_seq = PartitionImage(im2, BlockSize, BlockStep);
%     figure
%     subplot(2, 2, 1); imshow(im1, []);
%     subplot(2, 2, 2); imshow(im2, []);
%     subplot(2, 2, 3:4); plot(im1(256,:), 'r'); hold on; plot(im2(256, :), 'g')
    for seqIdx = 1:NMaxTraining
        im1c = im1_seq(:,:,seqIdx);
        im2c = im2_seq(:,:,seqIdx);

        im1c = (im1c - mean(im1c(:))) / var(im1c(:));
        im2c = (im2c - mean(im2c(:))) / var(im2c(:));
            
            
        for idx2 = 1:nDepth
            CostAll(idx, idx2, seqIdx) = sum(reshape(abs(blurImg(im1c, PSFSet{idx2, 3}, 'PSF', 'symmetric') - blurImg(im2c, PSFSet{idx2, 4}, 'PSF', 'symmetric')), 1, []));
            CostBETAll(idx, idx2, seqIdx) = sum(reshape(abs(blurImg(im1_seq(:,:,seqIdx), PSFSet{idx2, 2}, 'PSF', 'symmetric') - blurImg(im2_seq(:,:,seqIdx), PSFSet{idx2, 1}, 'PSF', 'symmetric')), 1, []));
        end
    end
end
Cost = mean(CostAll, 3); 
figure
imagesc(Cost) 

CostBET = mean(CostBETAll, 3);
figure
imagesc(CostBET)
%% eval 2 (quick test)
addpath ../DFD
addpath ../
addpath ../Normalization/
clc
clear

%DataFilename = 'FullRes_m200_cs1__1_10_0_0_10000_0_0_0_0_genaffine_tp0p25_sn1_g1.mat'; %'FullRes_m200_cs1__1_10_100_10000_10000_0_0_1.000000e-02_1.000000e-02_genaffine_tp0p25_sn1_g1.mat'; % 'FullRes_m200_cs1__1_10_100_10000_10000_1_0_1.000000e-01_1.000000e-01_affine_tp0p25_sn1_g1.mat'; %'FullRes_m20_cs1_100_expd100_expc100_affine_ns0p25_sn1.mat'; %'FullRes_m200_cs1_100_expd100_expc100_affine_ns0p25_sn10_g10.mat'; %'FullRes_m12_cs10_1_1_1e3_0_1e2_0_objfnperfilter_ns0p25_scalednorm10.mat'; %'FullRes_m12_cs10_1_10_1e3_10_1e2_0_objfnperfilter_ns0p25_scalednorm10.mat'; %'FullRes_m12_cs10_1_0_0_1_1e4_10_objv2_ns0p25_scalednorm10.mat'; %'FullRes_m12_cs10_1_0_0_1_1e4_10_objv2_ns0p25.mat'; %'FullRes_m12_cs10_1_1e4_0_1_objv2_randinit.mat'; %'FullRes_m12_cs10_1_1e4_0_1_0_10_objv2_ns0p25.mat'; %'FullRes_m1_cs10_10_1e3_0_2_objfnv2_tpinit0p25.mat'; % 'FullRes_m1_10_1e4_0_2_objfnv2.mat' %'FullRes_m1_10_1e3_0_10.mat' %'FullRes_Large_m10_10_1eM5.mat' 'FullRes_Large_vectest.mat' %res = load('FullRes.mat');
DataFilename = 'FullRes_m12_cs10__100_10_1_10000_10000_1_0_0_0_genMinL1_im1_tp0p25_sn1_g1_depth6.mat' %'FullRes_m30_cs10__100_10_1_10000_10000_1_0_0_0_genMinL1_im1_tp0p25_sn1_g1_depth_6.mat' %'FullRes_m20_cs1__1_10_1_10000_10000_1_0_0_0_genMinL1_im1_randinit_sn1_g1_run2.mat';
DataSetRes = load(DataFilename); 
training_set = BuildTrainingSet(DataSetRes.Im_seq, DataSetRes.KernelSetDisk, DataSetRes.params);
%%
KSet = DataSetRes.KernelSetDisk; %(:,1); %(:,2);
tparams = DataSetRes.params;
tparams.TruncatePercentage = 1; %.25;
tparams.NMaxTraining = 100;
tparams.fnPatchNormalization =  @(x) x;
tparams.fnTform = []; %@(x) 10 * x.^2;
tparams.fnCombine = @(x, y) [x;y]; %[x + y]; %[(x + y) ; (x - y)] %10 * (x.^2 + y.^2) %x.*y; %(x + y).^2 - (x - y).^2 %(x.^2 + y.^2); %@minus; %@plus;

NoiseStd = 1; %.01;

Im_seq_noise1 = DataSetRes.Im_seq;
Im_seq_noise2 = DataSetRes.Im_seq;
for it = 1:size(Im_seq_noise1, 2)
   Im_seq_noise1{it} = Im_seq_noise1{it} + NoiseStd * randn(size(Im_seq_noise1{it}));
   Im_seq_noise2{it} = Im_seq_noise2{it} + NoiseStd * randn(size(Im_seq_noise2{it}));
end
training_set_1 = BuildTrainingSet(Im_seq_noise1, KSet, tparams);
%
NSFilterRes = TrainNullSpaceFilters(Im_seq_noise2, KSet, tparams);
%
%tparams.fnNullSpace = @findNullSpaceFiltersPerDepth;
%NSFilterRes = TrainNullSpaceFilters(DataSetRes.Im_seq, KSet, tparams);
ImageDepthSet = training_set_1.ImageDepthSet;
NClasses = length(ImageDepthSet)
NFilters = tparams.NFilters;

[DataDim, ExamplesPerClass] = size(ImageDepthSet{1});
X = reshape(cell2mat(ImageDepthSet), [DataDim, ExamplesPerClass, NClasses]);
X_flat = reshape(X, [size(X, 1), size(X, 2) * size(X, 3)]);
%X_flat_noise = X_flat + NoiseStd * randn(size(X_flat));
[~, y] = meshgrid(1:size(X, 2), 1:size(X, 3)); %nan(1, NTrain);
                                               %%round((NClasses
                                               %- 1) * rand(1,
                                               %NTrain)) + 1
y = reshape(y', 1, []);
YmatIdx = sub2ind([length(y), max(y)], 1:length(y), y);
Ymat = zeros(length(y), max(y));
Ymat(YmatIdx) = 1;
Ymat = Ymat * diag(1./sum(Ymat)); % normalize

A = zeros(NClasses, NFilters * NClasses);
for i = 1:size(A, 1)
    A(i,(i - 1) * NFilters + 1:(i * NFilters)) = 1; 
end
%
WNS_mod = convertFilter3Dto2DFormat(NSFilterRes.FilterVecs);
%
fnCostEval = @L1CostVec; % %@QuadraticCostVec; %@MinL1CostVec; %@AffineCostVec; %
Cost = fnCostEval(X_flat, WNS_mod, A);
CostSum = Cost * Ymat;
figure; imagesc(CostSum)
figure; surf(CostSum); shading interp

% %
% %fnCostEval = @L1CostVec; %@MinL1CostVec; %@AffineCostVec; %
% CostSqr = fnCostEval(X_flat, WNS_mod.^2, A);
% CostSumSqr = CostSqr * Ymat;
% figure; imagesc(CostSumSqr)
% figure; surf(CostSumSqr); shading interp
% fnCostEval = @L1CostVec; %@MinL1CostVec; %@AffineCostVec; %
% Cost_noise = fnCostEval(X_flat_noise, WNS_mod, A);
% CostSum_noise = Cost_noise * Ymat;
% figure; imagesc(CostSum_noise)
% figure; surf(CostSum_noise); shading interp

[loss, dLdE, loss_per_label] = HingeLossPerLabel(Cost, tparams.M0, tparams.CorrScale, Ymat);
%[loss_noise, dLdE, loss_per_label] = HingeLossPerLabel(Cost_noise, tparams.M0, tparams.CorrScale, Ymat);
loss
%loss_noise
%%
params = DataSetRes.params;
if(~isfield(params, 'CorrScale'))
    params.CorrScale = 1;
end
%% test using training set
ImageDepthSet = training_set.ImageDepthSet;
NClasses = length(ImageDepthSet)
[DataDim, ExamplesPerClass] = size(ImageDepthSet{1});
X = reshape(cell2mat(ImageDepthSet), [DataDim, ExamplesPerClass, NClasses]);

fnEnergy = @(x, W) QuadraticCost(x, W);
fnLoss = @(a, b, c, d, e) HingeLoss(a, b, c, d, e);
% build W
if(~isfield(DataSetRes, 'W0') && isfield(DataSetRes, 'Dataset'))
    W_NS = DataSetRes.Dataset.W0;
else
    W_NS = DataSetRes.W0;
end
W = DataSetRes.resultD.Res.x;
W0 = params.W0;
NFilters = size(W_NS, 1)
Lambdas = DataSetRes.params.LAMBDAS; %[10 res.params.LAMBDAS, 10]
m = DataSetRes.params.M0

%
CostNS = zeros(NClasses, NClasses);
Cost = zeros(NClasses, NClasses);
CostNS_norm_perex = zeros(NClasses, NClasses); % per-example normalized cost
Cost_norm_perex = zeros(NClasses, NClasses);  % per-example normalized cost

% vectorize
W_NS_mod = reshape(permute(W_NS, [2, 1, 3]), [DataDim, NFilters * NClasses])'; % W is now (F x C) x D
W_mod = reshape(permute(W, [2, 1, 3]), [DataDim, NFilters * NClasses])';
W0_mod = reshape(permute(W0, [2, 1, 3]), [DataDim, NFilters * NClasses])';
bNoReshape = false; %true; 

X_flat = reshape(X, [size(X, 1), size(X, 2) * size(X, 3)]);
[~, y] = meshgrid(1:size(X, 2), 1:size(X, 3)); %nan(1, NTrain);
                                               %%round((NClasses
                                               %- 1) * rand(1,
                                               %NTrain)) + 1
y = reshape(y', 1, []);
YmatIdx = sub2ind([length(y), max(y)], 1:length(y), y);
Ymat = zeros(length(y), max(y));
Ymat(YmatIdx) = 1;
Ymat = Ymat * diag(1./sum(Ymat)); % normalize
%%
top_percentage = .25;
K2 = size(X_flat, 1)/2;
Xwhitening_data = reshape(X_flat, K2, []);
[WhiteningMat, x_new, res] = PCAWhitening(Xwhitening_data, top_percentage);
W_L = blkdiag(WhiteningMat, WhiteningMat);
X_flat_white = W_L * X_flat;
%%
LabelSelect = -1/19 * ones(1, NClasses);
LabelId = 5;
LabelSelect(LabelId) = 1;
XMult = diag(sum(Ymat * diag(LabelSelect), 2));
XX1 = X_flat * XMult * X_flat';
[UU, SS, VV] = svd(XX1);
[U, S, V] = svd(X_flat * XMult);
%%
tmp = (W_NS(1,:, LabelId) * X_flat).^2 * Ymat; figure; plot(tmp);
tmp = abs(W_NS(1,:, LabelId) * X_flat) * Ymat; figure; plot(tmp);
tmpW = (W_NS(1,:,LabelId) * X_flat_white).^2 * Ymat; figure; plot(tmpW);

tmp = (W(1,:,LabelId) * X_flat).^2 * Ymat; figure; plot(tmp);
tmp = abs(W(1,:,LabelId) * X_flat) * Ymat; figure; plot(tmp);
%%
tmp1 = (UU(:,end)' * X_flat).^2 * Ymat;
figure; plot(tmp1);
%
tmp1 = abs(UU(:,end)' * X_flat) * Ymat;
figure; plot(tmp1);
%
tmp1 = abs(U(:,end)' * X_flat) * Ymat;
figure; plot(tmp1);
%
tmp1 = abs(VV(:,end)' * X_flat) * Ymat;
figure; plot(tmp1);
%
%tmp1 = abs(V(:,end)' * X_flat) * Ymat;
%figure; plot(tmp1);
%%
A = zeros(NClasses, NFilters * NClasses);
for i = 1:size(A, 1)
    A(i,(i - 1) * NFilters + 1:(i * NFilters)) = 1; 
end

% XX = nan(size(X_flat, 1), size(X_flat, 1), size(X_flat, 2));
% for i = 1:size(X_flat, 2)
%    XX(:,:,i) = X(:,i) * X(:,i)';
% end
%%
fnCost = @(X, W, A) ( A * (W * X).^2 ); %@(X, W, A) ( A * abs(W * X) ); %
fnMinCost = @MinL1CostVec; %@L1CostVec; % @MinQuadraticCostVec;
fnMargin = @(Energy, corr_ind, m) ( bsxfun(@minus, Energy(corr_ind) + m, Energy) );
fnNormalize = @(x) (bsxfun(@rdivide, bsxfun(@minus, x, min(x)), max(x) - min(x)));

CostNSAll = fnCost(X_flat, W_NS_mod, A);
CostAllInit = fnCost(X_flat, W0_mod, A);
CostAll = fnCost(X_flat, W_mod, A);
CostNSAll_M = fnMinCost(X_flat, W_NS_mod, A);
CostAllInit_M = fnMinCost(X_flat, W0_mod, A);
CostAll_M = fnMinCost(X_flat, W_mod, A);

CostNSAll_norm = fnNormalize(CostNSAll);
CostAll_norm = fnNormalize(CostAll);
CostNSAll_M_norm = fnNormalize(CostNSAll_M);
CostAll_M_norm = fnNormalize(CostAll_M);
figure; imagesc(CostNSAll_norm); title('normalized CostNS');
figure; imagesc(CostNSAll_M_norm); title('normalized CostNS_M');
figure; imagesc(CostAll_norm); title('normalized Cost');
figure; imagesc(CostAll_M_norm); title('normalized Cost_M');

corr_ind = sub2ind(size(CostAll), y,1:size(CostAll, 2));

MarginLossNS = fnMargin(CostNSAll, corr_ind, 1);
MarginLossNS(corr_ind) = 0;
MarginLoss = fnMargin(CostAll, corr_ind, 1);
MarginLoss(corr_ind) = 0;
%%
CostNSAll_sum = CostNSAll * Ymat;
CostAllInit_sum = CostAllInit * Ymat;
CostAll_sum = CostAll * Ymat;
CostAllInit_M_sum = CostAllInit_M * Ymat;
CostAll_M_sum = CostAll_M * Ymat;
figure; imagesc(CostNSAll_sum);
figure; imagesc(CostAllInit_sum);
figure; imagesc(CostAll_sum);
figure; imagesc(CostAllInit_M_sum);
figure; imagesc(CostAll_M_sum);
%% cost eval
fnCostEval = @MinL1CostVec; %@AffineCostVec; %@L1CostVec; %
Cost = fnCostEval(X_flat, W_mod, A);
CostSum = Cost * Ymat;
figure; imagesc(CostSum)
figure; surf(CostSum); shading interp
[loss, dLdE, loss_per_label] = HingeLossPerLabel(Cost, params.M0, params.CorrScale, Ymat);
loss
%%
[lossNSAll, dLdE, loss_per_label_NS] = HingeLossPerLabel(CostNSAll, params.M0, params.CorrScale, Ymat);
[lossInitAll, dLdE, loss_per_label_init] = HingeLossPerLabel(CostAllInit, params.M0, params.CorrScale, Ymat);
[lossAll, dLdE, loss_per_label] = HingeLossPerLabel(CostAll, params.M0, params.CorrScale, Ymat);
lossNSAll
lossInitAll
lossAll
% for corrLabel = 1:NClasses
%     for testLabel = 1:NClasses
%         for exId = 1:ExamplesPerClass
%             tmpNS = fnEnergy(X(:, exId, corrLabel), W_NS(:,:,testLabel));
%             CostNS(corrLabel, testLabel) = CostNS(corrLabel, testLabel) + tmpNS; 
%             
%             tmp = fnEnergy(X(:, exId, corrLabel), W(:,:,testLabel));
%             Cost(corrLabel, testLabel) = Cost(corrLabel, testLabel) + fnEnergy(X(:, exId, corrLabel), W(:,:,testLabel));
%             
%             tmpNS = bsxfun(@rdivide, bsxfun(@minus, tmpNS, min(tmpNS, []
%             CostNS_norm_perex(corrLabel, testLabel) = CostNS_norm_perex(corrLabel, testLabel) + (tmpNS - min(tmpNS)) / (max(tmp) - min(tmp));
%             
%             Cost_norm_perex(corrLabel, testLabel) = Cost_norm_perex(corrLabel, testLabel) + (tmp - min(tmp)) / (max(tmp) - min(tmp));
%         end
%     end
% end
%
%figure; imagesc(CostNS_norm_perex); axis image
%figure; imagesc(Cost_norm_perex); axis image

%%
[TotalCostNS, grad_NS] = ObjFnL2Reg(X, W_NS(:), m, Lambdas, fnLoss, fnEnergy, NFilters, DataDim, NClasses);
[TotalCost, grad] = ObjFnL2Reg(X, W(:), m, Lambdas, fnLoss, fnEnergy, NFilters, DataDim, NClasses);
%
display(['Init Cost: ' num2str(TotalCostNS) ' , final cost = ' num2str(TotalCost)]);

%% check correctness of vectorized implementation

[TotalCostNS_Vec, grad_NS_vec] = ObjFnL2RegVec(X_flat, W_NS(:), y, A, XX, m, params.CorrScale, Lambdas, fnLoss, ...
                                      fnEnergy, NFilters, DataDim, NClasses, bNoReshape);
                                  
[TotalCost_Vec, grad_vec] = ObjFnL2RegVec(X_flat, W(:), y, A, XX, m, params.CorrScale, Lambdas, fnLoss, ...
                                      fnEnergy, NFilters, DataDim, NClasses, bNoReshape);
                                  
[TotalCost_Vec0, grad_vec0] = ObjFnL2RegVec(X_flat, W0(:), y, A, XX, m, params.CorrScale, Lambdas, fnLoss, ...
                                      fnEnergy, NFilters, DataDim, NClasses, bNoReshape);

display(['(Vectorized) NS Cost: ' num2str(TotalCostNS_Vec) ', Init Cost: ' num2str(TotalCost_Vec0)  ' , final cost = ' num2str(TotalCost_Vec)]);
%sum(abs(grad_NS(:) - grad_NS_vec(:)))
%sum(abs(grad(:) - grad_vec(:)))
%%
if(isfield(params, 'FiniteNorm'))
    params.FiniteNormConst = params.FiniteNorm;
end

[TotalCostNS_Vec1, grad_NS_vec1] = ObjFnV2(X_flat, W_NS_mod(:), y, A, params, NFilters, DataDim, NClasses);
[TotalCost_Vec1, grad_vec1] = ObjFnV2(X_flat, W_mod(:), y, A, params, NFilters, DataDim, NClasses);
[TotalCost_Vec1_0, grad_vec1_0] = ObjFnV2(X_flat, W0_mod(:), y, A, params, NFilters, DataDim, NClasses);
[TotalCost_Vec1_0_old, grad_vec1_0_old] = ObjFnV2_old(X_flat, W0(:), y, A, XX, params, NFilters, DataDim, NClasses, bNoReshape);
tmp = convertFilter2Dto3DFormat(grad_vec1_0, NFilters, NClasses);
sum(abs(tmp(:) - grad_vec1_0_old(:))) 
display(['V2: (Vectorized) NS Cost: ' num2str(TotalCostNS_Vec1) ', Init Cost: ' num2str(TotalCost_Vec1_0) ' , final cost = ' num2str(TotalCost_Vec1) ', old : ' num2str(TotalCost_Vec1_0_old)]);
%%
fnObj = @ObjFnGeneral % params.fnObj; % ObjFnPerFilter
params1 = params;
%params1.ExpZeroLossMult = 1;
%params1.LAMBDAS = [0, 0, 0, 0, 1, 0, 0, 0, 0];
%params.LAMBDAS = [1, 10, 1e3, 10, 1e2, 0]; %[1, 10, 1e2, 1e2, 1e2, 0]; %ones(1, 6); [1 0 0 1 10000 10]
[loss_W0, grad_W0] = fnObj(X_flat, W0_mod(:), y, A, params1, NFilters, DataDim, NClasses);
[loss_W, grad_W] = fnObj(X_flat, W_mod(:), y, A, params1, NFilters, DataDim, NClasses);
loss_W0
loss_W
%display(['V2: (Vectorized) NS Cost: ' num2str(TotalCostNS_Vec1) ', Init Cost: ' num2str(TotalCost_Vec1_0) ' , final cost = ' num2str(TotalCost_Vec1)]);
%%

%%
params2 = params;
params2.CorrScale = 1;
params2.ExpDecayAlpha = 1;
params.ExpPenaltyCenter = 1;
params.CorrExpPenaltyCenter = -1;
params2.fnCost = @AffineCostVec; %@MinL1CostVec;
params2.LAMBDAS = ones(1, 9);
[loss_W, grad_W] = ObjFnGeneral(X_flat, W_mod(:), y, A, params2, NFilters, DataDim, NClasses);
loss_W
%% display all filters
K = sqrt(size(W, 2)/2)
Input = W; %W_NS;
NFilters = size(Input,1 );
for i = 1:size(Input, 3)
    h = figure;
    for k = 1:NFilters
        subplot(NFilters, 2, 2 * (k-1) + 1)
        imagesc(reshape(W(k,1:K*K,i), [K, K])); axis image
        subplot(NFilters, 2, 2 * (k-1) + 2) 
        imagesc(reshape(W(k,((K*K) + 1):end,i), [K, K])); axis image
    end
end
%%
KernelSet = res.KernelSetDisk;
ImgFilename = 'cameraman.tif' %'brown_noise_pattern_512x512.png'; % 
im0 = im2double(imread(ImgFilename));
im1 = blurImg(im0, KernelSet{8, 1}, 'PSF', 'symmetric');
im2 = blurImg(im0, KernelSet{8, 2}, 'PSF', 'symmetric');
figure
imshow([im1 im2])

params.fnNormalize = @(x) normalizeMeanAndStd(x, true);
params.A = A;
params.W = W_NS_mod;
params.KernelSize = 13;
params.fnEnergy = @(a, b, c) QuadraticCostVec(a, b, c);
depthRes = EstDepthDiscrimFilter(im1, im2, params);

figure; hist(depthRes.LabelIdx(:))