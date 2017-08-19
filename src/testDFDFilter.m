% find filters that improve depth discrimination

% Build training set
% 5 training and 5 testing images
% error rate on training and testing images
% use 10 depths
close all
clear
clc

addpath ../
addpath ../DFD/
addpath ../../data/textures/
addpath ../Optimization/
addpath ../Levin_filts/

%run('initPathAndDataset.m')
bLoadPSF = true;
PSFFilename = 'PSFSet_Zhou_single1_s0_-3_s1_3_N11'; %'PSFSet_Levin_Synth_N20'; % 'PSFSet_Disk_single1_s0_1_s1_10_N19'; %'PSFSet_Zhou_single1_s0_-3_s1_3'; %'PSFSet_Disk_single1_s0_0_s1_6'; %
bSignClassifier = false; %true;
strSign = '';
if(bSignClassifier)
    strSign = '_sgn';
end
KernelSetLabel = [];
bLoadAndRun = true; 
DatasetName = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth1_20_20_PSFSet_Levin_Synth_N20_NF10_fmingdadam_run2'; %'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth1_20_20_PSFSet_Levin_Synth_N20_fmingdadam_run2';%'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth1_10_19_PSFSet_Disk_single1_s0_1_s1_10_N19_fmingdadam_run2'; %'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth1_10_19_PSFSet_Disk_single1_s0_1_s1_10_N19_fmingdadam_x4LabeLoss'; %
                %'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K9_depth-3_3_21_PSFSet_Zhou_single2_s0_-3_s1_3_fmingdadam'; %'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_21_fmingdadam_run2' %'FullRes_n0_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_randinit_sn1_g1_K13_depth0_6_21'; %'FullRes_n0_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_randinit_sn1_g1_K13_depth0_6_21'; %['FullRes_n1_m1_cs10__1.000000e-' ...
                  %'01_1_0_1000000_0_0_0_0_0_genQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_7_run2_run3_run4']; %'FullRes_n1_m1_cs1__1.000000e-01_1_0_1000000_0_1_0_1_1_genAffineBias_unnorm_im1_randinit_sn1_g1_K13_depth0_6_21_run2'; %'FullRes_n1_m1_cs1__1.000000e-01_1_0_1000000_0_0_0_0_0_genQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_7_run2_run3'; %'FullRes_n1_m1_cs10__1.000000e-01_1_0_1000000_0_0_0_0_0_genQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_7_run2'; %'FullRes_n1_m1_cs1__1.000000e-01_1_0_1000000_0_0_0_0_0_genQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_7_run2'; %'FullRes_n1_m1_cs10__1.000000e-01_1_0_1000000_0_0_0_0_0_genQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_7';
%DatasetName = 'FullRes_n1_m1_cs1__1.000000e-01_1_0_1000000_0_1_0_1_1_genAffineBias_unnorm_im1_randinit_sn1_g1_K13_depth0_6_21_run2'; %'FullRes_n1_m1_cs1__1.000000e-01_1_0_1000000_0_0_0_0_0_genQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_7';
%DatasetName ='FullRes_n1_m1_cs1__1.000000e-01_1_0_1000000_0_0_0_0_0_genAffineBias_unnorm_im1_randinit_sn1_g1_K13_depth0_6_7'; %'FullRes_n1_m1_cs1__1.000000e-01_1_0_1000000_0_1_0_1_1_genAffineBias_unnorm_im1_randinit_sn1_g1_K13_depth0_6_21'; % 'FullRes_n10_m1_cs1__1.000000e-01_1_0_1000000_0_1_0_1_1_genAffineBias_unnorm_im1_tp1_sn1_g1_K13_depth5_15_21'; %'FullRes_n10_m1_cs1__1.000000e-01_1_0_10000_0_1.000000e-02_0_0_0_genQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_21_run2'; %'FullRes_n10_m1_cs1__1.000000e-01_1_0_1000000_0_1_0_1_1_genAffine_unnorm_im1_tp1_sn1_g1_K13_depth5_15_21_run2_run3'; %'FullRes_n10_m1_cs1__1.000000e-01_1_0_10000_0_1.000000e-02_0_0_0_genQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_21'; %'FullRes_n1_m21_cs1__1.000000e-01_0_0_10000_0_10_0_0_0_genMinL1_mnorm_tp1rand_sn1_g1_K13_depth5_15_21_run2';
%'FullRes_m12_cs10__100_10_1_10000_10000_1_0_0_0_genMinL1_im1_randinit_sn1_g1_depth21';
%    'FullRes_m20_cs1__1_10_1_10000_10000_1_0_0_0_genMinL1_im1_randinit_sn1_g1';
if(bLoadAndRun)

Dataset = load([DatasetName '.mat'])
params = Dataset.params;
params.W0 = Dataset.resultD.Res.W;
params.MaxIter = 60000; %30000;
if(~isfield(Dataset, 'Im_seq'))
    Im_seq = Dataset.Dataset.Im_seq;
else
    Im_seq = Dataset.Im_seq;
end

if(isfield(Dataset, 'KernelSet'))
  KernelSet = Dataset.KernelSet;
else
  if(~isfield(Dataset, 'KernelSetDisk'))
    KernelSet = Dataset.Dataset.KernelSetDisk;
  else
    KernelSet = Dataset.KernelSetDisk;
  end
end
params.fnObj = @ObjFnV3;
if(length(params) == 9)
    params.LAMBDAS = [params.LAMBDAS, 0];
end
%params.fnLabelLoss = @(x) x.^4;
%params.LAMBDAS(10) = 1e-3;
params.options.step_size = 1e-6;
params.options.print_interval = 200;
resultD = TrainDiscrimFilters(Im_seq, KernelSet, ...
                              params)
outFilename = [DatasetName '_run3.mat']
save(outFilename);
return
end
bRunTest2 = 0;
ImFilename = {'brown_noise_pattern_512x512.png', 'D42_512x512.png',  'D94_512x512.png', 'merry_mtl07_023.tif'};
bVisualize = 1;
bEstRelBlur = 0;
bEstRelBlurGaussPSF = 0;
bFilterBasic = 1;
bFilterV1 = 0;
bFilterQuadFull = 1;
bFilterQCQP = 0;
bOnlySingleImage = 1;
strSingleImage = '';
if(bOnlySingleImage)
    strSingleImage = '_im1';
end
%% Test using Disk PSFs
Im_seq = cell(1, length(ImFilename));
for idx = 1:length(ImFilename)
    Im_seq{idx} = mean(im2double(imread(ImFilename{idx})), 3);
end
outfile_suffix = '';
if(bLoadPSF)
    PSFFile = load([PSFFilename '.mat']);
    outfile_suffix = ['_' PSFFilename];
    Z0 = PSFFile.s0;
    Z1 = PSFFile.s1;
    NDepth = PSFFile.NDepth;
    KernelSet = PSFFile.KernelSet;
    if(isfield(PSFFile, 'scale'))
        radii = PSFFile.scale;
    else
        radii = Z0:Z1;
    end
    
    if(bSignClassifier)
       KernelSetLabel = sign(radii); 
       KernelSetLabel = KernelSetLabel - min(KernelSetLabel) + 1;
    end
else
Z0 = 0; %5; 
Z1 = 6; %15; 
NDepth = 21; %6; %10; %20;
KernelCellSize = 2;
if(bOnlySingleImage)
    KernelCellSize = 1;
end
KernelSet = cell(NDepth, KernelCellSize);
radii = linspace(Z0, Z1, NDepth); %linspace(0, 5, NDepth); %
for idx = 1:NDepth
   KernelSet{idx, 1} = PillboxKernel(radii(idx), 4, 0);

   if(~bOnlySingleImage)
       KernelSet{idx, 2} = PillboxKernel(sqrt(9 + radii(idx)^2), ...
                                             4, 0); 
   end
end

end
%
params.NPairs = size(KernelSet, 2)
% if(bOnlySingleImage)
%     params.NPairs = 1;
% end
if(~isempty(KernelSetLabel))
    params.KernelSetLabel = KernelSetLabel;
end
params.TruncatePercentage = 1; %.25;
params.NMaxTraining = 1000;
params.KR = 15; %
params.KC = 15;
params.NFilters = 20; %5;
params.fnPatchNormalization = @(x) (x); %[x;ones(1, size(x, 2))]  %@(x) (x); % @(x) ( bsxfun(@rdivide, x, mean(x)) - 1); ...
%@(x) (x) % no normalization
params.fnCombine = []; %@(x,y) bsxfun(@rdivide, bsxfun(@minus, [x;y], ...
                       %                        mean([x;y])), std([x;y])); %@(x,y) [x;y]; %[x/mean(x) - 1; y/mean(y) - 1]; %@(x, y) [x + y] %[y-x] %[x - y]
result = TrainNullSpaceFilters(Im_seq, KernelSet, params)
W0 = result.FilterVecs;
%Filters = result.Filters;
D = size(W0, 2)  %numel(result.Filters{1, 1});

params.FiniteNormConst = 1;
randW = randn(size(W0)); %W0 + 0.0001 * randn(size(W0));
                         %randW = bsxfun(@rdivide, randW, sqrt(sum(randW.^2, 2)));
params.W0 = randW; %params.FiniteNormConst * W0; % + 0.01 * randn(size(W0)); %randn(size(W0)); %
params.CorrScale = 1;
params.M0 = 1; %1e-4; %20 * params.FiniteNormConst + params.CorrScale; %2 + params.CorrScale;
params.ZeroLossMult = 1; %2;
params.ExpZeroLossMult = .1; %2;
params.ExpDecayAlpha = .5;
params.ExpPenaltyCenter = 1;
params.CorrExpPenaltyCenter = -1;
params.options.step_size = 1e-2; %1e-5;
params.options.beta1 = 0.9;
params.options.beta2 = 0.99;
params.fnOpt = @fmingd_adam;
params.options.print_interval = 200;
params.fnLabelLoss = @(x) x.^2; %10;
params.fnCost =@(a, b, c) LogEnergy(a, b, c, @QuadraticCostVec); %@AffineCostVec; %@MinL1CostVec; %@L1CostVec; %
params.LAMBDAS = [0, 1, 0, 1e6, 0, 0, 0, 0, 0, 1]; %ones(1, 9) * 100; %[1, 1, 1e3, 10, 1e2, 0]; %[1, 10, 1e3, 10, 1e2, 0];
%params.LAMBDAS = [1, 0, 0, 1, 1e4, 10]; %[1, 0, 0, 0.01, 1e3, 10];
%%[1, 1e4, 0, 1]; %[10, 1e4, 0, 2];
strLambdas = sprintf('_%d', params.LAMBDAS) 
params.Display = 'iter'; %'off'; %
params.bSteepestDescent = true; %false; 
params.DerivCheck = 'off'; %'on'; %
params.MaxIter = 120000; %3000;
params.bUseVectorized = true; %
params.bBatchGD = false; %true;
params.NBatchIterations = 200;
params.BatchSize = 100;
params.fnObj = @ObjFnV3; %@ObjFnGeneral %@ObjFnPerFilter; %@ObjFnV2; %@ObjFnL2RegVec; %
params.bUsefminunc = false;
params
seed = 5219;
rng(seed)
resultD = TrainDiscrimFilters(Im_seq, KernelSet, params)
%save('FullRes_Large_m10_10_1eM5_BGD.mat')
%save('FullRes_Large_m100_2_10_0_1e3.mat')
%save('FullRes_m12_cs10_1_0_0_1_1e4_10_objv2_ns0p25_scalednorm10.mat')
%save('FullRes_m12_cs10_1_10_1e3_10_1e2_0_objfnperfilter_ns0p25_scalednorm100.mat')
%save(['FullRes_m12_cs10_1_1_1e3_10_1e2_0_objfnperfilter_ns0p25_scalednorm1.mat']) 
%save(['FullRes_m200_cs1_' strLambdas
%'_affine_randinit_sn1_g1.mat'])
%save(['FullRes_m200_cs1_' strLambdas '_genMinL1_tp0p25_sn1_g1.mat'])
outFilename = ['FullRes_n' num2str(params.FiniteNormConst) '_m' num2str(params.M0) '_cs' num2str(params.CorrScale) ...
      '_' strLambdas '_genLogQuad_unnorm' strSingleImage '_randinit_sn1_g1_K' num2str(params.KR) '_depth' num2str(Z0) '_' num2str(Z1) '_' num2str(NDepth) '_NF' num2str(params.NFilters) outfile_suffix ...
      '_fmingdadam' strSign '.mat']

save(outFilename)
%save('test.mat');
return;
result = TrainDiscrimFiltersQuadLoss(Im_seq, KernelSetDisk, params)
Z = 1./(1:NDepth);
PSFSet = cell(NDepth, 4);
PSFSGN = nan(1, NDepth);
FNumb = nan;
zNear = nan;
zFar = nan;
PSFSet(:,1:2) = KernelSetDisk;
PSFSet(:,3:4) = result.Filters;
save(['PSFSet_Disk_DFilter' num2str(params.KR) 'x' num2str(params.KC) '_Scale' num2str(NDepth) '.mat'], 'PSFSet', 'PSFSGN', 'Z', 'FNumb', 'zNear', 'zFar', 'result', 'params', 'radii', 'result');
return
%% basic test of rel blur and bet for zNear = 0.6096 and zFar = 1.5 with real data
KRes = load([ResultRoot '/AbsBlurDisk_All_Kernel_QuadProg_l1000_0_100_10_estkszth_pxsh1_estmodelrad.mat'])  
%KRes = load([ResultRoot '/AbsBlurDisk_All_Kernel_QuadProg_l1000_0_100_10_estkszth_estmodelrad.mat'])  
%KRes = load([ResultRoot '/AbsBlurDisk_All_Kernel_GaussPSF_pxsh1_estmodelrad.mat']) 
outFileSuffix = 'QPEst' %'GaussEst'; % 
im0 = im2double(imread(ImFilename{1}));

zNear = 0.6096; %0.7; %
zFar = 1.5; %1.2192; %
FNumb = 22; %16; %
FNumbSet = [22, 20, 16, 13, 11];
FNumbIdx = find(FNumbSet == FNumb)
Dists = KRes.DistSet;
FocusSet = KRes.FocusSet;
NDepth = length(Dists);

zNearIdx = find(FocusSet == zNear)
zFarIdx = find(FocusSet == zFar)

KernelSet = KRes.BlurKernel(:,[zNearIdx,zFarIdx],1); %_MeanNorm_pf1

%% visualize PSFs
outFileSuffix = [outFileSuffix '_FilterV0'];
if(bVisualize)
    PSFSet = cell(NDepth, 4);
    PSFSGN = zeros(1, NDepth);
    PSFSetMF = cell(NDepth, 4);
    PSFSet(:,3:4) = Filters;
    PSFSetMF(:,3:4) = Filters;
    
    for idx = 1:NDepth
        PSFSet{idx, 1} = KernelSet{idx,1};
        PSFSet{idx, 2} = KernelSet{idx,2};
        PSFSetMF{idx, 1} = medfilt2(KernelSet{idx,1});
        PSFSetMF{idx, 2} = medfilt2(KernelSet{idx,2});
        PSFSetMF{idx, 1} = PSFSetMF{idx, 1} / sum(sum(PSFSetMF{idx, 1}));
        PSFSetMF{idx, 2} = PSFSetMF{idx, 2} / sum(sum(PSFSetMF{idx, 2}));
        figure
        title(['Depth ' num2str(Dists(idx))])
        subplot(2, 2, 1); imagesc(KernelSet{idx,1}); axis image
        subplot(2, 2, 2); imagesc(KernelSet{idx,2}); axis image
        subplot(2, 2, 3); imagesc(PSFSetMF{idx, 1}); axis image
        subplot(2, 2, 4); imagesc(PSFSetMF{idx, 2}); axis image
    end
    Z = Dists;
    save(['PSFSet_' outFileSuffix '_zNear' num2str(zNear) '_zFar' num2str(zFar) '_f' num2str(FNumb) '.mat'], 'PSFSet', 'PSFSGN', 'Z', 'FNumb', 'zNear', 'zFar');
    PSFSet = PSFSetMF;
    save(['PSFSet_' outFileSuffix '_MF_zNear' num2str(zNear) '_zFar' num2str(zFar) '_f' num2str(FNumb) '.mat'], 'PSFSet', 'PSFSGN', 'Z', 'FNumb', 'zNear', 'zFar');
end
%%
if(bEstRelBlurGaussPSF)
    PSFSet = cell(NDepth, 4);
    LAMBDAS = zeros(1, 4);
    for idx = 1:NDepth
        PSFSet{idx, 1} = KernelSet{idx,1};
        PSFSet{idx, 2} = KernelSet{idx,2};
        s1 = KRes.BlurMatrix(idx, zNearIdx, FNumbIdx);
        s2 = KRes.BlurMatrix(idx, zFarIdx, FNumbIdx);
        sR2 = s2^2 - s1^2;
        sR = sqrt(abs(sR2));
        PSFSGN(idx) = sign(sR2);
        if(PSFSGN(idx) == -1) % |sig1| > |sig2|
            PSFSet{idx, 4} = GaussianKernel(sR);
        else
            PSFSet{idx, 3} = GaussianKernel(sR);
        end
        
%         
%         figure
%         title(['Depth ' num2str(Dists(idx))])
%         subplot(3, 2, 1); imagesc(K1); axis image
%         subplot(3, 2, 2); imagesc(K2); axis image
%         subplot(3, 2, 3); imagesc(HRel1); axis image
%         subplot(3, 2, 4); imagesc(HRel2); axis image
%         subplot(3, 2, 5); imagesc(Hest1); axis image
%         subplot(3, 2, 6); imagesc(Hest2); axis image
    end
    Z = Dists;
    save(['PSFSet_' outFileSuffix '_zNear' num2str(zNear) '_zFar' num2str(zFar) '_f' num2str(FNumb) '_GaussRelBlur.mat'], 'PSFSet', 'PSFSGN', 'Z', 'FNumb', 'zNear', 'zFar');
end
%%
if(bEstRelBlur)
    PSFSet = cell(NDepth, 4);
    LAMBDAS = zeros(1, 4);
    for idx = 1:NDepth
        PSFSet{idx, 1} = KernelSet{idx,1};
        PSFSet{idx, 2} = KernelSet{idx,2};
        [K1, K2] = PadEqualSize(PSFSet{idx, 1}, PSFSet{idx, 2});
        RelBlurSZ = 2 * size(K1) + 1;
        [HRel1,FVAL,EXITFLAG,OUTPUT] = EstPSFQuadProg(1e4 * K1, 1e4 * K2, RelBlurSZ, LAMBDAS);
        PSFSet{idx, 3} = HRel1;
        Hest2 = conv2(K1, HRel1, 'same');
        Err12 = sum(reshape(abs(K2 - Hest2), 1, []));
        [HRel2,FVAL,EXITFLAG,OUTPUT] = EstPSFQuadProg(1e4 * K2, 1e4 * K1, RelBlurSZ, LAMBDAS);
        PSFSet{idx, 4} = HRel2;
        Hest1 = conv2(K2, HRel2, 'same');
        Err21 = sum(reshape(abs(K1 - Hest1), 1, []));
        figure
        title(['Depth ' num2str(Dists(idx))])
        subplot(3, 2, 1); imagesc(K1); axis image
        subplot(3, 2, 2); imagesc(K2); axis image
        subplot(3, 2, 3); imagesc(HRel1); axis image
        subplot(3, 2, 4); imagesc(HRel2); axis image
        subplot(3, 2, 5); imagesc(Hest1); axis image
        subplot(3, 2, 6); imagesc(Hest2); axis image
    end
    Z = Dists;
    save(['PSFSet_' outFileSuffix '_zNear' num2str(zNear) '_zFar' num2str(zFar) '_f' num2str(FNumb) '.mat'], 'PSFSet', 'PSFSGN', 'Z', 'FNumb', 'zNear', 'zFar');
end
%% 
Z = 1.5; %0.6096;
ZIdx = find(Dists == Z)

% create defocus pair
im1 = blurImg(im0, KernelSet{ZIdx, 1}, 'PSF', 'symmetric');
im2 = blurImg(im0, KernelSet{ZIdx, 2}, 'PSF', 'symmetric');
figure
imshow([im1, zeros(size(im1, 1), 3), im2])

% find filter pair optimized for Z
KR = 13;
KC = 13;
HS = 50;
im1c = im1(256-HS:256+HS,256-HS:256+HS);
im2c = im2(256-HS:256+HS,256-HS:256+HS);
figure
imshow([im1c, zeros(size(im1c, 1), 3), im2c])

A1 = im2convmtx(im1c, KR, KC);
A2 = im2convmtx(im2c, KR, KC);
A = [A1 -A2];
AA = A'*A;
[U, S, V] = svd(AA);
f1 = reshape(V(1:169,end), 13, 13);
f2 = reshape(V(170:end,end), 13, 13);
figure; imagesc([f1, f2])
im1cb = conv2(im1c, f1, 'same');
im2cb = conv2(im2c, f2, 'same');
figure; imagesc([im1cb, im2cb])
sum((im1cb(:) - im2cb(:)).^2)
%% Find Filter set (Basic)
if(bFilterBasic)
    %%
Filters = cell(NDepth, 4);

KR = 13;
KC = 13;
HS = 50;
BlockSize = KR * 5 * ones(1, 2); %[101, 101];
BlockStep = 50;
X0 = [];
for idx = 1:NDepth
    % create defocus pair
    im1 = blurImg(im0, KernelSet{idx, 1}, 'PSF', 'symmetric');
    im2 = blurImg(im0, KernelSet{idx, 2}, 'PSF', 'symmetric');
    
    [K1, K2] = PadEqualSize(KernelSet{idx, 1}, KernelSet{idx, 2});
    % find filter pair optimized for Z
    im1_seq = PartitionImage(im1, BlockSize, BlockStep);
    im2_seq = PartitionImage(im2, BlockSize, BlockStep);
    AA = [];
    for seqIdx = 1:10 %size(im1_seq, 3)
        im1c = im1_seq(:,:,seqIdx);
        im2c = im2_seq(:,:,seqIdx);
        im1c = im1c/mean(im1c(:)) - 1;
        im1c = im1c / std(im1c(:));
        im2c = im2c / mean(im2c(:)) - 1;
        im2c = im2c / std(im2c(:));
        
        A1 = im2convmtx(im1c, KR, KC);
        A2 = im2convmtx(im2c, KR, KC);
        A = [A1, -A2];
        AAp = A'*A;
        if(seqIdx == 1)
            AA = AAp;
        else
            AA = AA + AAp;
        end
    end
    [U, S, V] = svd(AA);
    f1 = reshape(V(1:169,end), 13, 13);
    f2 = reshape(V(170:end,end), 13, 13);
    X0 = [X0;V];
    im1b = conv2(im1, f1, 'same');
    im2b = conv2(im2, f2, 'same');
    %figure; imagesc([im1cb, im2cb])
    cost = sum((im1b(:) - im2b(:)).^2)
    
    h = figure; subplot(2, 1, 1);imagesc([f1, f2]); title(['Z = ' num2str(Dists(idx))])
    title(['Cost = ' num2str(cost)]);
    
    % from PSFs
    A1 = im2convmtx(K1, KR, KC);
    A2 = im2convmtx(K2, KR, KC);
    A = [A1 -A2];
    AA = A'*A;
    [U, S, V] = svd(AA);
    Kf1 = reshape(V(1:169,end), 13, 13);
    Kf2 = reshape(V(170:end,end), 13, 13);
    
    K1b = conv2(K1, Kf1, 'same');
    K2b = conv2(K2, Kf2, 'same');
    %figure; imagesc([im1cb, im2cb])
    cost = sum((K1b(:) - K2b(:)).^2)
    figure(h); subplot(2, 1, 2); imagesc([Kf1, Kf2]); title(['Z = ' num2str(Dists(idx))])
    title(['Cost = ' num2str(cost)]);
    
    Filters{idx, 1} = f1;
    Filters{idx, 2} = f2;
    Filters{idx, 3} = Kf1;
    Filters{idx, 4} = Kf2;
end
end
%% Find Filter set V1
if(bFilterV1)
close all
FiltersV1 = cell(NDepth, 2);
ImageDepthSet = cell(1, NDepth);
KR = 13;
KC = 13;
HS = 50;
BlockSize = KR * 5 * ones(1, 2); %[101, 101];
BlockStep = 50;
for idx = 1:NDepth
    % create defocus pair
    im1 = blurImg(im0, KernelSet{idx, 1}, 'PSF', 'symmetric');
    im2 = blurImg(im0, KernelSet{idx, 2}, 'PSF', 'symmetric');
    
    %[K1, K2] = PadEqualSize(KernelSet{idx, 1}, KernelSet{idx, 2});
    % find filter pair optimized for Z
%     im1c = im1(256-HS:256+HS,256-HS:256+HS);
%     im2c = im2(256-HS:256+HS,256-HS:256+HS);
    
    im1_seq = PartitionImage(im1, BlockSize, BlockStep);
    im2_seq = PartitionImage(im2, BlockSize, BlockStep);
    AA = [];
    for seqIdx = 1:10 %size(im1_seq, 3)
        im1c = im1_seq(:,:,seqIdx);
        im2c = im2_seq(:,:,seqIdx);

        im1c = im1c/mean(im1c(:)) - 1;
        im1c = im1c / std(im1c(:));
        im2c = im2c / mean(im2c(:)) - 1;
        im2c = im2c / std(im2c(:));
        
        A1 = im2convmtx(im1c, KR, KC);
        A2 = im2convmtx(im2c, KR, KC);
        A = [A1 -A2];
        AAp = A'*A;
        if(seqIdx == 1)
            AA = AAp;
        else
            AA = AA + AAp;
        end
    end
    
    ImageDepthSet{idx} = AA;
end
%
lambda = 1e5;
for idx = 1:NDepth
    AA = NDepth * lambda * ImageDepthSet{idx};
    for idx2 = 1:NDepth
       if(idx ~= idx2)
          AA = AA - ImageDepthSet{idx2}; 
       end
    end
    [U, S, V] = svd(AA);
    f1 = reshape(V(1:169,end), 13, 13);
    f2 = reshape(V(170:end,end), 13, 13);
    
    % create defocus pair
    im1 = blurImg(im0, KernelSet{idx, 1}, 'PSF', 'symmetric');
    im2 = blurImg(im0, KernelSet{idx, 2}, 'PSF', 'symmetric');
    
    % find filter pair optimized for Z
    im1c = im1(256-HS:256+HS,256-HS:256+HS);
    im2c = im2(256-HS:256+HS,256-HS:256+HS);
    
    im1c = im1c/mean(im1c(:)) - 1;
    im1c = im1c / std(im1c(:));
    im2c = im2c / mean(im2c(:)) - 1;
    im2c = im2c / std(im2c(:));

    im1b = conv2(im1c, f1, 'same');
    im2b = conv2(im2c, f2, 'same');
    %figure; imagesc([im1cb, im2cb])
    cost = sum((im1b(:) - im2b(:)).^2)
    
    h = figure; imagesc([f1, f2]); title(['Z = ' num2str(Dists(idx))]) %subplot(2, 1, 1);
    title(['Cost = ' num2str(cost)]);
    
    FiltersV1{idx, 1} = f1;
    FiltersV1{idx, 2} = f2;
end
%%%% Test cost function
% for a given depth apply all filters and evaluate the curvature of the
% cost functions
h = figure;
% TODO: REMOVE BOUNDARY BEFORE COMPUTING COST
cost = nan(2, NDepth, NDepth);
for zIdx = 1:NDepth
    % create defocus pair
    im1 = blurImg(im0, KernelSet{zIdx, 1}, 'PSF', 'symmetric');
    im2 = blurImg(im0, KernelSet{zIdx, 2}, 'PSF', 'symmetric');
    
    for fIdx = 1:NDepth
        im1b = conv2(im1, FiltersV1{fIdx, 1}, 'same');
        im2b = conv2(im2, FiltersV1{fIdx, 2}, 'same');
        cost(1, fIdx, zIdx) = sum(abs(im1b(:) - im2b(:)).^2);
        
        %BET cost
        im1b = conv2(im1, KernelSet{fIdx, 2}, 'same');
        im2b = conv2(im2, KernelSet{fIdx, 1}, 'same');
        cost(2, fIdx, zIdx) = sum(abs(im1b(:) - im2b(:)).^2);
    end
    clr = rand(1, 3);
    figure(h)
    subplot(2, 1, 1); hold on; plot(Dists, cost(1,:,zIdx), 'Color', clr);
    subplot(2, 1, 2); hold on; plot(Dists, cost(2,:,zIdx), 'Color', clr);
    
    figure
    plot(Dists, cost(1,:,zIdx), 'Color', clr);
    title(['Z = ' num2str(Dists(zIdx))])
end
end
%%
if(bFilterQuadFull)
    close all
    ImageDepthSet = cell(1, NDepth);
    ImageDepthSetSingleFilter = cell(1, NDepth);
    KR = 13;
    KC = 13;
    HS = 50;
    BlockSize = KR * 5 * ones(1, 2); %[101, 101];
    BlockStep = 50;
    for idx = 1:NDepth
        % create defocus pair
        im1 = blurImg(im0, KernelSet{idx, 1}, 'PSF', 'symmetric');
        im2 = blurImg(im0, KernelSet{idx, 2}, 'PSF', 'symmetric');

        %[K1, K2] = PadEqualSize(KernelSet{idx, 1}, KernelSet{idx, 2});
        % find filter pair optimized for Z
    %     im1c = im1(256-HS:256+HS,256-HS:256+HS);
    %     im2c = im2(256-HS:256+HS,256-HS:256+HS);

        im1_seq = PartitionImage(im1, BlockSize, BlockStep);
        im2_seq = PartitionImage(im2, BlockSize, BlockStep);
        AA = [];
        AAv2 = [];
        for seqIdx = 1:10 %size(im1_seq, 3)
            im1c = im1_seq(:,:,seqIdx);
            im2c = im2_seq(:,:,seqIdx);

            im1c = im1c/mean(im1c(:)) - 1;
            im1c = im1c / std(im1c(:));
            im2c = im2c / mean(im2c(:)) - 1;
            im2c = im2c / std(im2c(:));

            A1 = im2convmtx(im1c, KR, KC);
            A2 = im2convmtx(im2c, KR, KC);
            A = [A1, -A2];
            AAp = A'*A; % only for quadratic distance
            Av2 = A1 - A2; % single filter
            AApv2 = Av2' * Av2; % only for quadratic distance (single filter)
            if(seqIdx == 1)
                AA = AAp;
                AAv2 = AApv2;
            else
                AA = AA + AAp;
                AAv2 = AAv2 + AApv2;
            end
        end

        ImageDepthSet{idx} = AA;
        ImageDepthSetSingleFilter{idx} = AAv2;
    end
    %
    %Res = EstDefocusFilterPairs(ImageDepthSet, {})
    ImageDepthSetSmall = ImageDepthSet(18:22);
    params.X0 = X0(:,end); %(17*338+1:22*338,end);
    %
    %Res = EstDefocusFilterPairsCOpt(ImageDepthSetSmall, params)
    params.ERR_CONST = 1; %1e4;
    Res = EstDefocusFilterPairsBasicQCQP(ImageDepthSet, params)
    
%     params = {}
%     params.ERR_CONST = 10;
%     ImageDepthSetSingleFilterSmall = ImageDepthSetSingleFilter(12:2:20);
%     Res = EstDefocusFilterPairsBasicQCQP(ImageDepthSetSingleFilterSmall, params, 1)    
%     Res1 = EstDefocusFilterPairsCOpt(ImageDepthSetSingleFilterSmall, params, 1)
    %%
    Filters = Res.Filters;
    %%
    if(size(Filters, 2) == 1)
        for zIdx = 1:size(Filters, 1)
            figure
            subplot(1,2,1); imagesc(Filters{zIdx}); axis image
            subplot(1,2,2); imagesc(log(1 + fftshift(abs(fft2(Filters{zIdx}))))); axis image
        end
    else
        for zIdx = 1:size(Filters, 1)
            figure
            subplot(2, 2, 1); imagesc(Filters{zIdx, 1}); axis image
            subplot(2, 2, 2); imagesc(Filters{zIdx, 2}); axis image
            subplot(2,2,3); imagesc(log(1 + (abs(fft2(Filters{zIdx, 1}))))); axis image
            subplot(2,2,4); imagesc(log(1 + (abs(fft2(Filters{zIdx, 2}))))); axis image
        end
    end
    %%
    %%%% Test cost function
    % for a given depth apply all filters and evaluate the curvature of the
    % cost functions
    h = figure;
    % TODO: REMOVE BOUNDARY BEFORE COMPUTING COST
    cost = nan(NDepth, NDepth);
    for zIdx = 1:NDepth
        % create defocus pair
        im1 = blurImg(im0, KernelSet{zIdx, 1}, 'PSF', 'symmetric');
        im2 = blurImg(im0, KernelSet{zIdx, 2}, 'PSF', 'symmetric');
        im1 = (im1 - mean(im1(:)))/mean(im1(:));
        im1 = im1 / std(im1(:));
        im2 = (im2 - mean(im2(:)))/mean(im2(:));
        im2 = im2 / std(im2(:));
        
        for fIdx = 1:length(Filters) %NDepth
            if(size(Filters, 2) > 1)
                im1b = conv2(im1, Filters{fIdx, 1}, 'same');
                im2b = conv2(im2, Filters{fIdx, 2}, 'same');
            elseif(size(Filters, 2) == 1)
                im1b = conv2(im1, Filters{fIdx}, 'same');
                im2b = conv2(im2, Filters{fIdx}, 'same');
            end
            
            cost(fIdx, zIdx) = sum(abs(im1b(:) - im2b(:)).^2);
        end
        clr = rand(1, 3);
        figure(h)
        hold on; plot(Dists, cost(:,zIdx), 'Color', clr);
        

        figure
        plot(Dists, cost(:,zIdx), 'Color', clr);
        title(['Z = ' num2str(Dists(zIdx))])
    end
    %% test basic constraint satisfaction for test data
    
end
if(bFilterQCQP)
  disp('here')  
end
%%
if(bRunTest2)
SIG1 = 0:8;
SIGR = 0:8;

im0 = im2double(imread(ImFilename{1}));

for s1 = SIG1
    im1 = defocusBlurImg(im0, s1, 'disk');
    for sR = SIGR
        s2 = sqrt(s1^2 + sR^2);
        im2 = defocusBlurImg(im0, s2, 'disk');
    end
end
end
