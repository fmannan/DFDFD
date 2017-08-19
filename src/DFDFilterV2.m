% find filters that improve depth discrimination

% Build training set
% 5 training and 5 testing images
% error rate on training and testing images
% use 10 depths

%%%
% Jun 14, 2016: added option for rerunning with noise added
%%%
close all
clear
clc

addpath ../
addpath ../DFD/
addpath ../cvpr16/
addpath ../DFD/code1_LSDFD
addpath ../../data/textures/
addpath ../Optimization/
addpath ../Levin_filts/

extraSuffix = ''; %'_v2' %''; % '' %''; %
bUseWNSceneSetup = false; %true;
bUseInvPSFFilters = false; %true;
InitFromFile = 0; % 0: false (no init), 1: from tmp file 2: other
WeightInitFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K3_depth0.529_0.869_51_NF3_PSFSet_WatanabeNayar_PillboxPSF_pair_s0_0.529_s1_0.869_N51_ZSpace_fmingdadam_r2_r3.mat'
%''; %'breakfast_W5_init.mat'; %'breakfast_run4_Wtop3.mat'; %'tmp_iter50000__PSFSet_dancing_pair_s0_0.9_s1_1.1_N51_RelSigmaSpace.mat'; %'tmp_iter50000__PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51.mat'; % when no init file set to empty []
if(bUseWNSceneSetup)
    bUseInvPSFFilters = false;
end
bPatchMeanNorm = false; %true; % change fnPatchNormalization
outfile_suffix = '';
bLoadRealData = false; %true;
real_imseq_suffix = '2A_zFocus609.6_N1_22_N2_11'; %'2F_zNear700_zFar1219.2_f22'; %'2A_zFocus1000_N1_22_N2_11'; %
ImageSeqFilename = ...
   { ['test_plane_OneOverF_ImageDataset_' real_imseq_suffix '.mat'], ...
     ['OneOverF_ImageDataset_' real_imseq_suffix '.mat'], ...
     ['Flower_ImageDataset_' real_imseq_suffix '.mat']};
if(bLoadRealData)
    outfile_suffix = ['_RealImDataV2_' real_imseq_suffix]; % '2F_zNear700_zFar1219.2_f22';
end
RealImageToUseInds = 1:2;
if(length(RealImageToUseInds) == 1)
    outfile_suffix = [outfile_suffix '_ind' num2str(RealImageToUseInds)];
end
display(outfile_suffix)
Z0 = 0.6096;
Z1 = 1.5;
NDepth = 27;
bPSFSetFormat = false; %true; %
PSFSetIndicesToUse = 1:2; % kernels to use
fnTfrom = @(x) (x); %imresize(x, .5); % transform training image sequence
fnBlur = []; %@defocusUnif; % using empty will let it use the default
BlurType = []; %%'gaussian_isotropic_diffusion';  % default:[]
if(bUseWNSceneSetup)
    fnTfrom = @(x) x; %@(x) imresize(x, .5); % transform training image sequence
    fnBlur = @defocusUnif; % using empty will let it use the default
    BlurType = 'gaussian_isotropic_diffusion';  % default:[]
end

bLoadPSF = true;
PSFFilename = 'PSFSet_WatanabeNayar_PillboxPSF_pair_s0_0.529_s1_0.869_N51_ZSpace'; %'PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51'; %'PSFSet_dancing_pair_s0_0.9_s1_1.1_N51_ZSpace'; %'PSFSet_WatanabeNayar_pair_s0_0.529_s1_0.869_N51_ZSpace'; %'PSFSet_WatanabeNayar_pair_s0_0.529_s1_0.869_N51_RelSigmaSpace'; %'PSFSet_Zhou_single1_s0_-5_s1_5_N11'; %'PSFSet_ZhouSingle_s0_-5_s1_5_N11'; %'PSFSet_Zhou_single1_s0_-5_s1_5_N11'; %'PSFSet_Zhou_single1_s0_-5_s1_5_N21'; %'PSFSet_Zhou_pair_s0_-5_s1_5_sR1_N21'; %'PSFSet_Zhou_single1_s0_-3_s1_3_N11'; %'PSFSet_QPPSF_2F_zNear700_zFar1219.2_f22'; %'PSFSet_breakfast_single1_s0_0.529_s1_0.869_N51'; % 'PSFSet_Levin_Synth_N20'; % 'PSFSet_Disk_single1_s0_1_s1_10_N19'; %'PSFSet_Zhou_single1_s0_-3_s1_3'; %'PSFSet_Disk_single1_s0_0_s1_6'; %
bSignClassifier = false; %true; 
strSign = '';
if(bSignClassifier)
    strSign = '_sgn';
end
KernelSetLabel = [];
bLoadAndRun = true; %
DatasetName = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genQuad_unnorm_tp1_sn1_g1_K5_depth0.529_0.869_51_NF1_PSFSet_WatanabeNayar_PillboxPSF_pair_s0_0.529_s1_0.869_N51_ZSpace_fmingdadam_r2'
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K5_depth0.529_0.869_51_NF3_PSFSet_WatanabeNayar_PillboxPSF_pair_s0_0.529_s1_0.869_N51_ZSpace_fmingdadam'
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K3_depth0.529_0.869_51_NF3_PSFSet_WatanabeNayar_PillboxPSF_pair_s0_0.529_s1_0.869_N51_ZSpace_fmingdadam_r2';
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K21_depth0.6096_1.2192_23_NF10_RealImData_f22_2A_zFocus609.6_N1_22_N2_11_ind1_fmingdadam_r2'
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_fileinit2_sn1_g1_K3_depth0.529_0.869_51_NF5_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam_v2'
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_fileinit2_sn1_g1_K3_depth0.529_0.869_51_NF3_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam_v2'
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_fileinit_sn1_g1_K3_depth0.529_0.869_51_NF10_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam_v2_run2_run3'
%'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth1_20_20_PSFSet_Levin_Synth_N20_NF10_fmingdadam_run2_run3_run4_run5';
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K3_depth0.529_0.869_51_NF3_PSFSet_WatanabeNayar_pair_s0_0.529_s1_0.869_N51_ZSpace_fmingdadam_run2x4'; 
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_fileinit_sn1_g1_K3_depth0.529_0.869_51_NF10_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam_v2';
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K3_depth0.529_0.869_51_NF3_PSFSet_WatanabeNayar_pair_s0_0.529_s1_0.869_N51_ZSpace_fmingdadam';
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K7_depth0.9_1.1_51_NF3_PSFSet_dancing_pair_s0_0.9_s1_1.1_N51_ZSpace_fmingdadam';
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K21_depth-5_5_11_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N11_fmingdadam_run2_run3_noise0p05run4';
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K21_depth-5_5_11_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N11_fmingdadam_run2_run3';
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_nsinvpsf_sn1_g1_K21_depth-5_5_5_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N5_fmingdadam_run2'
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K21_depth-5_5_5_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N5_fmingdadam_run2_run3'
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K21_depth-5_5_11_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N11_fmingdadam'
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_nsinvpsf_sn1_g1_K21_depth-5_5_5_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N5_fmingdadam'
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K21_depth-5_5_5_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N5_fmingdadam_run2'
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K21_depth-5_5_11_NF20_PSFSet_ZhouSingle_s0_-5_s1_5_N11_fmingdadam_run2'
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_randinit_sn1_g1_K21_depth-5_5_11_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N11_fmingdadam'; 
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K21_depth-5_5_21_NF10_PSFSet_Zhou_single1_s0_-5_s1_5_N21_fmingdadam'
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_randinit_sn1_g1_K21_depth-5_5_11_NF10_PSFSet_Zhou_single1_s0_-5_s1_5_N11_fmingdadam_run2_run3'
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1.000000e-03_genLogQuad_unnorm_tp1_sn1_g1_K21_depth-5_5_11_NF10_PSFSet_Zhou_single1_s0_-5_s1_5_N11_fmingdadam'
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_mn_tp1_sn1_g1_K21_depth0.6096_1.5_27_NF10_RealImDataV2_2F_zNear700_zFar1219.2_f22_fmingdadam'
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_mn_tp1_sn1_g1_K31_depth0.6096_1.5_27_NF10_RealImDataV2_2A_zFocus609.6_N1_22_N2_11_ind1_fmingdadam'
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_mn_tp1_sn1_g1_K13_depth0.6096_1.5_27_NF10_RealImDataV2_2F_zNear700_zFar1219.2_f22_fmingdadam_run2_x10run3';
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_mn_tp1_sn1_g1_K21_depth0.6096_1.5_27_NF10_RealImDataV2_2A_zFocus1000_N1_22_N2_11_ind1_fmingdadam'
                  %'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K21_depth0.6096_1.5_27_NF10_RealImData_2F_zNear700_zFar1219.2_f22_fmingdadam'; %'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K21_depth0.6096_1.5_27_NF10_RealImData_f22_2A_zFocus609.6_N1_22_N2_11_ind1_fmingdadam'; %'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K21_depth0.6096_1.5_27_NF10_RealImData_2A_zFocus609.6_N1_22_N2_11_fmingdadam';
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth0.6096_1.5_27_NF10_RealImData_2F_zNear700_zFar1219.2_f22_fmingdadam'; %'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K13_depth0.6096_1.5_27_NF10_RealImData_2A_zFocus1000_N1_22_N2_11_fmingdadam'; %'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K13_depth0.6096_1.5_27_NF10_RealImData_2A_zFocus1000_N1_22_N2_11_ind1_fmingdadam'; %'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K13_depth0.6096_1.5_27_NF10_RealImData_f22_2A_zFocus609.6_N1_22_N2_11_ind1_fmingdadam_run2';
%'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K13_depth0.6096_1.5_27_NF10_RealImData_2A_zFocus609.6_N1_22_N2_11_fmingdadam'
%'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K3_depth0.529_0.869_51_NF10_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam'; %'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth609.6_1500_27_NF5_PSFSet_QPPSF_2F_zNear700_zFar1219.2_f22_fmingdadam';
bUsePrev = false; % use previous image set
if(bLoadAndRun)
Dataset = load([DatasetName '.mat'])
params = Dataset.params;
params.NoiseStd = 0; %0.01;
params.W0 = Dataset.resultD.Res.W;
params.MaxIter = 120000; %60000; %100; %200000; %2000;
if(isfield(Dataset, 'W_NS'))
    W_NS = Dataset.W_NS;
end
if(isfield(Dataset, 'RealImageToUseInds'))
    RealImageToUseInds = Dataset.RealImageToUseInds;
end
if(isfield(Dataset, 'ImageDepthSet'))
   ImageDepthSet = Dataset.ImageDepthSet;
else % old version
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
    ImageDepthSet = BuildTrainingSet(Im_seq, KernelSet, params);
end
if(bUsePrev)
   ImageDepthSet = Dataset.Dataset.ImageDepthSet; 
end
if(params.NoiseStd > 0)
   for it = 1:size(ImageDepthSet, 2)
       ImageDepthSet{it} = ImageDepthSet{it} + params.NoiseStd * randn(size(ImageDepthSet{it}));
   end
end
params.fnObj = @ObjFnV3; %@ObjFnV4; %
if(length(params) == 9)
    params.LAMBDAS = [params.LAMBDAS, 0];
end
%params.fnLabelLoss = @(x) (3 * abs(x)).^.1; %@(x) WeightFn(x, .25, .09, -.1, 10); %@(x) (abs(x).^0); %@(x) (.2 * x).^4; %@(x) x.^2; % @(x) x.^4; %
%params.LAMBDAS(10) = 1e-4;
params.options.step_size = 1e-4; %1e-7; 
params.options.print_interval = 200;

params.bBatchGD = false; %true;
params.NBatchIterations = 20000;
params.BatchSize = 2000;

resultD = TrainDiscrimFilters(ImageDepthSet, params)
outFilename = [DatasetName '_r3.mat'] %'_run3absx.mat'
                                        %%%'_run5x0.mat' % '_run6_3absx0p1.mat'
                             %'_run5WFn.mat' % '_run6WFn.mat'
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
bOnlySingleImage = 0; %1;
strSingleImage = '';
if(bOnlySingleImage)
    strSingleImage = '_im1';
end
%% Test using Disk PSFs
Im_seq = cell(1, length(ImFilename));
for idx = 1:length(ImFilename)
    Im_seq{idx} = fnTfrom(mean(im2double(imread(ImFilename{idx})), 3));
end

if(~bLoadRealData)
if(bLoadPSF)
    PSFFile = load([PSFFilename '.mat']);
    outfile_suffix = ['_' PSFFilename];
    if(bPSFSetFormat)
        Z0 = PSFFile.Z(1);
        Z1 = PSFFile.Z(end);
        NDepth = length(PSFFile.Z);
        radii = 1:NDepth;
        PSFSGN = PSFFile.PSFSGN;
        KernelSet = PSFFile.PSFSet(:,PSFSetIndicesToUse);
    else
        Z0 = PSFFile.s0;
        Z1 = PSFFile.s1;
        NDepth = PSFFile.NDepth;
        KernelSet = PSFFile.KernelSet;
        if(isfield(PSFFile, 'scale'))
            radii = PSFFile.scale;
        else
            radii = Z0:Z1;
        end
    end
    
    if(bSignClassifier)
        if(exist('PSFSGN', 'var'))
            KernelSetLabel = PSFSGN;
        else
            KernelSetLabel = sign(radii); 
        end
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
params.NPairs = size(KernelSet, 2)
end
%

if(~isempty(KernelSetLabel))
    params.KernelSetLabel = KernelSetLabel;
end

params.fnBlur = fnBlur;
params.BlurType = BlurType;

params.TruncatePercentage = 1; %.25;
params.NMaxTraining = 1500;
params.KR = 5; %7; %5; %3; %21; %9; %31; %21; %13; %3; %
params.KC = 5; %7; %5; %3; %21; %9; %13; %31; %21; %13; %3; %
params.NFilters = 1; %3; %10; %4; %1; %10; %3; %2; %10; %20; %10; %5; %10; 

params.fnPatchNormalization = @(x) (x); %[x;ones(1, size(x, 2))]  %@(x) (x); % @(x) ( bsxfun(@rdivide, x, mean(x)) - 1); ...
%@(x) (x) % no normalization
MeanNormStr = 'unnorm';
if(bPatchMeanNorm)
    MeanNormStr = 'mn'
    params.fnPatchNormalization = @(x) ( bsxfun(@rdivide, x, mean(x)) );
end
params.fnCombine = []; %@(x,y) bsxfun(@rdivide, bsxfun(@minus, [x;y], ...
                       %                        mean([x;y])), std([x;y])); %@(x,y) [x;y]; %[x/mean(x) - 1; y/mean(y) - 1]; %@(x, y) [x + y] %[y-x] %[x - y]

%%% Build dataset
if(bLoadRealData)
    params.RealImageToUseInds = RealImageToUseInds;
    ImageDepthSet = BuildTrainingSetFromRealImg(ImageSeqFilename, params);
else
    ImageDepthSet = BuildTrainingSet(Im_seq, KernelSet, params);
end
%%%%
result = TrainNullSpaceFilters(ImageDepthSet, params)
W0 = result.FilterVecs;
W_NS = W0;
%%%%
if(bUseInvPSFFilters)
paramsNSInvPSF = params;
paramsNSInvPSF.PSFEnergyThreshold = .9;
paramsNSInvPSF.PatchSize = params.KR;
resNSInvPSF = TrainNullSpaceInvPSFFilters(Im_seq, KernelSet, paramsNSInvPSF);

W_NSInvPSF = resNSInvPSF.FilterVecs;
W_DeconvRecon = resNSInvPSF.FilterVecsInvPSF;
else
 W_NSInvPSF = [];
 W_DeconvRecon = [];
end
%%%%
%Filters = result.Filters;
D = size(W0, 2)  %numel(result.Filters{1, 1});

params.FiniteNormConst = 1;
randW = randn(size(W0)); %W0 + 0.0001 * randn(size(W0));
                         %randW = bsxfun(@rdivide, randW, sqrt(sum(randW.^2, 2)));
params.W0 = W0; 
%% %%
InitTypesStr = {'tp1', 'randinit', 'nsinvpsf', 'deconvrecon'};
WInit = {W_NS, randW, W_NSInvPSF, W_DeconvRecon};
InitTypeIdx = 1; %2; %3; %4; %
InitStr = InitTypesStr{InitTypeIdx}; %'tp1';
params.W0 = params.FiniteNormConst * WInit{InitTypeIdx};
if(InitFromFile > 0)
    InitStr = [InitStr '_fileinit' num2str(InitFromFile)];
    if(InitFromFile == 1) % init from tmp file
        WInitFile = load(WeightInitFilename);
        OptVarsRes = WInitFile.xbest;
        [NFilters, DataDim, NClasses] = size(W0);
        Wsize = NFilters * NClasses * DataDim;
        Wvec = reshape(OptVarsRes(1:Wsize), NFilters * NClasses, DataDim);
        params.W0 = convertFilter2Dto3DFormat(Wvec, NFilters, NClasses);
    elseif(InitFromFile == 2) % init from W
        WInitFile = load(WeightInitFilename);
        if(isfield(WInitFile, 'resultD'))
            W0Init = WInitFile.resultD.W;
        elseif(isfield(WInitFile, 'W'))
            W0Init = WInitFile.W;
        end
        if(size(W0Init, 1) < size(params.W0, 1))
            params.W0(1:size(W0Init,1),:,:) = W0Init;
        else
            params.W0 = W0Init;
        end
    end
end
% if(bUseRand)
%     params.W0 = randW;
%     InitStr = 'randinit';
% end
%%
LossFnStr = 'genQuad'; %'genLogQuad';
params.CorrScale = 1;
params.M0 = 1; %1e-4; %20 * params.FiniteNormConst + params.CorrScale; %2 + params.CorrScale;
params.ZeroLossMult = 1; %2;
params.ExpZeroLossMult = .1; %2;
params.ExpDecayAlpha = .5;
params.ExpPenaltyCenter = 1;
params.CorrExpPenaltyCenter = -1;
params.options.step_size = 1e-2; %6; %1e-5; %1e-4; 
params.options.beta1 = 0.9;
params.options.beta2 = 0.99;
params.fnOpt = @fmingd_adam;
params.options.print_interval = 200;
params.options.storage_interval = 25000; %10000; %
params.options.outfile_suffix = outfile_suffix;
params.fnLabelLoss = @(x) x.^2; %10;
if(strcmpi(LossFnStr,'genQuad'))
    params.fnCost = @QuadraticCostVec; %@(a, b, c) LogEnergy(a, b, c, @QuadraticCostVec); %@AffineCostVec; %@MinL1CostVec; %@L1CostVec; %
elseif(strcmpi(LossFnStr,'genLogQuad'))
    params.fnCost = @(a, b, c) LogEnergy(a, b, c, @QuadraticCostVec);
else
    assert(false)
end
params.LAMBDAS = [0, 0, 0, 1e6, 0, 0, 0, 0, 0, 1]; %ones(1, 9) * 100; %[1, 1, 1e3, 10, 1e2, 0]; %[1, 10, 1e3, 10, 1e2, 0];
%params.LAMBDAS = [1, 0, 0, 1, 1e4, 10]; %[1, 0, 0, 0.01, 1e3, 10];
%%[1, 1e4, 0, 1]; %[10, 1e4, 0, 2];
strLambdas = sprintf('_%d', params.LAMBDAS) 
params.Display = 'iter'; %'off'; %
params.bSteepestDescent = true; %false; 
params.DerivCheck = 'off'; %'on'; %
params.MaxIter = 60000; %120000; %1000; %200000; %1000; %3000;
params.bUseVectorized = true; %
params.bBatchGD = false; %
params.NBatchIterations = 10000;
params.BatchSize = 1000;
params.bOptA = false; %true; %
if(params.bOptA)
    params.fnCost = @(a, b, c, d) LogEnergyV2(a, b, c, d, @QuadraticCostVec); 
    params.fnObj = @ObjFnV4; 
    params.A0 = 1;
    params.A1 = nan;
    outfile_suffix = [outfile_suffix '_OptA']
else
    params.fnObj = @ObjFnV3; %@ObjFnGeneral %@ObjFnPerFilter; %@ObjFnV2; %@ObjFnL2RegVec; %
end
params.bUsefminunc = false;

outFilename = ['FullRes_n' num2str(params.FiniteNormConst) '_m' num2str(params.M0) '_cs' num2str(params.CorrScale) ...
      '_' strLambdas '_' LossFnStr '_' MeanNormStr strSingleImage '_' InitStr '_sn1_g1_K' num2str(params.KR) '_depth' num2str(Z0) '_' num2str(Z1) '_' num2str(NDepth) '_NF' num2str(params.NFilters) outfile_suffix ...
      '_fmingdadam' strSign extraSuffix '.mat']

params
seed = 5219;
rng(seed)
resultD = TrainDiscrimFilters(ImageDepthSet, params)

save(outFilename)

return;
