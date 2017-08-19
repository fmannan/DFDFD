% Jun 18, 2016 01:54 : TODO: ADD GRAPH CUT SUPPORT
close all
clear
clc

addpath ../GraphCut/
addpath ../DFD/
addpath ../DFD/code1_LSDFD/
addpath ../../data/dfd_data/favaro_soatto/s3/
addpath ../Normalization/

bGraphCut = false; %true; % 
DataSetIdx = 1; % 1: breakfast, 2: empire, 3: dancing
ext = '.ppm';
Dataset = {struct('im1Name', 'breakfast_near', 'im2Name', 'breakfast_far', 'outputName', 'breakfast', 'CROP_ROWS', 131:232, 'CROP_COLS', 326:427), ...
            struct('im1Name', 'empire_near', 'im2Name', 'empire_far', 'outputName', 'empire'), ...
            struct('im1Name', 'nearfocus', 'im2Name', 'farfocus', 'outputName', 'dancing'), ...
            };
    

im1Name = Dataset{DataSetIdx}.im1Name;
im2Name = Dataset{DataSetIdx}.im2Name;
outputName = Dataset{DataSetIdx}.outputName
CROP_ROWS = Dataset{DataSetIdx}.CROP_ROWS;
CROP_COLS = Dataset{DataSetIdx}.CROP_COLS;

ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K3_depth0.529_0.869_51_NF10_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam.mat';
ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K5_depth0.529_0.869_51_NF3_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam_v2.mat';
ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_fileinit_sn1_g1_K3_depth0.529_0.869_51_NF10_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam_v2.mat';
%ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_fileinit_sn1_g1_K5_depth0.9_1.1_51_NF10_PSFSet_dancing_pair_s0_0.9_s1_1.1_N51_RelSigmaSpace_fmingdadam.mat';
%ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K7_depth0.9_1.1_51_NF3_PSFSet_dancing_pair_s0_0.9_s1_1.1_N51_ZSpace_fmingdadam.mat';
%ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K3_depth0.529_0.869_51_NF3_PSFSet_WatanabeNayar_pair_s0_0.529_s1_0.869_N51_ZSpace_fmingdadam.mat';
%ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K5_depth0.529_0.869_51_NF4_PSFSet_WatanabeNayar_pair_s0_0.529_s1_0.869_N51_RelSigmaSpace_fmingdadam.mat';
ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K7_depth0.9_1.1_51_NF3_PSFSet_dancing_pair_s0_0.9_s1_1.1_N51_ZSpace_fmingdadam_run2.mat';
ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_fileinit_sn1_g1_K3_depth0.529_0.869_51_NF10_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam_v2_run2_run3.mat';
ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_fileinit_sn1_g1_K3_depth0.529_0.869_51_NF10_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam_v2_run2_run3_run4.mat';
ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_fileinit2_sn1_g1_K3_depth0.529_0.869_51_NF3_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam_v2.mat'
ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_fileinit2_sn1_g1_K3_depth0.529_0.869_51_NF5_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam_v2_r2.mat'
ResData = load(ResultFilename);
if(~isfield(ResData.params, 'NPairs'))
    ResData.params.NPairs = 1;
end
if(isfield(ResData.resultD, 'Filters'))
    Filters = ResData.resultD.Filters;
else
    Filters = ResData.resultD.W;
end
ResParam = BuildInputParams(Filters, ResData.params)
outDir = ['./' outputName '/'];
if(~exist(outDir, 'dir'))
    mkdir(outDir);
end
params.KernelSize = 3; %7; %5; %
fileSuffix = 'K3_NF5_N51_v2_r2'; %'K3_NF3_N51_v2'; %'K3_NF10_N51_v2_run4'; %'K3_NF10_N51_v2_run3'; %'K7_NF3_N51_run2'; %'K3_NF10_N51_v2'; %'K5_NF4_N51_RelSigmaSpace'; %'K3_NF3_N51_ZSpace'; %'K5_NF10_N51'; % 'K5_NF3_N51'; %'K3_NF10_N51';
%%%%%
W_NS = ResData.W_NS;
Wopt = Filters;
I1 = im2double(rgb2gray(imread([im1Name ext])));
I2 = im2double(rgb2gray(imread([im2Name ext])));
%%
H0 = nan(size(Wopt, 2), size(Wopt, 2), size(Wopt, 3));
HpOpt = nan(size(Wopt, 2), size(Wopt, 2), size(Wopt, 3));
for i = 1:size(Wopt, 3)
    H0(:,:,i) = W_NS(:, :, i)' * W_NS(:,:,i); 
    HpOpt(:,:,i) = Wopt(:, :, i)' * Wopt(:,:,i); 
end

plotfigures = false;
DepthInit = estimate_depth(I1,I2,H0,plotfigures);
%DistInit = Z(DepthInit);

DepthFilter = estimate_depth(I1,I2,HpOpt,plotfigures);
%DistOpt = Z(DepthOpt);

imwrite(uint8(255 * normalizeMaxMin(DepthInit)), [outDir 'DepthInit_from_estimate_depth.png'])
imwrite(uint8(255 * normalizeMaxMin(DepthFilter)), [outDir 'DepthFilter_from_estimate_depth' fileSuffix '.png'])
figure
imagesc(DepthInit)
title('init')

figure
imagesc(DepthFilter)
title('filter')
%%
DepthInit_2 = estimate_depth(I1,I2,W_NS,plotfigures);
DepthFilter_2 = estimate_depth(I1,I2,Wopt,plotfigures);
figure
imagesc(DepthInit_2)
title('init_2')

figure
imagesc(DepthFilter_2)
title('filter_2')
%%

params.HSmooth = []; % 1; %fspecial('average', 3); %fspecial('gaussian', 3, 0.5) %

NFilters = size(W_NS, 1); %3; %1; %2; %1:2; %
fileSuffix = [fileSuffix '_NFused' num2str(NFilters)];
params.W = W_NS(1:NFilters,:,:);%W0; %(1,:,:);
paramsInit = params;
resInitWODel = EstDepthDiscrimFilterConv(I1, I2, paramsInit)
figure; imagesc(resInitWODel.LabelIdx)
title('init w/o del2')

params.W = Filters(1:NFilters,:,:);
resOptWODel = EstDepthDiscrimFilterConv(I1, I2, params)
figure; imagesc(resOptWODel.LabelIdx)
title('opt w/o del')

imwrite(uint8(255 * normalizeMaxMin(resInitWODel.LabelIdx)), [outDir 'DepthInit_from_EstDepthDiscrimFilterConv' fileSuffix '.png'])
imwrite(uint8(255 * normalizeMaxMin(resOptWODel.LabelIdx)), [outDir 'DepthFilter_from_EstDepthDiscrimFilterConv' fileSuffix '.png'])

del2I1 = del2(I1);
del2I2 = del2(I2);
resInit = EstDepthDiscrimFilterConv(del2I1, del2I2, paramsInit)
figure; imagesc(resInit.LabelIdx)
title('init')

%params.W = Filters;
resOpt = EstDepthDiscrimFilterConv(del2I1, del2I2, params)
figure; imagesc(resOpt.LabelIdx)
title('opt')

imwrite(uint8(255 * normalizeMaxMin(resInit.LabelIdx)), [outDir 'LapDepthInit_from_EstDepthDiscrimFilterConv' fileSuffix '.png'])
imwrite(uint8(255 * normalizeMaxMin(resOpt.LabelIdx)), [outDir 'LapDepthFilter_from_EstDepthDiscrimFilterConv' fileSuffix '.png'])
%%
tmp1 = medfilt2(resInit.LabelIdx);
figure; imagesc(tmp1); axis image
title('init median filtered')

tmp2 = medfilt2(resOpt.LabelIdx);
figure; imagesc(tmp2); axis image
title('optimized median filtered')

figure
hold on
plot(squeeze(resInit.Cost(165, 336, :)), 'r')
plot(squeeze(resInit.Cost(165, 337, :)), 'g')
plot(squeeze(resInit.Cost(165, 338, :)), 'b')

plot(squeeze(resOpt.Cost(165, 336, :)), 'r--')
plot(squeeze(resOpt.Cost(165, 337, :)), 'g--')
plot(squeeze(resOpt.Cost(165, 338, :)), 'b--')
%%
tmp1 = medfilt2(resInit.LabelIdx);
figure; imagesc(-tmp1); axis image
colormap gray
title('init median filtered')

tmp2 = medfilt2(resOpt.LabelIdx);
figure; imagesc(-tmp2); axis image
colormap gray
title('optimized median filtered')

imwrite(uint8(255 - 255 * normalizeMaxMin(tmp1)), [outDir 'LapDepthInit_from_EstDepthDiscrimFilterConv_medfilt3' fileSuffix '.png'])
imwrite(uint8(255 - 255 * normalizeMaxMin(tmp2)), [outDir 'LapDepthFilter_from_EstDepthDiscrimFilterConv_medfilt3' fileSuffix '.png'])

tmp1Crop = tmp1(CROP_ROWS, CROP_COLS);
tmp2Crop = tmp2(CROP_ROWS, CROP_COLS);

figure; imagesc(-tmp1Crop); axis image
colormap gray
title('init median filtered')
figure; imagesc(-tmp2Crop); axis image
colormap gray
title('optimized median filtered')

imwrite(uint8(255 - 255 * normalizeMaxMin(tmp1Crop)), [outDir 'LapDepthInit_from_EstDepthDiscrimFilterConv_medfilt3' fileSuffix '_cropped.png'])
imwrite(uint8(255 - 255 * normalizeMaxMin(tmp2Crop)), [outDir 'LapDepthFilter_from_EstDepthDiscrimFilterConv_medfilt3' fileSuffix '_cropped.png'])
%%
tmp1 = medfilt2(resInit.LabelIdx, [5, 5]);
figure; imagesc(-tmp1); axis image
colormap gray
title('init median filtered 5x5')
tmp1Crop = tmp1(CROP_ROWS, CROP_COLS);

tmp2 = medfilt2(resOpt.LabelIdx, [5, 5]);
figure; imagesc(-tmp2); axis image
colormap gray
title('optimized median filtered 5x5')
tmp2Crop = tmp2(CROP_ROWS, CROP_COLS);
imwrite(uint8(255 - 255 * normalizeMaxMin(tmp1)), [outDir 'LapDepthInit_from_EstDepthDiscrimFilterConv_medfilt5' fileSuffix '.png'])
imwrite(uint8(255 - 255 * normalizeMaxMin(tmp2)), [outDir 'LapDepthFilter_from_EstDepthDiscrimFilterConv_medfilt5' fileSuffix '.png'])

%% GraphCut
if(exist('bGraphCut', 'var') && bGraphCut)
    SmoothnessParams = struct('lambda1', 10, 'lambda2', 1, 'lambda_ig', 10, 'expK', 2, 'beta', .01, 'nbhd', 8, 'tau', 100 );
    SmoothnessParams.a = 1;
    SmoothnessParams.b = 1;
    Unary = resOpt.Cost;
    Label1 = 1:size(Unary, 3);
    im11_meannorm_norm = normalizeMaxMin(del2I1); %img1; %(:,:,2);
    LabelIdx = Grid2DAExp(Unary, SmoothnessParams, [1:length(Label1(:))]',  double(im11_meannorm_norm));
    figure
    imagesc(-LabelIdx)
    axis image
    colormap gray
    title('GC label index', 'fontsize', 14)
    %%
    Unary = resInit.Cost;
    Label1 = 1:size(Unary, 3);
    im11_meannorm_norm = normalizeMaxMin(del2I1); %img1; %(:,:,2);
    LabelIdx = Grid2DAExp(Unary, SmoothnessParams, [1:length(Label1(:))]',  double(im11_meannorm_norm));
    figure
    imagesc(-LabelIdx)
    axis image
    colormap gray
    title('GC label index Null space', 'fontsize', 14)
end