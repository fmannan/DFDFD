%%% The difference from the first version (i.e. EvalDFDModels.m) is that this one doesn't evaluate
%%% the filters (only cvpr16 and crv16 results)
addpath ../CodedAperture/codedApertureSubspaceDepth/code/
addpath ../cvpr16/
addpath ../Deconv/
addpath ../OpticalFlow/
addpath ../DFD/

close all
clear
clc

GradObj = 'on'; %'off'; %
PreFilterSigma = .4; %.8;

zNear = 1500; %609.6; %500; %762; %1500; %700; %
zFar = 1500; %609.6; %inf; %1000; %1500; %609.6; %1219.2; %609.6; %
N1 = 22; %11; %22; %
N2 = 16; %13; %11; %22; %
px = (4288/23.6 + 2848/15.8)/2;
F = 50;
camera1.F = F;
camera1.px = px;
camera1.Fnumb = N1;
camera2 = camera1;
camera2.Fnumb = N2;

bMeanNormalize = 0;  %0: no normalization, 1: local mean norm, 2:
                     %global mean norm
ResMeanNormStr = 'mn'; %'unnorm'; %
extra_suffix = ''; % 
AvgFilter = fspecial('average', 31);
if(bMeanNormalize)
    if(bMeanNormalize == 1)
        extra_suffix = [extra_suffix '_mn']; %''; % 
    elseif(bMeanNormalize == 2)
        extra_suffix = [extra_suffix '_gmn'];
    else
        assert(false)
    end
end
ImageFilePrefix = 'OneOverF'; %'WhiteNoise' % 'test_plane_OneOverF'; %'TreeBark'; %'SlantedPlane2'; %
                           
suffix_str = '2A_zFocus1500_N1_22_N2_16'; %'2A_zFocus609.6_N1_22_N2_16'; %'2A_zFocus609.6_N1_22_N2_13'; %'2A_zFocus1500_N1_22_N2_11'; %'2A_zFocus609.6_N1_22_N2_11'; % '2F_zNear609.6_zFar1500_f22'; %'2F_zNear700_zFar1219.2_f22'; %  '2A_zFocus1000_N1_22_N2_11'; % 
%suffix_str = ['2F_zNear' num2str(zNear) '_zFar' num2str(zFar) '_f' num2str(N1)]; %

ImageDataset = load([ImageFilePrefix '_ImageDataset_' suffix_str '.mat'])
res_suffix = ''; %'_run2'; %'_run2_run3'; %'_run2_x10run3'; % ''; %

ImageFile = []; % load(['../cvpr16/' ImageFilePrefix '_StairImage_' suffix_str '.mat']); %load(['../DFD/slanted_plane2_' suffix_str '.mat']);% load('../DFD/slanted_plane2_2F_zNear700_zFar1219.2_f22.mat'); %
PSFFile = load(['../cvpr16/PSFSet_QPPSF_' suffix_str '.mat'])
PSFGaussFile = load(['../cvpr16/PSFSet_QPPSFGauss_' suffix_str '.mat'])

Z = PSFFile.Z / 1e3;
PSFSet = PSFFile.PSFSet;
PSFSetGauss = PSFGaussFile.PSFSet;


%%
addpath ../Normalization/

outDir = ['model_eval_' suffix_str '_gradobj' GradObj '_prefiltersig' num2str(PreFilterSigma) extra_suffix];

if(~exist(outDir, 'dir'))
    mkdir(outDir)
end
display(outDir)
%%

expK = 2;
crop_boundary = 100;
params.HSmooth = fspecial('average', 31);
ImageSet = ImageDataset.ImageSet;
NDepths = size(ImageSet, 1);
RelBlurRes = cell(NDepths, 1);
GaussRelBlurRes = cell(NDepths, 1);
BETRes = cell(NDepths, 1);
DeconvRes = cell(NDepths, 1);

params.crop_boundary_size = 0;
params.Z = Z;
params.z0 = min(Z(:)) * 1e3; % this is used by GaussRelBlur 
params.z1 = max(Z(:)) * 1e3;
params.zNear = zNear;
params.zFar = zFar;
params.camera1 = camera1;
params.camera2 = camera2;

GRelBlurParam = params;
GRelBlurParam.PatchSize = [201, 201];
GRelBlurParam.Stride = [100, 100]; %[50, 50]; %
GRelBlurParam.SigmaC = 0;
GRelBlurParam.LB = -30;
GRelBlurParam.UB = 30;
GaussRelBlurVar = cell(NDepths, 1);
GCRelBlurSigREst = nan(1, NDepths);
GRelBlurParam.GradObj = GradObj;
HFilter = GaussianKernel(PreFilterSigma); %fspecial('gaussian', PrefilterSize, PrefilterSize/7);
for idx = 1:NDepths
    im1 = ImageSet{idx, 1};
    im2 = ImageSet{idx, 2};
   
    if(bMeanNormalize == 1)
        im1M = conv2(im1, AvgFilter, 'same');
        im2M = conv2(im2, AvgFilter, 'same');

        im1 = im1 ./ im1M;
        im2 = im2 ./ im2M; 
    elseif(bMeanNormalize == 2)
        im1 = im1 / mean(im1(:));
        im2 = im2 / mean(im2(:));
    end

    im1 = imfilter(im1, HFilter, 'symmetric');
    im2 = imfilter(im2, HFilter, 'symmetric');

    GRelBlurParam.DepthGT = Z(idx); % * 1e3;
    if(zNear < zFar)
        GaussRelBlurVar{idx} = EvalGaussRelBlurDFDVar(im2, im1, crop_boundary, GRelBlurParam);
    else
        GaussRelBlurVar{idx} = EvalGaussRelBlurDFDVar(im1, im2, crop_boundary, GRelBlurParam);
    end
    GCRelBlurSigREst(idx) = GaussRelBlurVar{idx}.SigREstMean;
    
    RelBlurDepthRes = EstRelBlurDFD(im1, im2, PSFFile.PSFSet, PSFFile.PSFSGN, expK, Z, params.HSmooth, 0);
    GaussRelBlurDepthRes = EstRelBlurDFD(im1, im2, PSFGaussFile.PSFSet, PSFGaussFile.PSFSGN, expK, Z, params.HSmooth, 0);
    BETDepthRes = EstBETDFD(im1, im2, PSFFile.PSFSet, expK, Z, params.HSmooth, 0);
    DeconvDepthRes = EstDeconvWnrGenDFD(im1, im2, PSFFile.PSFSet, expK, Z, params.HSmooth, 0);

    RelBlurRes{idx} = ComputeErrorStat(Z(RelBlurDepthRes.LabelIdx), Z(idx), crop_boundary);
    GaussRelBlurRes{idx} = ComputeErrorStat(Z(GaussRelBlurDepthRes.LabelIdx), Z(idx), crop_boundary);
    BETRes{idx} = ComputeErrorStat(Z(BETDepthRes.LabelIdx), Z(idx), crop_boundary);
    DeconvRes{idx} = ComputeErrorStat(Z(DeconvDepthRes.LabelIdx), Z(idx), crop_boundary);
end
% Gaussian continuous relative blur
[GCRelBlurMeanDepth, GCRelBlurStdDepth, GCRelBlurDepthRMSE, GCRelBlurMeanInvDepth, GCRelBlurStdInvDepth, ...
 GCRelBlurInvDepthRMSE] = DepthErrorStatAll(GaussRelBlurVar);

[RelBlurMeanDepth, RelBlurStdDepth, RelBlurDepthRMSE, RelBlurMeanInvDepth, RelBlurStdInvDepth, ...
 RelBlurInvDepthRMSE] = DepthErrorStatAll(RelBlurRes);

[GaussRelBlurMeanDepth, GaussRelBlurStdDepth, GaussRelBlurDepthRMSE, GaussRelBlurMeanInvDepth, GaussRelBlurStdInvDepth, ...
 GaussRelBlurInvDepthRMSE] = DepthErrorStatAll(GaussRelBlurRes);

[BETMeanDepth, BETStdDepth, BETDepthRMSE, BETMeanInvDepth, BETStdInvDepth, ...
 BETInvDepthRMSE] = DepthErrorStatAll(BETRes);

[DeconvMeanDepth, DeconvStdDepth, DeconvDepthRMSE, DeconvMeanInvDepth, DeconvStdInvDepth, ...
 DeconvInvDepthRMSE] = DepthErrorStatAll(DeconvRes);
save([outDir '/ResAll.mat'], 'Z', 'GaussRelBlurVar', 'GCRelBlurSigREst', 'RelBlurRes', 'BETRes', 'DeconvRes', ...
    'RelBlurMeanDepth', 'RelBlurStdDepth', 'RelBlurDepthRMSE', 'RelBlurMeanInvDepth', 'RelBlurStdInvDepth', 'RelBlurInvDepthRMSE', ...
    'GaussRelBlurMeanDepth', 'GaussRelBlurStdDepth', 'GaussRelBlurDepthRMSE', 'GaussRelBlurMeanInvDepth', 'GaussRelBlurStdInvDepth', 'GaussRelBlurInvDepthRMSE', ...
    'BETMeanDepth', 'BETStdDepth', 'BETDepthRMSE', 'BETMeanInvDepth', 'BETStdInvDepth', 'BETInvDepthRMSE', ...
    'DeconvMeanDepth', 'DeconvStdDepth', 'DeconvDepthRMSE', 'DeconvMeanInvDepth', 'DeconvStdInvDepth', 'DeconvInvDepthRMSE', ...
    'GCRelBlurMeanDepth', 'GCRelBlurStdDepth', 'GCRelBlurDepthRMSE', 'GCRelBlurMeanInvDepth', 'GCRelBlurStdInvDepth', 'GCRelBlurInvDepthRMSE', ...
    'PreFilterSigma', 'HFilter', '-v7.3')

h = figure;
hold on;
plot(Z, Z, 'k')
errorbar(Z, RelBlurMeanDepth, RelBlurStdDepth, 'r')
errorbar(Z, GaussRelBlurMeanDepth, GaussRelBlurStdDepth, 'c')
errorbar(Z, BETMeanDepth, BETStdDepth, 'g')
errorbar(Z, DeconvMeanDepth, DeconvStdDepth, 'b')
legend('GT', 'RelBlur', 'Gaussian RelBlur', 'BET', 'Deconv', 'Location', 'Best');
saveas(h, [outDir '/DepthStatAll.png'])
print(h, '-depsc', [outDir '/DepthStatAll.eps'])
%%%%%%%%%%%%%
h = figure;
hold on;
InvZ = 1./Z;
plot(InvZ, InvZ, 'k')
errorbar(InvZ, RelBlurMeanInvDepth, RelBlurStdInvDepth, 'r')
errorbar(InvZ, GaussRelBlurMeanInvDepth, GaussRelBlurStdInvDepth, 'c')
errorbar(InvZ, BETMeanInvDepth, BETStdInvDepth, 'g')
errorbar(InvZ, DeconvMeanInvDepth, DeconvStdInvDepth, 'b')
legend('GT', 'RelBlur', 'Gaussian RelBlur', 'BET', 'Deconv', 'Location', 'Best');
saveas(h, [outDir '/InvDepthStatAll.png'])
print(h, '-depsc', [outDir '/InvDepthStatAll.eps'])

%%%%%%%%%%%%%%%%%
h = figure;
hold on;
plot(Z, Z, 'k')
errorbar(Z, RelBlurMeanDepth, RelBlurStdDepth, 'r')
errorbar(Z, GaussRelBlurMeanDepth, GaussRelBlurStdDepth, 'c')
errorbar(Z, GCRelBlurMeanDepth, GCRelBlurStdDepth, 'm')
legend('GT', 'RelBlur', 'Gaussian RelBlur', 'Gaussian RelBlur Cont', 'Location', 'Best');
saveas(h, [outDir '/DepthStatRelBlurAll.png'])
print(h, '-depsc', [outDir '/DepthStatRelBlurAll.eps'])
%%%%%%%%%%%%%
h = figure;
hold on;
InvZ = 1./Z;
plot(InvZ, InvZ, 'k')
errorbar(InvZ, RelBlurMeanInvDepth, RelBlurStdInvDepth, 'r')
errorbar(InvZ, GaussRelBlurMeanInvDepth, GaussRelBlurStdInvDepth, 'c')
errorbar(InvZ, GCRelBlurMeanInvDepth, GCRelBlurStdInvDepth, 'm')
legend('GT', 'RelBlur', 'Gaussian RelBlur', 'Gaussian RelBlur Cont', 'Location', 'Best');
saveas(h, [outDir '/InvDepthStatRelBlurAll.png'])
print(h, '-depsc', [outDir '/InvDepthStatRelBlurAll.eps'])
%%%%%%%%%%%%%%%%%%
h = figure;
hold on;
plot(Z, RelBlurStdDepth, 'r')
plot(Z, GaussRelBlurStdDepth, 'c')
plot(Z, GCRelBlurStdDepth, 'm')
legend('RelBlur', 'Gaussian RelBlur', 'Gaussian RelBlur Cont', 'Location', 'Best');
saveas(h, [outDir '/DepthStdRelBlurAll.png'])
print(h, '-depsc', [outDir '/DepthStdRelBlurAll.eps'])
%%%%%%%%%%%%%
h = figure;
hold on;
InvZ = 1./Z;
plot(InvZ, RelBlurStdInvDepth, 'r')
plot(InvZ, GaussRelBlurStdInvDepth, 'c')
plot(InvZ, GCRelBlurStdInvDepth, 'm')
legend('RelBlur', 'Gaussian RelBlur', 'Gaussian RelBlur Cont', 'Location', 'Best');
saveas(h, [outDir '/InvDepthStdRelBlurAll.png'])
print(h, '-depsc', [outDir '/InvDepthStdRelBlurAll.eps'])
