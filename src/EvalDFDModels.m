% for chapter 7 optimal model
% Jun 20, 2016 (12:59pm): see ../ThesisPlotScripts/plotEvalDFDModelWithDiscrimFilters.m
addpath ../CodedAperture/codedApertureSubspaceDepth/code/
addpath ../cvpr16/
addpath ../Deconv/
addpath ../OpticalFlow/
addpath ../DFD/

close all
clear
clc

bRunAll = true; %false;
%RealDataVerStr = ''; %'V2'; %v1 : ''
RealDataVerStr = '_RealImData' %'_RealImDataV2' %'_PSFSet_QPPSF' ; %
bMeanNormalize = 1;  %0: no normalization, 1: local mean norm, 2:
                     %global mean norm
ResMeanNormStr = 'unnorm'; %'mn'; %
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
ImageFilePrefix = 'WhiteNoise'; %'TreeBark'; %'OneOverF'; %'test_plane_OneOverF'; % 'SlantedPlane2'; %
suffix_str = '2A_zFocus609.6_N1_22_N2_11'; %'2F_zNear700_zFar1219.2_f22'; % '2A_zFocus1000_N1_22_N2_11'; % '2F_zNear609.6_zFar1500_f22'; % 
ImageDataset = load([ImageFilePrefix '_ImageDataset_' suffix_str '.mat'])
res_suffix = '_run2'; %''; %'_run2_run3'; %'_run2_x10run3'; % ''; %
KernelSize = 21 %31; % 13; %
RealImageToUseInds = 1; %1:2;
NFilters = 10; %5; 
ImageFile = load(['../cvpr16/' ImageFilePrefix '_StairImage_' suffix_str '.mat']); %load(['../DFD/slanted_plane2_' suffix_str '.mat']);% load('../DFD/slanted_plane2_2F_zNear700_zFar1219.2_f22.mat'); %
PSFFile = load(['../cvpr16/PSFSet_QPPSF_' suffix_str '.mat'])
PSFGaussFile = load(['../cvpr16/PSFSet_QPPSFGauss_' suffix_str '.mat'])

if(length(RealImageToUseInds) == 1)
    suffix_str = [suffix_str '_ind' num2str(RealImageToUseInds(1))]
end
bRunSingle = true;
if(isempty(ImageFile))
    bRunSingle = false;
end
Z = PSFFile.Z / 1e3;
PSFSet = PSFFile.PSFSet;
PSFSetGauss = PSFGaussFile.PSFSet;

if(bRunSingle)
if(isfield(ImageFile, 'ImStair1'))
    im1 = ImageFile.ImStair1;
else
    im1 = ImageFile.im1;
end
if(isfield(ImageFile, 'ImStair2'))
    im2 = ImageFile.ImStair2;
else
    im2 = ImageFile.im2;
end
GTLabel = [];
if(isfield(ImageFile, 'GTLabel'))
    GTLabel = ImageFile.GTLabel;
end
figure; imagesc([im1; im2]);
%imshow([im1; im2], [])
figure; imagesc(GTLabel);
axis image
colorbar

end
%%
addpath ../Normalization/
%imwrite(uint8(normalizeMaxMin(im1) * 255), [ImageFilePrefix '_StairImage1_' suffix_str '.png']);
%imwrite(uint8(normalizeMaxMin(im2) * 255), [ImageFilePrefix '_StairImage2_' suffix_str '.png']);
%FilterResFile = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth609.6_1500_27_NF5_PSFSet_QPPSF_2F_zNear700_zFar1219.2_f22_fmingdadam_run2_run3.mat';
%%
%dir_suffix = 'K13_depth0.6096_1.5_27_NF10_RealImData_2F_zNear700_zFar1219.2_f22';
%dir_suffix = 'K13_depth0.6096_1.5_27_NF10_RealImData_f22_2A_zFocus609.6_N1_22_N2_11_ind1';
%dir_suffix = 'K21_depth0.6096_1.5_27_NF10_RealImData_f22_2A_zFocus609.6_N1_22_N2_11_ind1'; %
%dir_suffix =
%'K13_depth0.6096_1.5_27_NF10_RealImData_f11_2A_zFocus609.6_N1_22_N2_11_ind2';
%%

dir_suffix = ['K' num2str(KernelSize) ...
              '_depth0.6096_1.5_27_NF' num2str(NFilters) RealDataVerStr '_' suffix_str] % 2A_zFocus609.6_N1_22_N2_11';

dir_suffix = 'K13_depth0.6096_1.5_27_NF10_RealImData_2A_zFocus1000_N1_22_N2_11_ind1'; % 
%dir_suffix = 'K13_depth0.6096_1.5_27_NF10_RealImData_2A_zFocus1000_N1_22_N2_11'; % 
%FilterResFile = ['FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_' dir_suffix '_fmingdadam_run2.mat'];
FilterResFile = ...
    ['FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_' ...
     ResMeanNormStr '_tp1_sn1_g1_' dir_suffix '_fmingdadam' res_suffix ...
     '.mat'];
%FilterResFile = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth609.6_1500_27_NF5_PSFSet_QPPSF_2F_zNear700_zFar1219.2_f22_fmingdadam_run2_run3.mat';
res = load(FilterResFile)

outDir = ['plots_stair_' ImageFilePrefix '_' ResMeanNormStr '_']
outDir = [outDir dir_suffix  res_suffix extra_suffix]; 
outDir = [outDir '/'];
if(~exist(outDir, 'dir'))
    mkdir(outDir)
end
display(outDir)
%%


%if(isfield(res, 'RealImageToUseInds'))
%    RealImageToUseInds = res.RealImageToUseInds;
%end
params = res.params;
params.KernelSize = res.params.KR;
params.W = res.resultD.W;
params.fnNormalize = params.fnPatchNormalization;
params.HSmooth = fspecial('gaussian', 11, 2); %fspecial('average', 3);

params_NS = params;
if(~isfield(res, 'W_NS'))
    if(isfield(res, 'ImageDepthSet'))
        ImageDepthSet = res.ImageDepthSet;
    else
        display('creating ImageDepthSet');
        ImageDepthSet = BuildTrainingSet(res.Im_seq, res.KernelSet, res.params);
    end
    resultNS = TrainNullSpaceFilters(ImageDepthSet, params_NS);
    params_NS.W = resultNS.FilterVecs;
else
    params_NS.W = res.W_NS;
end

params_init = params;
params_init.W = res.params.W0;
if(bRunSingle)
if(length(RealImageToUseInds) == 1)
   if(RealImageToUseInds(1) == 1)
       im2 = [];
   else
       im1 = im2;
       im2 = [];
   end
end
% Filters = res.resultD.W;
% H = permute(Filters, [2, 1, 3]);
% patchSizeX = res.params.KR; %27;
% patchSizeY = res.params.KC; %27;
% useNormalization = nan;
% insideDimensions = size(H, 2)
% mult = -1;
%%
%%
DepthRes = EstDepthDiscrimFilterConv(im1, im2, params)
figure
imagesc(Z(DepthRes.LabelIdx))
axis image
title('Optimized Filter');
colorbar
%%

DepthResInit = EstDepthDiscrimFilterConv(im1, im2, params_init)
figure
imagesc(Z(DepthResInit.LabelIdx))
axis image
colorbar
title('Init Filters')
%% NS

DepthResNS = EstDepthDiscrimFilterConv(im1, im2, params_NS)
figure
imagesc(Z(DepthResNS.LabelIdx))
axis image
colorbar
title('NS Filters')
%%
h = figure;
hold on;
plot(mean(Z(GTLabel)), 'k')
plot(nanmean(Z(DepthResInit.LabelIdx)), 'c')
plot(nanmean(Z(DepthRes.LabelIdx)), 'm')
legend('GT', 'Init', 'Optimized', 'Location', 'SouthEast')
saveas(h, [outDir '/stair_init_opt.png'])
print(h, '-depsc', [outDir '/stair_init_opt.eps'])
end
if(length(RealImageToUseInds) == 1) % end here because the rest
                                    % requires pair of images
if(bRunAll)
    crop_boundary = 100;
    params.HSmooth = fspecial('average', 31);
    ImageSet = ImageDataset.ImageSet;
    NDepths = size(ImageSet, 1);
    DiscrimRes = cell(NDepths, 1);
    DiscrimInitRes = cell(NDepths, 1);
    DiscrimNSRes = cell(NDepths, 1);
    %Z = Z/1e3; %ImageDataset.ObjDists / 1e3;
    params.Z = Z;
    params_init.Z = Z;
    params_NS.Z = Z;
    for idx = 1:NDepths
        DepthRes = EstDepthDiscrimFilterConv(ImageSet{idx, ...
                            RealImageToUseInds(1)}, [], params);
        DiscrimRes{idx}.DistStat = ComputeErrorStat(Z(DepthRes.LabelIdx), ...
                                                    Z(idx), crop_boundary);
        DiscrimRes{idx}.LabelStat = ComputeErrorStat(DepthRes.LabelIdx, ...
                                                     idx, crop_boundary);
    
        DepthRes = EstDepthDiscrimFilterConv(ImageSet{idx, ...
                            RealImageToUseInds(1)}, [], params_init);
        DiscrimInitRes{idx}.DistStat = ComputeErrorStat(Z(DepthRes.LabelIdx), ...
                                                    Z(idx), crop_boundary);
        DiscrimInitRes{idx}.LabelStat = ComputeErrorStat(DepthRes.LabelIdx, ...
                                                     idx, crop_boundary);
                                                 
        DepthRes = EstDepthDiscrimFilterConv(ImageSet{idx, ...
                            RealImageToUseInds(1)}, [], params_NS);
        DiscrimNSRes{idx}.DistStat = ComputeErrorStat(Z(DepthRes.LabelIdx), ...
                                                    Z(idx), crop_boundary);
        DiscrimNSRes{idx}.LabelStat = ComputeErrorStat(DepthRes.LabelIdx, ...
                                                     idx, crop_boundary);
    end
    [MeanDepth, StdDepth, DepthRMSE, MeanInvDepth, StdInvDepth, ...
     InvDepthRMSE] = DepthErrorStatAll(DiscrimRes);
 
    [InitMeanDepth, InitStdDepth, InitDepthRMSE, InitMeanInvDepth, InitStdInvDepth, ...
     InitInvDepthRMSE] = DepthErrorStatAll(DiscrimInitRes);
 
    [NSMeanDepth, NSStdDepth, NSDepthRMSE, NSMeanInvDepth, NSStdInvDepth, ...
     NSInvDepthRMSE] = DepthErrorStatAll(DiscrimNSRes);
    save([outDir '/ResDiscrim.mat'], 'DiscrimRes', 'Z', ...
      'MeanDepth', 'StdDepth', 'DepthRMSE', 'MeanInvDepth', 'StdInvDepth', 'InvDepthRMSE', ...
      'InitMeanDepth', 'InitStdDepth', 'InitDepthRMSE', 'InitMeanInvDepth', 'InitStdInvDepth', 'InitInvDepthRMSE', ...
      'NSMeanDepth', 'NSStdDepth', 'NSDepthRMSE', 'NSMeanInvDepth', 'NSStdInvDepth', 'NSInvDepthRMSE', ...
      '-v7.3')
    h = figure;
    hold on;
    plot(Z, Z, 'k')
    errorbar(Z, MeanDepth, StdDepth, 'r')
    errorbar(Z, InitMeanDepth, InitStdDepth, 'g')
    errorbar(Z, NSMeanDepth, NSStdDepth, 'c')
    legend('GT', 'Filter', 'Init', 'NS', 'Location', 'SouthEast')
    saveas(h, [outDir '/DepthStat.png'])
    print(h, '-depsc', [outDir '/DepthStat.eps'])

    h = figure;
    hold on;
    plot(1./Z, 1./Z, 'k')
    errorbar(1./Z, MeanInvDepth, StdInvDepth, 'r')
    errorbar(1./Z, InitMeanInvDepth, InitStdInvDepth, 'g')
    errorbar(1./Z, NSMeanInvDepth, NSStdInvDepth, 'c')
    legend('GT', 'Filter', 'Init', 'NS', 'Location', 'SouthEast')
    saveas(h, [outDir '/InvDepthStat.png'])
    print(h, '-depsc', [outDir '/InvDepthStat.eps'])
    
end

    return;
end
%%
if(bRunSingle)
expK = 2;
crop_boundary = 0;
%resEstDepth = EstDepth(im1, im2, PSFFile.PSFSet, PSFFile.PSFSGN, expK, PSFFile.Z, PSFFile.Z(GTLabel), params.HSmooth, [], true);
resEstDepthRelBlur = EstRelBlurDFD(im1, im2, PSFFile.PSFSet, PSFFile.PSFSGN, expK, PSFFile.Z, params.HSmooth, crop_boundary);
figure;
imagesc(resEstDepthRelBlur.Depth)
axis image
colorbar
title('Rel Blur');

%
resEstDepthBET = EstBETDFD(im1, im2, PSFFile.PSFSet, expK, PSFFile.Z, params.HSmooth, crop_boundary);
figure;
imagesc(resEstDepthBET.Depth)
axis image
colorbar
title('BET')

resDeconv = EstDeconvWnrGenDFD(im1, im2, PSFFile.PSFSet, expK, PSFFile.Z, params.HSmooth, crop_boundary);
%%
h = figure
imagesc(Z(resDeconv.LabelIdx))
axis image
colorbar
title('Generalized Wiener Deconvolution')

%%
h = figure;
hold on
plot(mean(Z(GTLabel)), 'k')
plot(nanmean(Z(resEstDepthRelBlur.LabelIdx)), 'r')
plot(nanmean(Z(resEstDepthBET.LabelIdx)), 'g')
plot(nanmean(Z(resDeconv.LabelIdx)), 'b')
plot(nanmean(Z(DepthRes.LabelIdx)), 'm')
legend('GT', 'Rel Blur', 'BET', 'Deconv', 'Filter (Opt)', 'Location', 'SouthEast')
saveas(h, [outDir '/stair_all.png'])
print(h, '-depsc', [outDir '/stair_all.eps'])
%%
h = figure;
hold on
plot(mean(Z(GTLabel)), 'k')
plot(nanmean(Z(resEstDepthRelBlur.LabelIdx)), 'r')
plot(nanmean(Z(resEstDepthBET.LabelIdx)), 'g')
plot(nanmean(Z(resDeconv.LabelIdx)), 'b')
plot(nanmean(Z(DepthRes.LabelIdx)), 'm')
plot(nanmean(Z(DepthResInit.LabelIdx)), 'c')
legend('GT', 'Rel Blur', 'BET', 'Deconv', 'Filter (Opt)', 'Filter (NS)', 'Location', 'SouthEast')
saveas(h, [outDir '/stair_ns_all.png'])
print(h, '-depsc', [outDir '/stair_ns_all.eps'])
%save([outDir '/impair_resall.mat'], 'resEstDepthRelBlur', 'resEstDepthBET', ...
%      'resDeconv', 'DepthRes', 'DepthResInit');
end
%%
if(bRunAll)
    expK = 2;
    crop_boundary = 100;
    params.HSmooth = fspecial('average', 31);
    ImageSet = ImageDataset.ImageSet;
    NDepths = size(ImageSet, 1);
    DiscrimRes = cell(NDepths, 1);
    DiscrimResNS = cell(NDepths, 1);
    RelBlurRes = cell(NDepths, 1);
    GaussRelBlurRes = cell(NDepths, 1);
    BETRes = cell(NDepths, 1);
    DeconvRes = cell(NDepths, 1);
    %Z = Z / 1e3;
    params.crop_boundary_size = 0;
    params.Z = Z;
    params_NS.Z = Z;
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
        DiscrimDepthRes = EstDepthDiscrimFilterConv(im1, im2, params);
        DiscrimDepthResNS = EstDepthDiscrimFilterConv(im1, im2, params_NS);
        RelBlurDepthRes = EstRelBlurDFD(im1, im2, PSFFile.PSFSet, PSFFile.PSFSGN, expK, Z, params.HSmooth, 0);
        GaussRelBlurDepthRes = EstRelBlurDFD(im1, im2, PSFGaussFile.PSFSet, PSFGaussFile.PSFSGN, expK, Z, params.HSmooth, 0);
        BETDepthRes = EstBETDFD(im1, im2, PSFFile.PSFSet, expK, Z, params.HSmooth, 0);
        DeconvDepthRes = EstDeconvWnrGenDFD(im1, im2, PSFFile.PSFSet, expK, Z, params.HSmooth, 0);
        
        DiscrimRes{idx} = ComputeErrorStat(Z(DiscrimDepthRes.LabelIdx), Z(idx), crop_boundary);
        DiscrimResNS{idx} = ComputeErrorStat(Z(DiscrimDepthResNS.LabelIdx), Z(idx), crop_boundary);
        RelBlurRes{idx} = ComputeErrorStat(Z(RelBlurDepthRes.LabelIdx), Z(idx), crop_boundary);
        GaussRelBlurRes{idx} = ComputeErrorStat(Z(GaussRelBlurDepthRes.LabelIdx), Z(idx), crop_boundary);
        BETRes{idx} = ComputeErrorStat(Z(BETDepthRes.LabelIdx), Z(idx), crop_boundary);
        DeconvRes{idx} = ComputeErrorStat(Z(DeconvDepthRes.LabelIdx), Z(idx), crop_boundary);
    end
    [DiscrimMeanDepth, DiscrimStdDepth, DiscrimDepthRMSE, DiscrimMeanInvDepth, DiscrimStdInvDepth, ...
     DiscrimInvDepthRMSE] = DepthErrorStatAll(DiscrimRes);
 
    [DiscrimNSMeanDepth, DiscrimNSStdDepth, DiscrimNSDepthRMSE, DiscrimNSMeanInvDepth, DiscrimNSStdInvDepth, ...
     DiscrimNSInvDepthRMSE] = DepthErrorStatAll(DiscrimResNS);
 
    [RelBlurMeanDepth, RelBlurStdDepth, RelBlurDepthRMSE, RelBlurMeanInvDepth, RelBlurStdInvDepth, ...
     RelBlurInvDepthRMSE] = DepthErrorStatAll(RelBlurRes);
 
    [GaussRelBlurMeanDepth, GaussRelBlurStdDepth, GaussRelBlurDepthRMSE, GaussRelBlurMeanInvDepth, GaussRelBlurStdInvDepth, ...
     GaussRelBlurInvDepthRMSE] = DepthErrorStatAll(GaussRelBlurRes);
 
    [BETMeanDepth, BETStdDepth, BETDepthRMSE, BETMeanInvDepth, BETStdInvDepth, ...
     BETInvDepthRMSE] = DepthErrorStatAll(BETRes);
 
    [DeconvMeanDepth, DeconvStdDepth, DeconvDepthRMSE, DeconvMeanInvDepth, DeconvStdInvDepth, ...
     DeconvInvDepthRMSE] = DepthErrorStatAll(DeconvRes);
    save([outDir '/ResAll.mat'], 'DiscrimRes', 'DiscrimResNS', 'RelBlurRes', 'BETRes', 'DeconvRes', ...
        'DiscrimMeanDepth', 'DiscrimStdDepth', 'DiscrimDepthRMSE', 'DiscrimMeanInvDepth', 'DiscrimStdInvDepth', 'DiscrimInvDepthRMSE', ...
        'DiscrimNSMeanDepth', 'DiscrimNSStdDepth', 'DiscrimNSDepthRMSE', 'DiscrimNSMeanInvDepth', 'DiscrimNSStdInvDepth', 'DiscrimNSInvDepthRMSE', ...
        'RelBlurMeanDepth', 'RelBlurStdDepth', 'RelBlurDepthRMSE', 'RelBlurMeanInvDepth', 'RelBlurStdInvDepth', 'RelBlurInvDepthRMSE', ...
        'GaussRelBlurMeanDepth', 'GaussRelBlurStdDepth', 'GaussRelBlurDepthRMSE', 'GaussRelBlurMeanInvDepth', 'GaussRelBlurStdInvDepth', 'GaussRelBlurInvDepthRMSE', ...
        'BETMeanDepth', 'BETStdDepth', 'BETDepthRMSE', 'BETMeanInvDepth', 'BETStdInvDepth', 'BETInvDepthRMSE', ...
        'DeconvMeanDepth', 'DeconvStdDepth', 'DeconvDepthRMSE', 'DeconvMeanInvDepth', 'DeconvStdInvDepth', 'DeconvInvDepthRMSE', ...
        '-v7.3')
    
    h = figure;
    hold on;
    plot(Z, Z, 'k')
    errorbar(Z, RelBlurMeanDepth, RelBlurStdDepth, 'r')
    errorbar(Z, BETMeanDepth, BETStdDepth, 'g')
    errorbar(Z, DeconvMeanDepth, DeconvStdDepth, 'b')
    errorbar(Z, DiscrimMeanDepth, DiscrimStdDepth, 'm')
    legend('GT', 'RelBlur', 'BET', 'Deconv', 'Filter', 'Location', 'SouthEast');
    axis tight
    saveas(h, [outDir '/DepthStatAll.png'])
    print(h, '-depsc', [outDir '/DepthStatAll.eps'])

    h = figure;
    hold on;
    plot(Z, Z, 'k')
    errorbar(Z, RelBlurMeanDepth, RelBlurStdDepth, 'r')
    errorbar(Z, GaussRelBlurMeanDepth, GaussRelBlurStdDepth, 'c')
    errorbar(Z, BETMeanDepth, BETStdDepth, 'g')
    errorbar(Z, DeconvMeanDepth, DeconvStdDepth, 'b')
    errorbar(Z, DiscrimMeanDepth, DiscrimStdDepth, 'm')
    axis tight
    legend('GT', 'RelBlur', 'Gaussian RelBlur', 'BET', 'Deconv', 'Filter', 'Location', 'SouthEast');
    saveas(h, [outDir '/DepthStatAll.png'])
    print(h, '-depsc', [outDir '/DepthStatAllGauss.eps'])
    %%%%%%%%%%%%%
    h = figure;
    hold on;
    InvZ = 1./Z;
    plot(InvZ, InvZ, 'k')
    errorbar(InvZ, RelBlurMeanInvDepth, RelBlurStdInvDepth, 'r')
    errorbar(InvZ, BETMeanInvDepth, BETStdInvDepth, 'g')
    errorbar(InvZ, DeconvMeanInvDepth, DeconvStdInvDepth, 'b')
    errorbar(InvZ, DiscrimMeanInvDepth, DiscrimStdInvDepth, 'm')
    legend('GT', 'RelBlur', 'BET', 'Deconv', 'Filter', 'Location', 'SouthEast');
    axis tight
    saveas(h, [outDir '/InvDepthStatAll.png'])
    print(h, '-depsc', [outDir '/InvDepthStatAll.eps'])
    
    h = figure;
    hold on;
    InvZ = 1./Z;
    plot(InvZ, InvZ, 'k')
    errorbar(InvZ, RelBlurMeanInvDepth, RelBlurStdInvDepth, 'r')
    errorbar(InvZ, GaussRelBlurMeanInvDepth, GaussRelBlurStdInvDepth, 'c')
    errorbar(InvZ, BETMeanInvDepth, BETStdInvDepth, 'g')
    errorbar(InvZ, DeconvMeanInvDepth, DeconvStdInvDepth, 'b')
    errorbar(InvZ, DiscrimMeanInvDepth, DiscrimStdInvDepth, 'm')
    legend('GT', 'RelBlur', 'Gaussian RelBlur', 'BET', 'Deconv', 'Filter', 'Location', 'SouthEast');
    axis tight
    saveas(h, [outDir '/InvDepthStatAll.png'])
    print(h, '-depsc', [outDir '/InvDepthStatAllGauss.eps'])
    %%%% compare NS and opt
    h = figure;
    hold on;
    plot(Z, Z, 'k')
    errorbar(Z, DiscrimMeanDepth, DiscrimStdDepth, 'm')
    errorbar(Z, DiscrimNSMeanDepth, DiscrimNSStdDepth, 'c')
    legend('GT', 'Filter', 'Filter (NS)', 'Location', 'SouthEast');
    axis tight
    saveas(h, [outDir '/DepthStat_opt_ns.png'])
    print(h, '-depsc', [outDir '/DepthStat_opt_ns.eps'])

    h = figure;
    hold on;
    InvZ = 1./Z;
    plot(InvZ, InvZ, 'k')
    errorbar(InvZ, DiscrimMeanInvDepth, DiscrimStdInvDepth, 'm')
    errorbar(InvZ, DiscrimNSMeanInvDepth, DiscrimNSStdInvDepth, 'c')
    legend('GT', 'Filter', 'Filter (NS)', 'Location', 'SouthEast');
    axis tight
    saveas(h, [outDir '/InvDepthStat_opt_ns.png'])
    print(h, '-depsc', [outDir '/InvDepthStat_opt_ns.eps'])
end
