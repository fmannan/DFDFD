% Evaluate training and test error for loss optimization based filters
% this is a better version of evalDFDFilter.m and based on
% test*Composition.m
%
% Jun 17, 2016 (18:00): Added support for storing the result so that it can
% be plotted later using a plotting script from ../ThesisPlotScripts/ folder.
close all
clear
clc

addpath ../
addpath ../DFD
addpath ../Normalization/
addpath ../../data/textures

COMMENTS = ['Results produced on ' datestr(now)]
bEvalLargeFilterSet = true; %false;
bSkipNSInvPSF = true; %false; % skip if blur is computed by some other methods than using a PSF (e.g. heat diffusion)
MAXLARGEFILTER = 4; %200; %50; %
%ResultFilename = 'FullRes_n1_m1_cs1__1.000000e-01_1_0_1000000_0_1_0_1_1_genAffineBias_unnorm_im1_randinit_sn1_g1_K13_depth0_6_21_run2_run3.mat'; %'FullRes_n1_m1_cs10__1.000000e-01_1_0_1000000_0_0_0_0_0_genQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_7_run2_run3_run4_run5.mat'; %'FullRes_n1_m1_cs1__1.000000e-01_1_0_1000000_0_1_0_1_1_genAffineBias_unnorm_im1_randinit_sn1_g1_K13_depth0_6_21_run2.mat'; %'FullRes_n1_m0_cs1__1.000000e-01_1_0_10000_0_1.000000e-02_0_0_0_genQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_21.mat'; %'FullRes_n10_m1_cs1__1.000000e-01_1_0_1000000_0_1.000000e-02_0_0_0_genQuad_unnorm_tp1_sn1_g1_K13_depth0_6_21.mat' %'FullRes_n1_m0_cs1__1.000000e-01_1_0_10000_0_1.000000e-02_0_0_0_genQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_21.mat'; %'FullRes_m12_cs10__100_10_1_10000_10000_1_0_0_0_genMinL1_im1_randinit_sn1_g1_depth21_run2.mat'; %'FullRes_n1_m0.0001_cs1__1.000000e-01_1_0_10000_0_1.000000e-02_0_0_0_genL1_unnorm_im1_tp1_sn1_g1_K13_depth0_6_21.mat'; %'FullRes_n1_m0_cs1__1.000000e-01_1_0_10000_0_1.000000e-02_0_0_0_genL1_unnorm_im1_tp1_sn1_g1_K13_depth0_6_21.mat'; %'FullRes_m20_cs1__1_10_1_10000_10000_1_0_0_0_genMinL1_plus_randinit_sn1_g1_run2_run3_run4.mat'; %'FullRes_n1_m1_cs1__1.000000e-01_1_0_1000000_0_1.000000e-02_0_0_0_genQuad_unnorm_tp1_sn1_g1_K13_depth0_6_21_30it.mat'; %'FullRes_n10_m1_cs1__1.000000e-01_1_0_10000_0_1.000000e-02_0_0_0_genQuadCost_unnorm_tp1_sn1_g1_K13_depth5_15_21_20it.mat'; %'FullRes_n1_m0_cs1__1.000000e-01_1_0_10000_0_1_0_0_0_genMinL1_unnorm_tp1_sn1_g1_K13_depth5_15_21_20it.mat'; %'FullRes_n1_m21_cs1__1.000000e-01_0_0_10000_0_10_0_0_0_genMinL1_mnorm_tp1rand_sn1_g1_K13_depth5_15_21.mat';
%ResultFilename = 'FullRes_n1_m2_cs1__1.000000e-01_1_0_1000000_0_0_0_0_0_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_7.mat'; %'FullRes_n10_m1_cs1__1.000000e-01_1_0_1000000_0_0_0_0_0_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_7.mat'; %'FullRes_n1_m1_cs1__1.000000e-01_1_0_1000000_0_0_0_0_0_genQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_7.mat'; %'FullRes_n1_m1_cs1__1.000000e-01_1_0_1000000_0_0_0_0_0_genAffineBias_unnorm_im1_randinit_sn1_g1_K13_depth0_6_7_run2.mat'; %'FullRes_n10_m0.0001_cs1__1.000000e-01_1_0_10000_0_1.000000e-02_0_0_0_genL1_unnorm_im1_tp1_sn1_g1_K13_depth0_6_21.mat'
%ResultFilename = 'FullRes_n10_m1_cs1__1.000000e-01_1_0_1000000_0_0_0_0_0_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth1_6_21.mat'; % 'FullRes_n20_m1_cs1__1.000000e-01_1_0_1000000_0_0_0_0_0_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_21.mat'; %'FullRes_n10_m1_cs1__1.000000e-01_1_0_1000000_0_0_0_0_0_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_21.mat';
%ResultFilename = 'FullRes_n1_m1_cs1__1.000000e-01_1_0_1000000_0_0_0_0_0_genQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_7_run2_run3_run4.mat'; %'FullRes_n10_m0.0001_cs1__1.000000e-01_1_0_10000_0_1.000000e-02_0_0_0_genL1_unnorm_im1_tp1_sn1_g1_K13_depth0_6_21.mat';
%ResultFilename = 'FullRes_n10_m1_cs1__1.000000e-01_1_0_1000000_0_0_0_0_0_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_21.mat'; %
%ResultFilename = 'FullRes_n10_m0.0001_cs1__1.000000e-01_1_0_10000_0_1.000000e-02_0_0_0_genL1_unnorm_im1_tp1_sn1_g1_K13_depth0_6_21.mat';

%ResultFilename = 'FullRes_n0_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_randinit_sn1_g1_K13_depth0_6_21_run2.mat'; %'FullRes_n0_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_randinit_sn1_g1_K13_depth0_6_21_run2_0.mat';

%ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_21.mat'
%ResultFilename = 'FullRes_n0_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_randinit_sn1_g1_K13_depth0_6_21_run2_run3.mat'; %'FullRes_n0_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_randinit_sn1_g1_K13_depth0_6_21_run2_0.mat';
%ResultFilename = 'FullRes_n0_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_randinit_sn1_g1_K13_depth-3_3_21_PSFSet_Zhou_pair_s0_-3_s1_3.mat'; %'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_21.mat'
%ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_21.mat'
%ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_21_fmingd.mat'

%ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_21_fmingdadam.mat'
%ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_21_fmingdadam_run2_run3.mat';
%ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K9_depth-3_3_21_PSFSet_Zhou_single1_s0_-3_s1_3_fmingdadam.mat';
%ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K9_depth0_12_21_PSFSet_Disk_single1_s0_0_s1_12_fmingdadam.mat'
%ResultFilename = 'FullRes_n0_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_randinit_sn1_g1_K13_depth-3_3_21_PSFSet_Zhou_single1_s0_-3_s1_3_fmingdadam_run2.mat';
%ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K9_depth-3_3_21_PSFSet_Zhou_single1_s0_-3_s1_3_fmingdadam_run2.mat'; %

%ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_12_21_PSFSet_Disk_single1_s0_0_s1_12_fmingdadam.mat';
%ResultFilename = 'FullRes_n0_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_randinit_sn1_g1_K13_depth-3_3_21_PSFSet_Zhou_single1_s0_-3_s1_3_fmingdadam_run2_run3.mat';
%ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_21_PSFSet_Zhou_single1_s0_0_s1_6_fmingdadam.mat';
%ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_21_fmingdadam_run2_run3_x4labelloss_run4.mat';
%ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_21_fmingdadam_run2_run3_run4.mat';

%ResultFilename = 'FullRes_n0_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_randinit_sn1_g1_K13_depth-3_3_21_PSFSet_Zhou_single1_s0_-3_s1_3_fmingdadam_run2_run3_x4labelloss_run4.mat'; %'disk_pair_13x13_1_10.mat'; %'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth5_15_21_PSFSet_Disk_pair_s0_5_s1_15_fmingdadam.mat'
%ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K23_depth1_10_19_PSFSet_Disk_single1_s0_1_s1_10_N19_fmingdadam_x2LL.mat';
%ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth1_10_19_PSFSet_Disk_single1_s0_1_s1_10_N19_fmingdadam_run2.mat';
%ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth1_10_19_PSFSet_Disk_single1_s0_1_s1_10_N19_fmingdadam_x4LabeLoss_run2.mat';
% ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth1_9_9_PSFSet_Levin_N9_fmingdadam.mat';
% ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K15_depth1_9_9_PSFSet_Levin_N9_fmingdadam.mat';
% ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth1_20_20_PSFSet_Levin_Synth_N20_fmingdadam.mat';
% %ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K15_depth1_9_9_PSFSet_Levin_N9_fmingdadam_run2.mat';
% %ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth1_9_9_PSFSet_Levin_N9_fmingdadam_200K.mat';
% ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K15_depth1_9_9_PSFSet_Levin_N9_fmingdadam_run2_run3.mat';
% ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth1_20_20_PSFSet_Levin_Synth_N20_fmingdadam_run2.mat';
% ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K21_depth1_20_20_PSFSet_Levin_Synth_N20_fmingdadam.mat';
ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth-3_3_11_NF5_PSFSet_Zhou_single1_s0_-3_s1_3_N11_fmingdadam_sgn.mat';
ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_randinit_sn1_g1_K13_depth-3_3_11_NF20_PSFSet_Zhou_single1_s0_-3_s1_3_N11_fmingdadam_sgn_x10LL.mat';
ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth1_20_20_PSFSet_Levin_Synth_N20_NF10_fmingdadam_x4LLossrun2.mat';

ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth-3_3_11_PSFSet_Zhou_single1_s0_-3_s1_3_N11_fmingdadam_sgn.mat';
ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_randinit_sn1_g1_K15_depth-3_3_11_NF20_PSFSet_Zhou_single1_s0_-3_s1_3_N11_fmingdadam.mat';
ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K15_depth609.6_1500_27_NF10_PSFSet_QPPSF_2F_zNear700_zFar1219.2_f22_fmingdadam.mat';
ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth609.6_1500_27_NF5_PSFSet_QPPSF_2F_zNear700_zFar1219.2_f22_fmingdadam_run2.mat';
ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K21_depth1_20_20_PSFSet_Levin_Synth_N20_NF10_fmingdadam_run2.mat';
ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K3_depth0.529_0.869_51_NF10_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam.mat';
ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K3_depth0.529_0.869_51_NF10_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam_run2.mat';
% %ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth1_20_20_PSFSet_Levin_Synth_N20_NF10_fmingdadam_run2_run3.mat';
ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth1_20_20_PSFSet_Levin_Synth_N20_NF10_fmingdadam_run2_run3_run4.mat';
%ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K9_depth0.529_0.869_51_NF10_PSFSet_breakfast_single1_s0_0.529_s1_0.869_N51_fmingdadam.mat';

% ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth1_20_20_PSFSet_Levin_Synth_N20_NF10_fmingdadam_run2_run3_run4_run5.mat';
% ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_randinit_sn1_g1_K21_depth-5_5_11_NF10_PSFSet_Zhou_single1_s0_-5_s1_5_N11_fmingdadam_run2_run3_run4.mat'
% ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1.000000e-03_genLogQuad_unnorm_tp1_sn1_g1_K21_depth-5_5_11_NF10_PSFSet_Zhou_single1_s0_-5_s1_5_N11_fmingdadam_run2.mat';
% %ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K21_depth-5_5_21_NF10_PSFSet_Zhou_single1_s0_-5_s1_5_N21_fmingdadam_run2.mat'
% ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K21_depth-5_5_11_NF20_PSFSet_ZhouSingle_s0_-5_s1_5_N11_fmingdadam.mat';
% ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_randinit_sn1_g1_K21_depth-5_5_11_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N11_fmingdadam.mat'
% ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K21_depth-5_5_11_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N11_fmingdadam.mat'
% %ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K21_depth-5_5_5_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N5_fmingdadam_run2.mat'
% %ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_nsinvpsf_sn1_g1_K21_depth-5_5_5_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N5_fmingdadam.mat'
% ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K21_depth-5_5_5_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N5_fmingdadam_run2_run3.mat'
% ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_nsinvpsf_sn1_g1_K21_depth-5_5_5_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N5_fmingdadam_run2.mat' %_run3
% ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K21_depth-5_5_11_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N11_fmingdadam_bgdrun2.mat'
% ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_nsinvpsf_sn1_g1_K21_depth-5_5_11_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N11_fmingdadam.mat'
% ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K21_depth-5_5_11_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N11_fmingdadam_run2_run3.mat'
% ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K21_depth-5_5_11_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N11_fmingdadam_run2_run3_noise0p05run4.mat'
% ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K21_depth-5_5_11_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N11_fmingdadam_run2_run3_noise0p01run4.mat'
% ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K21_depth-5_5_11_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N11_fmingdadam_run2_run3_noise0p01run4_run5.mat'
% ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K21_depth1_20_20_PSFSet_Levin_Synth_N20_NF10_fmingdadam_run2.mat';
% ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K5_depth0.529_0.869_51_NF3_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam_v2.mat';
% ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_fileinit_sn1_g1_K3_depth0.529_0.869_51_NF10_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam_v2.mat';
% ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_randinit_sn1_g1_K21_depth-5_0_11_NF10_PSFSet_Zhou_single1_s0_-5_s1_0_N11_fmingdadam_run2_run3_run4.mat'
% ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_randinit_sn1_g1_K21_depth-5_0_11_NF10_PSFSet_Zhou_single1_s0_-5_s1_0_N11_fmingdadam.mat'
%ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_fileinit_sn1_g1_K5_depth0.9_1.1_51_NF10_PSFSet_dancing_pair_s0_0.9_s1_1.1_N51_RelSigmaSpace_fmingdadam.mat';
%ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K7_depth0.9_1.1_51_NF3_PSFSet_dancing_pair_s0_0.9_s1_1.1_N51_ZSpace_fmingdadam.mat';
ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K3_depth0.529_0.869_51_NF3_PSFSet_WatanabeNayar_pair_s0_0.529_s1_0.869_N51_ZSpace_fmingdadam.mat';
ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth1_20_20_PSFSet_Levin_Synth_N20_fmingdadam_run2.mat';
ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth1_20_20_PSFSet_Levin_Synth_N20_fmingdadam_run2_run3.mat';
ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth1_20_20_PSFSet_Levin_Synth_N20_NF10_fmingdadam_run2_run3_run4.mat';
ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth1_20_20_PSFSet_Levin_Synth_N20_NF10_fmingdadam_run2_run3_run4_x4LLrun5.mat';
ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth1_20_20_PSFSet_Levin_Synth_N20_NF10_fmingdadam_run2_run3_run4_run5WFn.mat'
ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth1_20_20_PSFSet_Levin_Synth_N20_NF10_fmingdadam_run2_run3_run4_run5x0.mat'
ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth1_20_20_PSFSet_Levin_Synth_N20_NF10_fmingdadam_run2_run3_run4_run5.mat'
ResultFilename = ...
    'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_sn1_g1_K7_depth0.9_1.1_51_NF3_PSFSet_dancing_pair_s0_0.9_s1_1.1_N51_ZSpace_fmingdadam_run2.mat'
ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_fileinit_sn1_g1_K3_depth0.529_0.869_51_NF10_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam_v2_run2.mat';
ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_fileinit_sn1_g1_K3_depth0.529_0.869_51_NF10_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam_v2_run2_run3.mat';
ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_fileinit_sn1_g1_K3_depth0.529_0.869_51_NF10_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam_v2_run2_run3_run4.mat';
ResultFilename = ...
    'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_fileinit2_sn1_g1_K3_depth0.529_0.869_51_NF3_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam_v2.mat'
ResultFilename = 'FullRes_n1_m1_cs1__0_0_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_tp1_fileinit2_sn1_g1_K3_depth0.529_0.869_51_NF3_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam_v2_r2.mat';
%%%%%%%%%%%%
% outDir = 'plots_K3_depth0.529_0.869_51_NF10_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51'; %'plots_K13_depth1_20_20_PSFSet_Levin_Synth_N20_NF10_fmingdadam_run2_run3'; %
% outDir = 'plots_K3_depth0.529_0.869_51_NF10_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_run2'; %
% outDir = 'plots_K13_depth1_20_20_PSFSet_Levin_Synth_N20_NF10_fmingdadam_run2_run3_run4';
% outDir = 'plots_K21_depth1_20_20_PSFSet_Levin_Synth_N20_NF10_fmingdadam_run2'; %
%outDir = 'plots_K9_depth0.529_0.869_51_NF10_PSFSet_breakfast_single1_s0_0.529_s1_0.869_N51';
% outDir = 'plots_K13_depth1_20_20_PSFSet_Levin_Synth_N20_NF10_fmingdadam_run2_run3_run4_run5';
% outDir = 'plots_randinit_K21_depth-5_5_11_NF10_PSFSet_Zhou_single1_s0_-5_s1_5_N11_fmingdadam_run2_run3_run4';
% outDir = 'plots_tp1_K21_depth-5_5_11_NF10_PSFSet_Zhou_single1_s0_-5_s1_5_N11_fmingdadam_run2'
% %outDir = 'plots_tp1_K21_depth-5_5_11_NF10_PSFSet_Zhou_single1_s0_-5_s1_5_N21_fmingdadam_run2'
% outDir = 'plots_tp1_K21_depth-5_5_11_NF20_PSFSet_ZhouSingle_s0_-5_s1_5_N11_fmingdadam'
% outDir = 'plots_randinit_K21_depth-5_5_11_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N11_fmingdadam'
% outDir = 'plots_tp1_K21_depth-5_5_11_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N11_fmingdadam'
% %outDir = 'plots_tp1_K21_depth-5_5_5_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N5_fmingdadam_run2'
% %outDir =
% %'plots_nsinvpsf_K21_depth-5_5_5_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N5_fmingdadam'
% outDir = 'plots_tp1_K21_depth-5_5_5_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N5_fmingdadam_run2_run3'
% outDir = 'plots_nsinvpsf_sn1_g1_K21_depth-5_5_5_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N5_fmingdadam_run2' %_run3
% outDir = 'plots_tp1_sn1_g1_K21_depth-5_5_11_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N11_fmingdadam_bgdrun2'
% outDir = 'plots_nsinvpsf_sn1_g1_K21_depth-5_5_11_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N11_fmingdadam'
% outDir = 'plots_tp1_sn1_g1_K21_depth-5_5_11_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N11_fmingdadam_run2_run3'
% outDir = 'plots_tp1_sn1_g1_K21_depth-5_5_11_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N11_fmingdadam_run2_run3_noise0p05run4'
% outDir = 'plots_tp1_sn1_g1_K21_depth-5_5_11_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N11_fmingdadam_run2_run3_noise0p01run4'
% outDir = 'plots_tp1_sn1_g1_K21_depth-5_5_11_NF20_PSFSet_Zhou_single1_s0_-5_s1_5_N11_fmingdadam_run2_run3_noise0p01run4_run5'
% outDir = 'plots_tp1_sn1_g1_K5_depth0.529_0.869_51_NF3_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam_v2';
% outDir = 'plots_tp1_fileinit_sn1_g1_K3_depth0.529_0.869_51_NF10_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam_v2'
% outDir = ['plots_randinit_sn1_g1_K21_depth-5_0_11_NF10_PSFSet_Zhou_single1_s0_-5_s1_0_N11_fmingdadam_run2_run3_run4']
% outDir = 'plots_randinit_sn1_g1_K21_depth-5_0_11_NF10_PSFSet_Zhou_single1_s0_-5_s1_0_N11_fmingdadam'
%outDir = 'plots_tp1_fileinit_sn1_g1_K5_depth0.9_1.1_51_NF10_PSFSet_dancing_pair_s0_0.9_s1_1.1_N51_RelSigmaSpace_fmingdadam';
%outDir = 'plots_tp1_sn1_g1_K7_depth0.9_1.1_51_NF3_PSFSet_dancing_pair_s0_0.9_s1_1.1_N51_ZSpace_fmingdadam'
outDir = 'plots_tp1_sn1_g1_K3_depth0.529_0.869_51_NF3_PSFSet_WatanabeNayar_pair_s0_0.529_s1_0.869_N51_ZSpace_fmingdadam';
outDir = 'plots_ver2_tp1_sn1_g1_K13_depth1_20_20_PSFSet_Levin_Synth_N20_fmingdadam_run2';
outDir = 'plots_ver2_tp1_sn1_g1_K13_depth1_20_20_PSFSet_Levin_Synth_N20_fmingdadam_run2_run3';
outDir = 'plots_tp1_sn1_g1_K13_depth1_20_20_PSFSet_Levin_Synth_N20_NF10_fmingdadam_run2_run3_run4';
outDir = 'plots_tp1_sn1_g1_K13_depth1_20_20_PSFSet_Levin_Synth_N20_NF10_fmingdadam_run2_run3_run4_x4LLrun5';
outDir = 'plots_tp1_sn1_g1_K13_depth1_20_20_PSFSet_Levin_Synth_N20_NF10_fmingdadam_run2_run3_run4_run5WFn';
outDir = 'plots_tp1_sn1_g1_K13_depth1_20_20_PSFSet_Levin_Synth_N20_NF10_fmingdadam_run2_run3_run4_run5x0';
outDir = 'plots_tp1_sn1_g1_K13_depth1_20_20_PSFSet_Levin_Synth_N20_NF10_fmingdadam_run2_run3_run4_run5';
outDir = ...
    'plots_tp1_sn1_g1_K7_depth0.9_1.1_51_NF3_PSFSet_dancing_pair_s0_0.9_s1_1.1_N51_ZSpace_fmingdadam_run2';
outDir = 'plots_tp1_fileinit_sn1_g1_K3_depth0.529_0.869_51_NF10_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam_v2_run2'
outDir = ...
    'plots_tp1_fileinit_sn1_g1_K3_depth0.529_0.869_51_NF10_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam_v2_run2_run3'
outDir = 'plots_tp1_fileinit_sn1_g1_K3_depth0.529_0.869_51_NF10_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam_v2_run2_run3_run4'
outDir = ...
    'plots_tp1_fileinit2_sn1_g1_K3_depth0.529_0.869_51_NF3_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam_v2';
outDir = 'plots_tp1_fileinit2_sn1_g1_K3_depth0.529_0.869_51_NF3_PSFSet_breakfast_pair_s0_0.529_s1_0.869_N51_fmingdadam_v2_r2';
if(~exist(outDir, 'dir'))
    mkdir(outDir)
end

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

if(isfield(ResData, 'Im_seq'))
    Im_seq = ResData.Im_seq;
elseif(isfield(ResData.Dataset, 'Im_seq'))
    Im_seq = ResData.Dataset.Im_seq;
end
   
if(isfield(ResData, 'KernelSet'))
  KernelSet = ResData.KernelSet;
else
  if(isfield(ResData, 'KernelSetDisk'))
    KernelSet = ResData.KernelSetDisk;
  elseif(isfield(ResData, 'KernelSet'))
    KernelSet = ResData.KernelSet;
  elseif(isfield(ResData, 'Dataset'))
          if(isfield(ResData.Dataset, 'KernelSet'))
              KernelSet = ResData.Dataset.KernelSet;
          elseif(isfield(ResData.Dataset, 'KernelSetDisk'))
              KernelSet = ResData.Dataset.KernelSetDisk;
          end
  end
end
% if(isfield(ResData, 'ImageDepthSet'))
%    ImageDepthSet = ResData.ImageDepthSet;
% else % old version
%     ImageDepthSet = BuildTrainingSet(Im_seq, KernelSet, ResData.params);
% end
if(isfield(ResData, 'ImageDepthSet'))
   ImageDepthSet = ResData.ImageDepthSet;
else % old version
    if(~isfield(ResData, 'Im_seq'))
        Im_seq = ResData.Dataset.Im_seq;
    else
        Im_seq = ResData.Im_seq;
    end

    if(isfield(ResData, 'KernelSet'))
      KernelSet = ResData.KernelSet;
    else
      if(~isfield(ResData, 'KernelSetDisk'))
        KernelSet = ResData.Dataset.KernelSetDisk;
      else
        KernelSet = ResData.KernelSetDisk;
      end
    end
    ImageDepthSet = BuildTrainingSet(Im_seq, KernelSet, ResData.params);
end
%%% This is not safe. Because files may change or get deleted !!!
% if(~exist('KernelSet', 'var'))
%     if(ResData.bLoadPSF)
%         PSFFile = load([ResData.PSFFilename '.mat']);
% 
%         if(ResData.bPSFSetFormat)
%             KernelSet = PSFFile.PSFSet(:,ResData.PSFSetIndicesToUse);
%         else
%             KernelSet = PSFFile.KernelSet;
%         end
%     end
% end
if(~exist('KernelSet', 'var'))
   % loop throught dataset
   Dataset = ResData.Dataset;
   while(1)
       if(~isfield(Dataset, 'Dataset'))
           break;
       end
       Dataset = Dataset.Dataset;
   end
   KernelSet = Dataset.KernelSet;
end
% if(isfield(ResData, 'KernelSetDisk'))
% KernelSet = ResData.KernelSetDisk;
% else
% KernelSet = ResData.KernelSet;
% end
% %NPairs = 1;
% %
% 
% Im_seq = ResData.Im_seq;
fnTform = @(x) x; % del2(x)
if(isfield(ResData, 'fnTform'))
    fnTform = ResData.fnTform;
end

if(~exist('Im_seq', 'var'))
    ImFilename = {'brown_noise_pattern_512x512.png', 'D42_512x512.png',  'D94_512x512.png', 'merry_mtl07_023.tif'};
    Im_seq = cell(1, length(ImFilename));
    for idx = 1:length(ImFilename)
        Im_seq{idx} = fnTform(mean(im2double(imread(ImFilename{idx})), 3));
    end 
end

%%
params = ResData.params;
params.NMaxTraining = 1000;
%% Test seq
ImTestFilename = {'D13.gif' , 'merry_mtl07_053.tif', 'pippin_Peel071.tif'}; % 'D11.gif'
ImTest_seq = cell(1, length(ImTestFilename));
for idx = 1:length(ImTestFilename)
    ImTest_seq{idx} = fnTform(mean(im2double(imread(ImTestFilename{idx})), 3));
end
TestImageDepthSet = BuildTrainingSet(ImTest_seq, KernelSet, ResData.params);

%%
%resultNS = ResParam; %ResData.Dataset.result; %TrainNullSpaceFilters(Im_seq, KernelSet, params)
resultNS = TrainNullSpaceFilters(ImageDepthSet, params)
paramsRS = params;
paramsRS.fnNullSpace = @findLeftRankSpaceFiltersPerDepth;
resultRS = TrainNullSpaceFilters(ImageDepthSet, paramsRS)

% params.KR = KernelSize;
% params.KC = KernelSize;
% params.NMaxTraining = NMaxTraining;
% params.NFilters = 20;
% Im_seq = {im};
if(~bSkipNSInvPSF) % may be needed if e.g. blur is computed using heat diffusion
paramsNSInvPSF = params;
paramsNSInvPSF.PSFEnergyThreshold = .9;
paramsNSInvPSF.PatchSize = params.KR;
%paramsNSInvPSF.NMaxTraining = 2000;
%paramsNSInvPSF.NFilters = 200;
resNSInvPSF = TrainNullSpaceInvPSFFilters(Im_seq, KernelSet, paramsNSInvPSF);
end
if(bEvalLargeFilterSet)
    LargeFilterNums =  min(MAXLARGEFILTER, size(Filters, 2));
    display(['NFilters for Large : ' num2str(LargeFilterNums)])
    paramsNSLarge = params;
    paramsNSLarge.NFilters = LargeFilterNums;
    resultNSLarge = TrainNullSpaceFilters(ImageDepthSet, paramsNSLarge)
    paramsRSLarge = paramsRS;
    paramsRSLarge.NFilters = paramsNSLarge.NFilters;
    resultRSLarge = TrainNullSpaceFilters(ImageDepthSet, paramsRSLarge)
    
end
%%
%params.NMaxTraining = 2;
TrainSet = ImageDepthSet; %BuildTrainingSet(Im_seq, KernelSet, params);
TestSet = TestImageDepthSet; %BuildTrainingSet(ImTest_seq, KernelSet, params);
%%
params = ResData.params;
if(~isfield(params, 'NPairs'))
    params.NPairs = NPairs;
    
end
Lambdas_old = params.LAMBDAS;
%params.M0 = 1; %.1; %0.5;
%params.LAMBDAS = [1, 1, 0, 0, 10, 0, 0, 0, 0];
%params.LAMBDAS = [0, 0, 0, 0, 1e4, 0, 0, 0, 0];
%
%params = ResData.params;
%params.fnCost = @L1CostVec; %@QuadraticCostVec; %
%params.M0 = 0; %.01;
fnLabelLoss = @(x) x.^2;
TrainSetData = genDataMatrix(TrainSet, fnLabelLoss);
TestSetData = genDataMatrix(TestSet, fnLabelLoss);
W_NS = convertFilter3Dto2DFormat(resultNS.FilterVecs);
W0 = convertFilter3Dto2DFormat(params.W0);
WFinal = ResParam.W;

Scale = 1; %/10;
W_NS = W_NS; % * ResData.params.FiniteNormConst * Scale;
W0 = W0 * Scale;
WFinal = WFinal * Scale;
params.PerLabelLoss = TrainSetData.PerLabelLoss;
ResParam.fnObj(TrainSetData.X_flat, W_NS, TrainSetData.y, ResParam.A, params, ResParam.NFilters, TrainSetData.DataDim, ResParam.NClasses)
ResParam.fnObj(TrainSetData.X_flat, W0, TrainSetData.y, ResParam.A, params, ResParam.NFilters, TrainSetData.DataDim, ResParam.NClasses)
ResParam.fnObj(TrainSetData.X_flat, ResParam.W, TrainSetData.y, ResParam.A, params, ResParam.NFilters, TrainSetData.DataDim, ResParam.NClasses)

params.PerLabelLoss = TestSetData.PerLabelLoss;
ResParam.fnObj(TestSetData.X_flat, W_NS, TestSetData.y, ResParam.A, params, ResParam.NFilters, TestSetData.DataDim, ResParam.NClasses)
ResParam.fnObj(TestSetData.X_flat, W0, TestSetData.y, ResParam.A, params, ResParam.NFilters, TestSetData.DataDim, ResParam.NClasses)
ResParam.fnObj(TestSetData.X_flat, ResParam.W, TestSetData.y, ResParam.A, params, ResParam.NFilters, TestSetData.DataDim, ResParam.NClasses)
%% FIXED NOISE DATASET GENERATION EVALUATION 
NoiseStd = 0.01; %.01;

%Im_seq_noise1 = Im_seq;
% for it = 1:size(Im_seq_noise1, 2)
%    Im_seq_noise1{it} = Im_seq_noise1{it} + NoiseStd * randn(size(Im_seq_noise1{it}));
% end
TrainSetNoise = BuildTrainingSet(Im_seq, KernelSet, params);
for it = 1:size(TrainSetNoise, 2)
   TrainSetNoise{it} = TrainSetNoise{it} + NoiseStd * randn(size(TrainSetNoise{it}));
end
%%
% ImTest_seq_noise = ImTest_seq;
% for it = 1:size(ImTest_seq_noise, 2)
%    ImTest_seq_noise{it} = ImTest_seq{it} + NoiseStd * randn(size(ImTest_seq{it}));
% end
TestSetNoise = BuildTrainingSet(ImTest_seq, KernelSet, params);
for it = 1:size(TestSetNoise, 2)
   TestSetNoise{it} = TestSetNoise{it} + NoiseStd * randn(size(TestSetNoise{it}));
end
%%
resultEvalNS = resultNS.FilterVecs; % 
resultEvalRS = resultRS.FilterVecs;
resultEval = ResParam.FilterVecs; %
resultEvalInit = params.W0;
NClasses = ResParam.NClasses;
fnCost = ResParam.fnCost; %@(a, b, c) LogEnergy(a,b,c, @L1CostVec); %@L1CostVec; %@(a, b, c) LogEnergy(a,b,c, @QuadraticCostVec); %
CostScale = 1; % 1/10;
%
Train_unnorm = EvalFilterCost(TrainSet, resultEval, fnCost, @min, CostScale)    
Test_unnorm = EvalFilterCost(TestSet, resultEval, fnCost, @min)
Train_unnorm_noise = EvalFilterCost(TrainSetNoise, resultEval, fnCost, @min)
Test_unnorm_noise = EvalFilterCost(TestSetNoise, resultEval, fnCost, @min)
%%% W0 Init
Train_unnorm_init = EvalFilterCost(TrainSet, resultEvalInit, fnCost, @min, CostScale)    
Test_unnorm_init = EvalFilterCost(TestSet, resultEvalInit, fnCost, @min)
Train_unnorm_noise_init = EvalFilterCost(TrainSetNoise, resultEvalInit, fnCost, @min)
Test_unnorm_noise_init = EvalFilterCost(TestSetNoise, resultEvalInit, fnCost, @min)
%
%%%
Train_unnorm_NS = EvalFilterCost(TrainSet, resultEvalNS, fnCost, @min, CostScale)    
Test_unnorm_NS = EvalFilterCost(TestSet, resultEvalNS, fnCost, @min)
Train_unnorm_noise_NS = EvalFilterCost(TrainSetNoise, resultEvalNS, fnCost, @min)
Test_unnorm_noise_NS = EvalFilterCost(TestSetNoise, resultEvalNS, fnCost, @min)

Train_unnorm_RS = EvalFilterCost(TrainSet, resultEvalRS, fnCost, @max, CostScale)    
Test_unnorm_RS = EvalFilterCost(TestSet, resultEvalRS, fnCost, @max)
Train_unnorm_noise_RS = EvalFilterCost(TrainSetNoise, resultEvalRS, fnCost, @max)
Test_unnorm_noise_RS = EvalFilterCost(TestSetNoise, resultEvalRS, fnCost, @max)

%%%
if(~bSkipNSInvPSF)
Train_NSInv = EvalFilterCost(TrainSet, resNSInvPSF.FilterVecs, fnCost, @min, CostScale)    
Test_NSInv = EvalFilterCost(TestSet, resNSInvPSF.FilterVecs, fnCost, @min)
Train_noise_NSInv = EvalFilterCost(TrainSetNoise, resNSInvPSF.FilterVecs, fnCost, @min)
Test_noise_NSInv = EvalFilterCost(TestSetNoise, resNSInvPSF.FilterVecs, fnCost, @min)

Train_InvPSF = EvalFilterCost(TrainSet, resNSInvPSF.FilterVecsInvPSF, fnCost, @min, CostScale)    
Test_InvPSF = EvalFilterCost(TestSet, resNSInvPSF.FilterVecsInvPSF, fnCost, @min)
Train_noise_InvPSF = EvalFilterCost(TrainSetNoise, resNSInvPSF.FilterVecsInvPSF, fnCost, @min)
Test_noise_InvPSF = EvalFilterCost(TestSetNoise, resNSInvPSF.FilterVecsInvPSF, fnCost, @min)
else
    Train_NSInv = [];
    Test_NSInv = [];
    Train_noise_NSInv = [];
    Test_noise_NSInv = [];
    
    Train_InvPSF = [];
    Test_InvPSF = [];
    Train_noise_InvPSF = [];
    Test_noise_InvPSF = [];

end
%%% Larger number of filters
if(bEvalLargeFilterSet)
Train_NS_L = EvalFilterCost(TrainSet, resultNSLarge.FilterVecs, fnCost, @min, CostScale)    
Test_NS_L = EvalFilterCost(TestSet, resultNSLarge.FilterVecs, fnCost, @min)
Train_noise_NS_L = EvalFilterCost(TrainSetNoise, resultNSLarge.FilterVecs, fnCost, @min)
Test_noise_NS_L = EvalFilterCost(TestSetNoise, resultNSLarge.FilterVecs, fnCost, @min)

Train_RS_L = EvalFilterCost(TrainSet, resultRSLarge.FilterVecs, fnCost, @max, CostScale)    
Test_RS_L = EvalFilterCost(TestSet, resultRSLarge.FilterVecs, fnCost, @max)
Train_noise_RS_L = EvalFilterCost(TrainSetNoise, resultRSLarge.FilterVecs, fnCost, @max)
Test_noise_RS_L = EvalFilterCost(TestSetNoise, resultRSLarge.FilterVecs, fnCost, @max)
else
    Train_NS_L = [];
    Test_NS_L = [];
    Train_noise_NS_L = [];
    Test_noise_NS_L = [];

    Train_RS_L = [];
    Test_RS_L = [];
    Train_noise_RS_L = [];
    Test_noise_RS_L = [];
end
FullPath = [pwd '/' outDir];
%%
% save([outDir '/Res.mat'], 'COMMENTS', 'ResultFilename', 'FullPath', 'NClasses', 'Train_unnorm', 'Test_unnorm', 'Train_unnorm_noise', ...
%         'Test_unnorm_noise', 'Train_unnorm_init', 'Test_unnorm_init', ...
%         'Train_unnorm_noise_init', 'Test_unnorm_noise_init', 'Train_unnorm_NS', ...
%         'Test_unnorm_NS', 'Train_unnorm_noise_NS', 'Test_unnorm_noise_NS', ...
%         'Train_unnorm_RS', 'Test_unnorm_RS', 'Train_unnorm_noise_RS', ...
%         'Test_unnorm_noise_RS', 'Train_NSInv', 'Test_NSInv', 'Train_noise_NSInv', ...
%         'Test_noise_NSInv', 'Train_NS_L', 'Test_NS_L', 'Train_noise_NS_L', ...
%         'Test_noise_NS_L', 'Train_RS_L', 'Test_RS_L', 'Train_noise_RS_L', ...
%         'Test_noise_RS_L', 'Train_InvPSF', 'Test_InvPSF', 'Train_noise_InvPSF', 'Test_noise_InvPSF');
% save([outDir '/Res.mat'], 'COMMENTS', 'ResultFilename', 'FullPath', 'NClasses', 'Train_unnorm', 'Test_unnorm', 'Train_unnorm_noise', ...
%         'Test_unnorm_noise', 'Train_unnorm_init', 'Test_unnorm_init', ...
%         'Train_unnorm_noise_init', 'Test_unnorm_noise_init', 'Train_unnorm_NS', ...
%         'Test_unnorm_NS', 'Train_unnorm_noise_NS', 'Test_unnorm_noise_NS', ...
%         'Train_unnorm_RS', 'Test_unnorm_RS', 'Train_unnorm_noise_RS', ...
%         'Test_unnorm_noise_RS', 'Train_NSInv', 'Test_NSInv', 'Train_noise_NSInv', ...
%         'Test_noise_NSInv', ...
%         'Train_InvPSF', 'Test_InvPSF', 'Train_noise_InvPSF', 'Test_noise_InvPSF');
        %         'Train_NS_L', 'Test_NS_L', 'Train_noise_NS_L', ...
%         'Test_noise_NS_L', 'Train_RS_L', 'Test_RS_L', 'Train_noise_RS_L', ...
%         'Test_noise_RS_L', ...
%save([outDir '/ResBasic.mat'], 'COMMENTS', 'ResultFilename', 'FullPath', 'NClasses', 'Train_unnorm', 'Test_unnorm', ...
%       'Train_unnorm_init', 'Test_unnorm_init', ...
%       'Train_unnorm_NS', 'Test_unnorm_NS', ...
%       'Train_unnorm_RS', 'Test_unnorm_RS');

save([outDir '/ResBasic.mat'], 'COMMENTS', 'ResultFilename', 'FullPath', 'NClasses', 'Train_unnorm', 'Test_unnorm', ...
        'Train_unnorm_init', 'Test_unnorm_init', ...
        'Train_unnorm_NS', 'Test_unnorm_NS', ...
        'Train_unnorm_RS', 'Test_unnorm_RS', ...
        'Train_NS_L', 'Test_NS_L', 'Train_RS_L', 'Test_RS_L', 'MAXLARGEFILTER');

%%
display(['(NS) Train : ' num2str(Train_unnorm_NS.TEPercent) '% , Train Noise: ' num2str(Train_unnorm_noise_NS.TEPercent) '%']);
display(['(NS) Test : ' num2str(Test_unnorm_NS.TEPercent) '% , Test Noise: ' num2str(Test_unnorm_noise_NS.TEPercent) '%']);
display(['(RS) Train : ' num2str(Train_unnorm_RS.TEPercent) '% , Train Noise: ' num2str(Train_unnorm_noise_RS.TEPercent) '%']);
display(['(RS) Test : ' num2str(Test_unnorm_RS.TEPercent) '% , Test Noise: ' num2str(Test_unnorm_noise_RS.TEPercent) '%']);
%%%%
if(bEvalLargeFilterSet)
display(['(NS Large) Train : ' num2str(Train_NS_L.TEPercent) '% , Train Noise: ' num2str(Train_noise_NS_L.TEPercent) '%']);
display(['(NS Large) Test : ' num2str(Test_NS_L.TEPercent) '% , Test Noise: ' num2str(Test_noise_NS_L.TEPercent) '%']);
display(['(RS Large) Train : ' num2str(Train_RS_L.TEPercent) '% , Train Noise: ' num2str(Train_noise_RS_L.TEPercent) '%']);
display(['(RS Large) Test : ' num2str(Test_RS_L.TEPercent) '% , Test Noise: ' num2str(Test_noise_RS_L.TEPercent) '%']);
end
%%%%
if(~bSkipNSInvPSF)
display(['(NSInvPSF) Train : ' num2str(Train_NSInv.TEPercent) '% , Train Noise: ' num2str(Train_noise_NSInv.TEPercent) '%']);
display(['(NSInvPSF) Test : ' num2str(Test_NSInv.TEPercent) '% , Test Noise: ' num2str(Test_noise_NSInv.TEPercent) '%']);

display(['(InvPSF) Train : ' num2str(Train_InvPSF.TEPercent) '% , Train Noise: ' num2str(Train_noise_InvPSF.TEPercent) '%']);
display(['(InvPSF) Test : ' num2str(Test_InvPSF.TEPercent) '% , Test Noise: ' num2str(Test_noise_InvPSF.TEPercent) '%']);
end
%%%%
display(['(init) Train : ' num2str(Train_unnorm_init.TEPercent) '% , Train Noise: ' num2str(Train_unnorm_noise_init.TEPercent) '%']);
display(['(init) Test : ' num2str(Test_unnorm_init.TEPercent) '% , Test Noise: ' num2str(Test_unnorm_noise_init.TEPercent) '%']);
display(['Train : ' num2str(Train_unnorm.TEPercent) '% , Train Noise: ' num2str(Train_unnorm_noise.TEPercent) '%']);
display(['Test : ' num2str(Test_unnorm.TEPercent) '% , Test Noise: ' num2str(Test_unnorm_noise.TEPercent) '%']);
%
display('Mean RMSE')
display(['(NS) Train : ' num2str(mean(Train_unnorm_NS.RMSE)) ' , Train Noise: ' num2str(mean(Train_unnorm_noise_NS.RMSE))]);
display(['(NS) Test : ' num2str(mean(Test_unnorm_NS.RMSE)) ' , Test Noise: ' num2str(mean(Test_unnorm_noise_NS.RMSE))]);
display(['(RS) Train : ' num2str(mean(Train_unnorm_RS.RMSE)) ' , Train Noise: ' num2str(mean(Train_unnorm_noise_RS.RMSE))]);
display(['(RS) Test : ' num2str(mean(Test_unnorm_RS.RMSE)) ' , Test Noise: ' num2str(mean(Test_unnorm_noise_RS.RMSE))]);
%%%%%
if(bEvalLargeFilterSet)
display(['(NS Large) Train : ' num2str(mean(Train_NS_L.RMSE)) ' , Train Noise: ' num2str(mean(Train_noise_NS_L.RMSE))]);
display(['(NS Large) Test : ' num2str(mean(Test_NS_L.RMSE)) ' , Test Noise: ' num2str(mean(Test_noise_NS_L.RMSE))]);
display(['(RS Large) Train : ' num2str(mean(Train_RS_L.RMSE)) ' , Train Noise: ' num2str(mean(Train_noise_RS_L.RMSE))]);
display(['(RS Large) Test : ' num2str(mean(Test_RS_L.RMSE)) ' , Test Noise: ' num2str(mean(Test_noise_RS_L.RMSE))]);
end
%%%%
if(~bSkipNSInvPSF)
display(['(NSInvPSF) Train : ' num2str(mean(Train_NSInv.RMSE)) ' , Train Noise: ' num2str(mean(Train_noise_NSInv.RMSE))]);
display(['(NSInvPSF) Test : ' num2str(mean(Test_NSInv.RMSE)) ' , Test Noise: ' num2str(mean(Test_noise_NSInv.RMSE))]);

display(['(InvPSF) Train : ' num2str(mean(Train_InvPSF.RMSE)) ' , Train Noise: ' num2str(mean(Train_noise_InvPSF.RMSE))]);
display(['(InvPSF) Test : ' num2str(mean(Test_InvPSF.RMSE)) ' , Test Noise: ' num2str(mean(Test_noise_InvPSF.RMSE))]);
end
%%%%
display(['(init) Train : ' num2str(mean(Train_unnorm_init.RMSE)) ' , Train Noise: ' num2str(mean(Train_unnorm_noise_init.RMSE))]);
display(['(init) Test : ' num2str(mean(Test_unnorm_init.RMSE)) ' , Test Noise: ' num2str(mean(Test_unnorm_noise_init.RMSE))]);
display(['Train : ' num2str(mean(Train_unnorm.RMSE)) ' , Train Noise: ' num2str(mean(Train_unnorm_noise.RMSE))]);
display(['Test : ' num2str(mean(Test_unnorm.RMSE)) ' , Test Noise: ' num2str(mean(Test_unnorm_noise.RMSE))]);
%%
h = figure; 
imagesc(Train_unnorm.fnMaxMinNorm(Train_unnorm.CostSum))
xlabel('Ground Truth', 'fontsize', 18)
ylabel('Estimated Label', 'fontsize', 18)
title('Training Average Cost (Optimized)', 'fontsize', 18)
saveas(h, [outDir '/TrainMeanCostOptimized.png'])
print(h, '-depsc', [outDir '/TrainMeanCostOptimized.eps'])

h = figure; 
imagesc(Train_unnorm_NS.fnMaxMinNorm(Train_unnorm_NS.CostSum))
xlabel('Ground Truth', 'fontsize', 18)
ylabel('Estimated Label', 'fontsize', 18)
title('Training Average Cost (Null Space)', 'fontsize', 18)
saveas(h, [outDir '/TrainMeanCostNS.png'])
print(h, '-depsc', [outDir '/TrainMeanCostNS.eps'])
%%
h = figure;
hold on
plot(1:NClasses, 1:NClasses, 'k')
plot(1:NClasses, Train_unnorm.MeanEstLabel, 'r')
plot(1:NClasses, Train_unnorm_noise.MeanEstLabel, 'r--')
plot(1:NClasses, Test_unnorm.MeanEstLabel, 'b')
plot(1:NClasses, Test_unnorm_noise.MeanEstLabel, 'b--')

plot(1:NClasses, Train_unnorm_NS.MeanEstLabel, 'm')
plot(1:NClasses, Train_unnorm_noise_NS.MeanEstLabel, 'm--')
plot(1:NClasses, Test_unnorm_NS.MeanEstLabel, 'g')
plot(1:NClasses, Test_unnorm_noise_NS.MeanEstLabel, 'g--')

axis equal
legend('GT', 'Train', 'Train+noise', 'Test', 'Test+noise', ...
    'Train (NS)', 'Train+noise (NS)', 'Test (NS)', 'Test+noise (NS)', ...
     'location', 'SouthEast');
%title('Single Image Blur Estimation', 'fontsize', 20)
saveas(h, [outDir '/single_im_discrim_filters_gt_opt_ns.png'])
print(h, '-depsc', [outDir '/single_im_discrim_filters_gt_opt_ns.eps'])
%
h = figure;
hold on
plot(1:NClasses, 1:NClasses, 'k')
plot(1:NClasses, Train_unnorm.MeanEstLabel, 'r')
plot(1:NClasses, Train_unnorm_noise.MeanEstLabel, 'r--')
plot(1:NClasses, Test_unnorm.MeanEstLabel, 'b')
plot(1:NClasses, Test_unnorm_noise.MeanEstLabel, 'b--')

plot(1:NClasses, Train_unnorm_NS.MeanEstLabel, 'm')
plot(1:NClasses, Train_unnorm_noise_NS.MeanEstLabel, 'm--')
plot(1:NClasses, Test_unnorm_NS.MeanEstLabel, 'g')
plot(1:NClasses, Test_unnorm_noise_NS.MeanEstLabel, 'g--')

plot(1:NClasses, Train_unnorm_RS.MeanEstLabel, 'c')
plot(1:NClasses, Train_unnorm_noise_RS.MeanEstLabel, 'c--')
plot(1:NClasses, Test_unnorm_RS.MeanEstLabel, 'y')
plot(1:NClasses, Test_unnorm_noise_RS.MeanEstLabel, 'y--')
axis equal
legend('GT', 'Train', 'Train+noise', 'Test', 'Test+noise', ...
    'Train (NS)', 'Train+noise (NS)', 'Test (NS)', 'Test+noise (NS)', ...
    'Train (RS)', 'Train+noise (RS)', 'Test (RS)', 'Test+noise (RS)', 'location', 'SouthEast');
%title('Single Image Blur Estimation', 'fontsize', 20)
saveas(h, [outDir '/single_im_discrim_filters_gt.png'])
if(bEvalLargeFilterSet)
   % plot with NS and RS replaced by large filters 
    h = figure;
    hold on
    plot(1:NClasses, 1:NClasses, 'k')
    plot(1:NClasses, Train_unnorm.MeanEstLabel, 'r')
    plot(1:NClasses, Train_unnorm_noise.MeanEstLabel, 'r--')
    plot(1:NClasses, Test_unnorm.MeanEstLabel, 'b')
    plot(1:NClasses, Test_unnorm_noise.MeanEstLabel, 'b--')

    plot(1:NClasses, Train_NS_L.MeanEstLabel, 'm')
    plot(1:NClasses, Train_noise_NS_L.MeanEstLabel, 'm--')
    plot(1:NClasses, Test_NS_L.MeanEstLabel, 'g')
    plot(1:NClasses, Test_noise_NS_L.MeanEstLabel, 'g--')

    plot(1:NClasses, Train_RS_L.MeanEstLabel, 'c')
    plot(1:NClasses, Train_noise_RS_L.MeanEstLabel, 'c--')
    plot(1:NClasses, Test_RS_L.MeanEstLabel, 'y')
    plot(1:NClasses, Test_noise_RS_L.MeanEstLabel, 'y--')
    axis equal
    legend('GT', 'Train', 'Train+noise', 'Test', 'Test+noise', ...
        'Train (NS)', 'Train+noise (NS)', 'Test (NS)', 'Test+noise (NS)', ...
        'Train (RS)', 'Train+noise (RS)', 'Test (RS)', 'Test+noise (RS)', 'location', 'SouthEast');
    title(['Single Image Blur Estimation (' num2str(LargeFilterNums) ' filters/depth)'], 'fontsize', 20)
    saveas(h, [outDir '/single_im_discrim_filters_large_gt.png'])

end
%
h = figure;
hold on
plot(1:NClasses, 1:NClasses, 'k')
errorbar(1:NClasses, Train_unnorm.MeanEstLabel, Train_unnorm.StdEstLabel, 'r')
errorbar(1:NClasses, Train_unnorm_noise.MeanEstLabel, Train_unnorm_noise.StdEstLabel,'r--')
errorbar(1:NClasses, Test_unnorm.MeanEstLabel, Test_unnorm.StdEstLabel, 'b')
errorbar(1:NClasses, Test_unnorm_noise.MeanEstLabel, Test_unnorm_noise.StdEstLabel, 'b--')

errorbar(1:NClasses, Train_unnorm_NS.MeanEstLabel, Train_unnorm_NS.StdEstLabel, 'm')
errorbar(1:NClasses, Train_unnorm_noise_NS.MeanEstLabel, Train_unnorm_noise_NS.StdEstLabel,'m--')
errorbar(1:NClasses, Test_unnorm_NS.MeanEstLabel, Test_unnorm_NS.StdEstLabel,'g')
errorbar(1:NClasses, Test_unnorm_noise_NS.MeanEstLabel, Test_unnorm_noise_NS.StdEstLabel, 'g--')

errorbar(1:NClasses, Train_unnorm_RS.MeanEstLabel, Train_unnorm_RS.StdEstLabel, 'c')
errorbar(1:NClasses, Train_unnorm_noise_RS.MeanEstLabel, Train_unnorm_noise_RS.StdEstLabel, 'c--')
errorbar(1:NClasses, Test_unnorm_RS.MeanEstLabel, Test_unnorm_RS.StdEstLabel, 'y')
errorbar(1:NClasses, Test_unnorm_noise_RS.MeanEstLabel, Test_unnorm_noise_RS.StdEstLabel, 'y--')
axis equal
legend('GT', 'Train', 'Train+noise', 'Test', 'Test+noise', ...
    'Train (NS)', 'Train+noise (NS)', 'Test (NS)', 'Test+noise (NS)', ...
    'Train (RS)', 'Train+noise (RS)', 'Test (RS)', 'Test+noise (RS)', 'location', 'SouthEast');
%title('Single Image Blur Estimation', 'fontsize', 20)
saveas(h, [outDir '/single_im_discrim_filters_gt_errbar.png'])
%
h = figure;
hold on
plot(1:NClasses, 1:NClasses, 'k')
errorbar(1:NClasses, Train_unnorm.MeanEstLabel, Train_unnorm.StdEstLabel, 'r')
errorbar(1:NClasses, Train_unnorm_noise.MeanEstLabel, Train_unnorm_noise.StdEstLabel,'r--')
errorbar(1:NClasses, Test_unnorm.MeanEstLabel, Test_unnorm.StdEstLabel, 'b')
errorbar(1:NClasses, Test_unnorm_noise.MeanEstLabel, Test_unnorm_noise.StdEstLabel, 'b--')

errorbar(1:NClasses, Train_unnorm_NS.MeanEstLabel, Train_unnorm_NS.StdEstLabel, 'm')
errorbar(1:NClasses, Train_unnorm_noise_NS.MeanEstLabel, Train_unnorm_noise_NS.StdEstLabel,'m--')
errorbar(1:NClasses, Test_unnorm_NS.MeanEstLabel, Test_unnorm_NS.StdEstLabel,'g')
errorbar(1:NClasses, Test_unnorm_noise_NS.MeanEstLabel, Test_unnorm_noise_NS.StdEstLabel, 'g--')

axis equal
legend('GT', 'Train', 'Train+noise', 'Test', 'Test+noise', ...
    'Train (NS)', 'Train+noise (NS)', 'Test (NS)', 'Test+noise (NS)', ...
     'location', 'SouthEast');
%title('Single Image Blur Estimation', 'fontsize', 20)
saveas(h, [outDir '/single_im_discrim_filters_gt_errbar_opt_ns.png'])

h = figure;
hold on
plot(1:NClasses, 1:NClasses, 'k')
errorbar(1:NClasses, Train_unnorm.MeanEstLabel, Train_unnorm.StdEstLabel, 'r')
errorbar(1:NClasses, Train_unnorm_noise.MeanEstLabel, Train_unnorm_noise.StdEstLabel,'r--')
errorbar(1:NClasses, Test_unnorm.MeanEstLabel, Test_unnorm.StdEstLabel, 'b')
errorbar(1:NClasses, Test_unnorm_noise.MeanEstLabel, Test_unnorm_noise.StdEstLabel, 'b--')

axis equal
legend('GT', 'Train', 'Train+noise', 'Test', 'Test+noise', ...
         'location', 'Best');
%title('Single Image Blur Estimation', 'fontsize', 20)
saveas(h, [outDir '/single_im_discrim_filters_gt_errbar_opt.png'])
print(h, '-depsc', [outDir '/single_im_discrim_filters_gt_errbar_opt.eps'])

h = figure;
hold on
plot(1:NClasses, 1:NClasses, 'k')
errorbar(1:NClasses, Train_unnorm_NS.MeanEstLabel, Train_unnorm_NS.StdEstLabel, 'm')
errorbar(1:NClasses, Train_unnorm_noise_NS.MeanEstLabel, Train_unnorm_noise_NS.StdEstLabel,'m--')
errorbar(1:NClasses, Test_unnorm_NS.MeanEstLabel, Test_unnorm_NS.StdEstLabel,'g')
errorbar(1:NClasses, Test_unnorm_noise_NS.MeanEstLabel, Test_unnorm_noise_NS.StdEstLabel, 'g--')

axis equal
legend('GT', 'Train (NS)', 'Train+noise (NS)', 'Test (NS)', 'Test+noise (NS)', ...
     'location', 'SouthEast');
%title('Single Image Blur Estimation', 'fontsize', 20)
saveas(h, [outDir '/single_im_discrim_filters_gt_errbar_ns.png'])
print(h, '-depsc', [outDir '/single_im_discrim_filters_gt_errbar_ns.eps'])
%
%
h = figure;
hold on
plot(1:NClasses, Train_unnorm.RMSE, 'r')
plot(1:NClasses, Train_unnorm_noise.RMSE, 'r--')
plot(1:NClasses, Test_unnorm.RMSE, 'b')
plot(1:NClasses, Test_unnorm_noise.RMSE, 'b--')

plot(1:NClasses, Train_unnorm_NS.RMSE, 'm')
plot(1:NClasses, Train_unnorm_noise_NS.RMSE, 'm--')
plot(1:NClasses, Test_unnorm_NS.RMSE, 'g')
plot(1:NClasses, Test_unnorm_noise_NS.RMSE, 'g--')

plot(1:NClasses, Train_unnorm_RS.RMSE, 'c')
plot(1:NClasses, Train_unnorm_noise_RS.RMSE, 'c--')
plot(1:NClasses, Test_unnorm_RS.RMSE, 'y')
plot(1:NClasses, Test_unnorm_noise_RS.RMSE, 'y--')

legend('Train', 'Train+noise', 'Test', 'Test+noise', ...
    'Train (NS)', 'Train+noise (NS)', 'Test (NS)', 'Test+noise (NS)', ...
    'Train (RS)', 'Train+noise (RS)', 'Test (RS)', 'Test+noise (RS)');
xlabel('Label', 'Fontsize', 22);
ylabel('RMSE', 'Fontsize', 22);
%title('Single Image Blur Estimation', 'fontsize', 20)
saveas(h, [outDir '/single_im_discrim_filters_rmse.png'])
print(h, '-depsc', [outDir '/single_im_discrim_filters_rmse.eps'])
%%
h = figure;
hold on
xrange = 0:(NClasses - 1);
plot(xrange, Train_unnorm.CError, 'r')
plot(xrange, Train_unnorm_noise.CError, 'r--')
plot(xrange, Test_unnorm.CError, 'b')
plot(xrange, Test_unnorm_noise.CError, 'b--')

plot(xrange, Train_unnorm_NS.CError, 'm')
plot(xrange, Train_unnorm_noise_NS.CError, 'm--')
plot(xrange, Test_unnorm_NS.CError, 'g')
plot(xrange, Test_unnorm_noise_NS.CError, 'g--')

plot(xrange, Train_unnorm_RS.CError, 'c')
plot(xrange, Train_unnorm_noise_RS.CError, 'c--')
plot(xrange, Test_unnorm_RS.CError, 'y')
plot(xrange, Test_unnorm_noise_RS.CError, 'y--')

legend('Train', 'Train+noise', 'Test', 'Test+noise', ...
    'Train (NS)', 'Train+noise (NS)', 'Test (NS)', 'Test+noise (NS)', ...
    'Train (RS)', 'Train+noise (RS)', 'Test (RS)', 'Test+noise (RS)');
xlabel('Label Diff Threshold', 'Fontsize', 22);
ylabel('Error', 'Fontsize', 22);
title('Error with threshold', 'fontsize', 20)
saveas(h, [outDir '/cumulative_error.png'])
print( h, '-depsc', [outDir '/cumulative_error.eps'])
%%
h = figure;
hold on
xrange = 0:(NClasses - 1);
plot(xrange, Train_unnorm.CError, 'r')
plot(xrange, Train_unnorm_noise.CError, 'r--')
plot(xrange, Test_unnorm.CError, 'b')
plot(xrange, Test_unnorm_noise.CError, 'b--')

plot(xrange, Train_unnorm_NS.CError, 'm')
plot(xrange, Train_unnorm_noise_NS.CError, 'm--')
plot(xrange, Test_unnorm_NS.CError, 'g')
plot(xrange, Test_unnorm_noise_NS.CError, 'g--')

legend('Train', 'Train+noise', 'Test', 'Test+noise', ...
    'Train (NS)', 'Train+noise (NS)', 'Test (NS)', 'Test+noise (NS)');
xlabel('Label Diff Threshold', 'Fontsize', 22);
ylabel('Error', 'Fontsize', 22);
title('Error with threshold', 'fontsize', 20)
saveas(h, [outDir '/cumulative_error_opt_ns.png'])
print( h, '-depsc', [outDir '/cumulative_error_opt_ns.eps'])

%%
xrange = 0:(NClasses - 1);
h = figure;
semilogy(xrange, Train_unnorm.CError, 'r')
hold on
semilogy(xrange, Train_unnorm_noise.CError, 'r--')
semilogy(xrange, Test_unnorm.CError, 'b')
semilogy(xrange, Test_unnorm_noise.CError, 'b--')

semilogy(xrange, Train_unnorm_NS.CError, 'm')
semilogy(xrange, Train_unnorm_noise_NS.CError, 'm--')
semilogy(xrange, Test_unnorm_NS.CError, 'g')
semilogy(xrange, Test_unnorm_noise_NS.CError, 'g--')

legend('Train', 'Train+noise', 'Test', 'Test+noise', ...
    'Train (NS)', 'Train+noise (NS)', 'Test (NS)', 'Test+noise (NS)', 'Location', 'SouthWest');
xlabel('Label Diff Threshold', 'Fontsize', 22);
ylabel('Error', 'Fontsize', 22);
title('Error with threshold', 'fontsize', 20)
saveas(h, [outDir '/semilogy_cumulative_error_opt_ns.png'])
print( h, '-depsc', [outDir '/semilogy_cumulative_error_opt_ns.eps'])

%%
h = figure;
semilogy(xrange, Train_unnorm.CError, 'r')
hold on
semilogy(xrange, Train_unnorm_noise.CError, 'r--')
semilogy(xrange, Test_unnorm.CError, 'b')
semilogy(xrange, Test_unnorm_noise.CError, 'b--')

semilogy(xrange, Train_unnorm_init.CError, 'm')
semilogy(xrange, Train_unnorm_noise_init.CError, 'm--')
semilogy(xrange, Test_unnorm_init.CError, 'c')
semilogy(xrange, Test_unnorm_noise_init.CError, 'c--')

legend('Train', 'Train+noise', 'Test', 'Test+noise', ...
    'Train (init)', 'Train+noise (init)', 'Test (init)', 'Test+noise (init)', 'Location', 'SouthWest');
xlabel('Label Diff Threshold', 'Fontsize', 22);
ylabel('Error', 'Fontsize', 22);
title('Error with threshold', 'fontsize', 20)
saveas(h, [outDir '/semilogy_cumulative_error_opt_optinit.png'])
print( h, '-depsc', [outDir '/semilogy_cumulative_error_opt_optinit.eps'])
