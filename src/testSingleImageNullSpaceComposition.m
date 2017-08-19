% test the idea of finding the joint NS (FS approach) by composing
% individual null space
% [i1; i2] = [A1; A2] i0 
% [NS1, NS2] [i1; i2] = [NS1, NS2] [A1; A2] i0
% so Hperp_d = [NS1, NS2]
% the advantage is that for n depths and m configurations joint training
% requires O(nm^2) but if they can be separated then O(nm) steps are
% needed.

close all
clear
clc

addpath ../
addpath ../DFD
addpath ../../data/textures


bOnlyNorm = false; %true;
fnTform = @(x) x; % del2(x)
ImFilename = {'brown_noise_pattern_512x512.png', 'D42_512x512.png',  'D94_512x512.png', 'merry_mtl07_053.tif'}; %'white_noise_2_512x512.png', 
Im_seq = cell(1, length(ImFilename));
for idx = 1:length(ImFilename)
    Im_seq{idx} = fnTform(mean(im2double(imread(ImFilename{idx})), 3));
end

%% Test seq
ImTestFilename = {'D11.gif' , 'merry_mtl07_023.tif', 'pippin_Peel071.tif'};
ImTest_seq = cell(1, length(ImTestFilename));
for idx = 1:length(ImTestFilename)
    ImTest_seq{idx} = fnTform(mean(im2double(imread(ImTestFilename{idx})), 3));
end
%% Train
NDepth = 21;
NImgs = 2;
KernelSet = cell(NDepth, NImgs);
Z0 = 5; %0; %
Z1 = 15; %6; %
RelBlur = 3; %1; %3; %4; %1.732; %3; %1
radii = linspace(Z0, Z1, NDepth);
for idx = 1:NDepth
   KernelSet{idx, 1} = PillboxKernel(radii(idx), 4, 0); %padarray(fspecial('disk', radii(idx)), [4, 4], 0); 
   if(NImgs == 2)
    KernelSet{idx, 2} = PillboxKernel(sqrt(RelBlur^2 + radii(idx)^2), 4, 0); %padarray(fspecial('disk', sqrt(9 + radii(idx)^2)), [4, 4], 0); 
   end
end

%%
params.TruncatePercentage = 1; %.25
params.NMaxTraining = 300;
params.KR = 13; %
params.KC = 13;
params.NFilters = 5;
params.fnPatchNormalization = @(x) x;
params.fnTform = []; %@(x) 10 * x.^2;
params.fnCombine = @(x, y) [x; y]; %[x+y; x - y]; %[x + y]; %[(x + y) ; (x - y)] %10 * (x.^2 + y.^2) %x.*y; %(x + y).^2 - (x - y).^2 %(x.^2 + y.^2); %@minus; %@plus;

if(~bOnlyNorm)
    resultPair = TrainNullSpaceFilters(Im_seq, KernelSet, params)
end
%
paramsnorm = params;
paramsnorm.fnPatchNormalization = @(x) (x); %[]; % @(x) ( (x - min(x(:))) / (max(x(:)) - min(x(:)))); %@(x) (x / mean(x(:)) - 1); %
paramsnorm.fnCombine = @(x, y) (bsxfun(@rdivide, bsxfun(@minus, [x ; y], mean([x;y])), std([x;y]))); %
resultPairNorm = TrainNullSpaceFilters(Im_seq, KernelSet, paramsnorm)
%%
NoiseStd = 0.01; %.01;

Im_seq_noise1 = Im_seq;
%Im_seq_noise2 = Im_seq;
for it = 1:size(Im_seq_noise1, 2)
   Im_seq_noise1{it} = Im_seq_noise1{it} + NoiseStd * randn(size(Im_seq_noise1{it}));
   %Im_seq_noise2{it} = Im_seq_noise2{it} + NoiseStd * randn(size(Im_seq_noise2{it}));
end
if(~bOnlyNorm)
    TrainSetNoise = BuildTrainingSet(Im_seq_noise1, KernelSet, params);
end
TrainSetNoiseNorm = BuildTrainingSet(Im_seq_noise1, KernelSet, paramsnorm);
%%
ImTest_seq_noise = ImTest_seq;
for it = 1:size(ImTest_seq_noise, 2)
   ImTest_seq_noise{it} = ImTest_seq{it} + NoiseStd * randn(size(ImTest_seq{it}));
end
if(~bOnlyNorm)
    TestSetNoise = BuildTrainingSet(ImTest_seq_noise, KernelSet, params);
end
TestSetNoiseNorm = BuildTrainingSet(ImTest_seq_noise, KernelSet, paramsnorm);
%%
%training_set_im1 = BuildTrainingSet(Im_seq_noise1, KernelSet(:,1), params);

if(~bOnlyNorm)
TrainSetUnnorm = BuildTrainingSet(Im_seq, KernelSet, params);
TestSetUnnorm = BuildTrainingSet(ImTest_seq, KernelSet, params);
end
TrainSetNorm = BuildTrainingSet(Im_seq, KernelSet, paramsnorm);
TestSetNorm = BuildTrainingSet(ImTest_seq, KernelSet, paramsnorm);
%% %%
% resultIm1 = TrainNullSpaceFilters(Im_seq, KernelSet(:,1), params);
% 
% resultIm2 = TrainNullSpaceFilters(Im_seq, KernelSet(:,2), params);
% 
% %%
% filterevalres = EvalFilterCost(training_set_1.ImageDepthSet, resultPair, @L1CostVec); 
% %%
% filterevalres_norm = EvalFilterCost(training_set_1_norm.ImageDepthSet, resultPair, @L1CostVec); 
% 
% %%
% filterevalres_norm_norm = EvalFilterCost(training_set_1_norm.ImageDepthSet, resultPairNorm, @L1CostVec); 
% %%
% filterevalres_im1 = EvalFilterCost(training_set_im1.ImageDepthSet, resultIm1, @L1CostVec); 
% %%
% paramsnorm = params;
% paramsnorm.fnPatchNormalization =  []; %@(x) x;
% resultim1Norm = TrainNullSpaceFilters(Im_seq, KernelSet(:,1), paramsnorm)
% training_set_im1_norm = BuildTrainingSet(Im_seq_noise1, KernelSet(:,1), paramsnorm);
% fresim1_norm = EvalFilterCost(training_set_im1_norm.ImageDepthSet, resultim1Norm, @L1CostVec);
% %%
% %paramsnorm.TruncatePercentage = 1; %.25 %1;
% resultim2Norm = TrainNullSpaceFilters(Im_seq, KernelSet(:,2), paramsnorm)
% %
% training_set_im2_norm = BuildTrainingSet(Im_seq_noise1, KernelSet(:,2), paramsnorm);
% %%
% %training_set_im2_norm = BuildTrainingSet(ImTest_seq, KernelSet(:,2), paramsnorm);
% fresim2_norm = EvalFilterCost(training_set_im2_norm.ImageDepthSet, resultim2Norm, @L1CostVec); 

%%
fnCost = @L1CostVec; %@QuadraticCostVec;
if(~bOnlyNorm)
Train_unnorm = EvalFilterCost(TrainSetUnnorm.ImageDepthSet, resultPair, fnCost)    
Test_unnorm = EvalFilterCost(TestSetUnnorm.ImageDepthSet, resultPair, fnCost)
Train_unnorm_noise = EvalFilterCost(TrainSetNoise.ImageDepthSet, resultPair, fnCost)
Test_unnorm_noise = EvalFilterCost(TestSetNoise.ImageDepthSet, resultPair, fnCost)

%Train_unnorm_norm = EvalFilterCost(TrainSetUnnorm.ImageDepthSet, resultPairNorm, fnCost) 
%Test_unnorm_norm = EvalFilterCost(TestSetUnnorm.ImageDepthSet, resultPairNorm, fnCost) 
end

Train_norm = EvalFilterCost(TrainSetNorm.ImageDepthSet, resultPairNorm, fnCost) 

%%
Test_norm = EvalFilterCost(TestSetNorm.ImageDepthSet, resultPairNorm, fnCost) 


%

Train_norm_noise = EvalFilterCost(TrainSetNoiseNorm.ImageDepthSet, resultPairNorm, fnCost)


Test_norm_noise = EvalFilterCost(TestSetNoiseNorm.ImageDepthSet, resultPairNorm, fnCost)

%
if(~bOnlyNorm)
Train_unnorm_err = Train_unnorm.TotalError / size(Train_unnorm.Error, 1) * 100
%Train_unnorm_norm_err = Train_unnorm_norm.TotalError / size(Train_unnorm_norm.Error, 1) * 100
end
Train_norm_err = Train_norm.TotalError / size(Train_norm.Error, 1) * 100


if(~bOnlyNorm)
Test_unnorm_err = Test_unnorm.TotalError / size(Test_unnorm.Error, 1) * 100
%Test_unnorm_norm_err = Test_unnorm_norm.TotalError / size(Test_unnorm_norm.Error, 1) * 100
end
Test_norm_err = Test_norm.TotalError / size(Test_norm.Error, 1) * 100

%
if(~bOnlyNorm)
Train_unnorm_noise_err = Train_unnorm_noise.TotalError / size(Train_unnorm_noise.Error, 1) * 100
end
Train_norm_noise_err = Train_norm_noise.TotalError / size(Train_norm_noise.Error, 1) * 100

if(~bOnlyNorm)
Test_unnorm_noise_err = Test_unnorm_noise.TotalError / size(Test_unnorm_noise.Error, 1) * 100
end
Test_norm_noise_err = Test_norm_noise.TotalError / size(Test_norm_noise.Error, 1) * 100