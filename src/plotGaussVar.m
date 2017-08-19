close all
clear
clc

%res = load('model_eval_2F_zNear609.6_zFar1500_f22/ResAll.mat')
res = load('model_eval_2F_zNear700_zFar1219.2_f22/ResAll.mat')
Z = res.Z;
NDepths = length(Z);

SigREstStd = nan(1, NDepths);
SigREstMean = nan(1, NDepths);
for idx = 1:NDepths
    SigREstStd(idx) = res.GaussRelBlurVar{idx}.SigEstStd;
    SigREstMean(idx) = res.GaussRelBlurVar{idx}.SigEstMean;
end

h = figure;
plot(1./Z, SigREstStd, 'r')

h = figure;
plot(1./Z, SigREstMean, 'g')