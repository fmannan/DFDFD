function [loss, dW] = ObjFnGeneral(X, Wvec, y, A, params, ...
                                        NFilters, DataDim, NClasses)
% The desired cost function is passes as a parameter in params.fnCost
% This objective function ensures that each filter has non-zero response
% (and preferrably > 1) wrong labels.
% NOTE (TODO): Use fnEnergy
% NFiters : F
% DataDim : D
% NClasses : C
% l1 : |W^2 - 1|
% l2 : energy L1
% l3 : sum of loss per filter for non-class example
% l4 : sum of loss per example
% l5 : max loss per example
% l6 : loss exp decay for wrong labels
% l7 : loss for wrong labels being close to 0. sum_{j \neq y_i} 1/(1 + (a E_j)^2)
% l8 : exp penalize incorr labels for being close to limit point
% l9 : exp penalize corr labels for being close to pos limit
% l10: frobenius norm sqr

% Energy function Objective:
% sum_i QuadCost(W_{y_i} x_i) + sum_i sum_{j \neq y_i} [l_2 *  max{0, s E^1_{y_i} -
% E^1_{j} + m + s} + l_3 * exp(-a E^1_j) ] + |x^Tx - N|
% Init solution can be chosen to be the null space of
% WXX' where X = [x_i,..., -x_j] where i \in current_label and j \nin
% current_label

% backprop type (reverse deriv) implementation
% obj = sum_k lambda_k * loss_k(x)
% [c1, g1] = fnLoss1(x, fnEnergy2)
% [c2, g2] = fnLoss2(x, fnEnergy1)
% cost = c1 + c2; grad = g1 + g2;

m = params.M0;
corr_scale = params.CorrScale;
lambdas = params.LAMBDAS;
PerLabelLoss = params.PerLabelLoss;
% should be called with the correct length to avoid extra computation (this code gets called several thousand times from the optimizer)
if(length(lambdas) < 10) % quick fix for old length
    tmp = zeros(1, 10);
    tmp(1:length(lambdas)) = lambdas;
    lambdas = tmp;
end

W = reshape(Wvec, NFilters * NClasses, DataDim);

ZeroLossMult = params.ZeroLossMult;
ExpZeroLossMult = params.ExpZeroLossMult;
ExpDecayAlpha = params.ExpDecayAlpha;

NTrain = size(X, 2);

% A is C x (F x C)
[Energy, ~, ~, Eres] = params.fnCost(X, W, A); %NOTE: Scores = -Energy
[ind, ExLabelMat] = getLabelIdxMat(size(Energy), y);

%%% Loss
loss = 0;
SumOfLosses = 0;
dW = 0;
if(lambdas(1) > 0)
    [AQNPenalty, dAQNdW] = AbsL2SqrNormPenalty(W, params.FiniteNormConst);
    loss = loss + lambdas(1) * AQNPenalty;
    dW = dW + lambdas(1) * dAQNdW;
end
if(lambdas(10) > 0)
    [FNPenalty, dFNdW] = FrobSqrNormPenalty(W);
    loss = loss + lambdas(10) * FNPenalty;
    dW = dW + lambdas(10) * dFNdW;
end
lossScaled = 0;
if(lambdas(2) > 0)
    [LEnergyLoss, dLEdE] = LowEnergyLoss(Energy, ind);
    lossScaled = lossScaled + lambdas(2) * LEnergyLoss;
    SumOfLosses = SumOfLosses + lambdas(2) * dLEdE;
end
if(lambdas(3) > 0)
    [HLossPerFilter, dLdE] = HingeLossPerLabel(Energy, m, corr_scale, ExLabelMat);
    lossScaled = lossScaled + lambdas(3) * HLossPerFilter;
    SumOfLosses = SumOfLosses + lambdas(3) * dLdE;
end

[HLoss, dHLdE, HLMargin] = HingeLoss(Energy, m, corr_scale, ind, PerLabelLoss);
lossScaled = lossScaled + lambdas(4) * HLoss;
SumOfLosses = SumOfLosses + lambdas(4) * dHLdE;

if(lambdas(5) > 0)
    [MHLoss, dMHLdE] = MaxHingeLoss(Energy, m, corr_scale, ind, PerLabelLoss, HLMargin);
    lossScaled = lossScaled + lambdas(5) * MHLoss;
    SumOfLosses = SumOfLosses + lambdas(5) * dMHLdE;
end

if(lambdas(6) > 0)
    [EAPenalty, dExpdE] = ExpAbsPenalty(Energy, ExpZeroLossMult, 0, ind);
    lossScaled = lossScaled + lambdas(6) * EAPenalty;
    SumOfLosses = SumOfLosses + lambdas(6) * dExpdE;
end

if(lambdas(7) > 0)
    [LAbsPenalty, dLAdE] = LorentzianAbsPenalty(Energy, ZeroLossMult, 0, ind);
    lossScaled = lossScaled + lambdas(7) * LAbsPenalty;
    SumOfLosses = SumOfLosses + lambdas(7) * dLAdE;
end

if(abs(lambdas(8)) > 1e-16)
    [EPenalty, dE2dE] = ExpPenalty(Energy, ExpDecayAlpha, params.ExpPenaltyCenter, ind, false);
    lossScaled = lossScaled + lambdas(8) * EPenalty;
    SumOfLosses = SumOfLosses + lambdas(8) * dE2dE;
end

if(abs(lambdas(9)) > 1e-6)
    [CorrEPenalty, dCorrE2dE] = ExpPenalty(Energy, -ExpDecayAlpha, params.CorrExpPenaltyCenter, ind, true);
    lossScaled = lossScaled + lambdas(9) * CorrEPenalty;
    SumOfLosses = SumOfLosses + lambdas(9) * dCorrE2dE;
end          

loss = loss + lossScaled / NTrain;

dW = dW + Eres.fnGrad(SumOfLosses) / NTrain;
