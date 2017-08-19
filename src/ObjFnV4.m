function [loss, deriv] = ObjFnV4(X, ParamsVec, y, A0, params, ...
                                        NFilters, DataDim, NClasses)
% considers A as parameters and optimizes it with W.
% The desired cost function is passed as a parameter in params.fnCost
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

WSize = NFilters * NClasses * DataDim;
W = reshape(ParamsVec(1:WSize), NFilters * NClasses, DataDim);
A = reshape(ParamsVec(WSize+1:end), NClasses, []);
if(A0 ~= 1)
    assert(size(A, 2) == size(A0, 1));
end
% ZeroLossMult = params.ZeroLossMult;
% ExpZeroLossMult = params.ExpZeroLossMult;
% ExpDecayAlpha = params.ExpDecayAlpha;

assert(lambdas(1) == 0 && lambdas(3) == 0 && all(lambdas(5:9) == 0))

NTrain = size(X, 2);

% W is (F x C1) x D
% A0 is C1 x (F x C1)
% A is C x C1
[Energy, ~, ~, Eres] = params.fnCost(X, W, A, A0); %NOTE: Scores = -Energy
[ind, ExLabelMat] = getLabelIdxMat(size(Energy), y);

%%% Loss
%[AQNPenalty, dAQNdW] = AbsL2SqrNormPenalty(W, params.FiniteNormConst);
[FNPenalty, dFNdW] = FrobSqrNormPenalty(W);

[LEnergyLoss, dLEdE] = LowEnergyLoss(Energy, ind);
%[HLossPerFilter, dLdE] = HingeLossPerLabel(Energy, m, corr_scale, ExLabelMat);
[HLoss, dHLdE, HLMargin] = HingeLoss(Energy, m, corr_scale, ind, PerLabelLoss);
%[MHLoss, dMHLdE] = MaxHingeLoss(Energy, m, corr_scale, ind, PerLabelLoss, HLMargin);
%[EAPenalty, dExpdE] = ExpAbsPenalty(Energy, ExpZeroLossMult, 0, ind);
%[LAbsPenalty, dLAdE] = LorentzianAbsPenalty(Energy, ZeroLossMult, 0, ind);
%[EPenalty, dE2dE] = ExpPenalty(Energy, ExpDecayAlpha, params.ExpPenaltyCenter, ind, false);
%[CorrEPenalty, dCorrE2dE] = ExpPenalty(Energy, -ExpDecayAlpha, params.CorrExpPenaltyCenter, ind, true);


loss = lambdas(10) * FNPenalty  ...
       + (lambdas(2) * LEnergyLoss + lambdas(4) * HLoss) / NTrain;

%%%% compute loss derivatives
% C x N loss derivatives
SumOfLosses = lambdas(2) * dLEdE +  lambdas(4) * dHLdE;

% if(abs(lambdas(8)) > 1e-16)
%        loss = loss + lambdas(8) * EPenalty / NTrain;
%        SumOfLosses = SumOfLosses + lambdas(8) * dE2dE;
% end
% if(abs(lambdas(9)) > 1e-6)
%     loss = loss + lambdas(9) * CorrEPenalty / NTrain;
%     SumOfLosses = SumOfLosses + lambdas(9) * dCorrE2dE;
% end          
[dW, dA] = Eres.fnGrad(SumOfLosses);
dW = dW / NTrain + lambdas(10) * dFNdW;
dA = dA / NTrain;
deriv = [dW(:); dA(:)];