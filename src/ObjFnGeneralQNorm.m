function [loss, dW] = ObjFnGeneralQNorm(X, Wvec, y, A, params, ...
                                        NFilters, DataDim, NClasses)
% The desired cost function is passes as a parameter in params.fnCost
% This objective function ensures that each filter has non-zero response
% (and preferrably > 1) wrong labels.
% NOTE (TODO): Use fnEnergy
% NFiters : F
% DataDim : D
% NClasses : C
% l1 : (W^2 - 1)^2
% l2 : energy L1
% l3 : sum of loss per filter for non-class example
% l4 : sum of loss per example
% l5 : max loss per example
% l6 : loss exp decay for wrong labels
% l7 : loss for wrong labels being close to 0. sum_{j \neq y_i} 1/(1 + (a E_j)^2)

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

W = reshape(Wvec, NFilters * NClasses, DataDim);

ZeroLossMult = params.ZeroLossMult;
ExpZeroLossMult = params.ExpZeroLossMult;
ExpDecayAlpha = params.ExpDecayAlpha;

NTrain = size(X, 2);

% A is C x (F x C)
[Energy, ~, ~, Eres] = params.fnCost(X, W, A); %NOTE: Scores = -Energy
[ind, ExLabelMat] = getLabelIdxMat(size(Energy), y);

%%% Loss
[QQNPenalty, dQQNdW] = QuadL2SqrNormPenalty(W, params.FiniteNormConst);

[LEnergyLoss, dLEdE] = LowEnergyLoss(Energy, ind);
[HLossPerFilter, dLdE] = HingeLossPerLabel(Energy, m, corr_scale, ExLabelMat);
[HLoss, dHLdE, HLMargin] = HingeLoss(Energy, m, corr_scale, ind);
[MHLoss, dMHLdE] = MaxHingeLoss(Energy, m, corr_scale, ind, HLMargin);
[EAPenalty, dExpdE] = ExpAbsPenalty(Energy, ExpZeroLossMult, 0, ind);
[LAbsPenalty, dLAdE] = LorentzianAbsPenalty(Energy, ZeroLossMult, 0, ind);
[EPenalty, dE2dE] = ExpPenalty(Energy, ExpDecayAlpha, params.ExpPenaltyCenter, ind, false);
[CorrEPenalty, dCorrE2dE] = ExpPenalty(Energy, -ExpDecayAlpha, params.CorrExpPenaltyCenter, ind, true);


loss = lambdas(1) * QQNPenalty  ...
       + (lambdas(2) * LEnergyLoss + lambdas(3) * HLossPerFilter ...
       + lambdas(4) * HLoss + lambdas(5) * MHLoss   ...
       + lambdas(6) * EAPenalty + lambdas(7) * LAbsPenalty) / NTrain;

% if(isnan(loss))
%     display(num2str(loss))
%     AQNPenalty
%     LEnergyLoss
%     HLossPerFilter
%     HLoss
%     EAPenalty
%     LAbsPenalty
%     MHLoss
% end
    
%%%% compute loss derivatives
% C x N loss derivatives
SumOfLosses = lambdas(2) * dLEdE + lambdas(3) * dLdE + ...
              lambdas(4) * dHLdE + lambdas(5) * dMHLdE + ...
              lambdas(6) * dExpdE + lambdas(7) * dLAdE ;
if(abs(lambdas(8)) > 1e-16)
       loss = loss + lambdas(8) * EPenalty / NTrain;
       SumOfLosses = SumOfLosses + lambdas(8) * dE2dE;
end
if(abs(lambdas(9)) > 1e-6)
    loss = loss + lambdas(9) * CorrEPenalty / NTrain;
    SumOfLosses = SumOfLosses + lambdas(9) * dCorrE2dE;
end          
dW = lambdas(1) * dQQNdW + Eres.fnGrad(SumOfLosses) / NTrain;
