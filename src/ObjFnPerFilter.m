function [loss, dW] = ObjFnPerFilter(X, Wvec, y, A, params, ...
                                        NFilters, DataDim, NClasses)
% implement vectorized objective function with hinge loss and *min* of quadratic
% cost of filters
% This objective function ensures that each filter has non-zero response
% (and preferrably > 1) wrong labels.
% NOTE (TODO): Use fnEnergy
% NFiters : F
% DataDim : D
% NClasses : C
% l1 : energy L1
% l2 : |W^2 - 1|
% l3 : sum of loss per filter for non-class example
% l4 : max loss per example
% l5 : loss exp decay for wrong labels
% l6 : loss for wrong labels being close to 0. sum_{j \neq y_i} 1/(1 + (a E_j)^2)

% Energy function Objective:
% sum_i QuadCost(W_{y_i} x_i) + sum_i sum_{j \neq y_i} [l_2 *  max{0, s E^1_{y_i} -
% E^1_{j} + m + s} + l_3 * exp(-a E^1_j) ] + |x^Tx - N|
% Init solution can be chosen to be the null space of
% WXX' where X = [x_i,..., -x_j] where i \in current_label and j \nin
% current_label

m = params.M0;
corr_scale = params.CorrScale;
lambdas = params.LAMBDAS;
% 
% if(~exist('bNoReshape', 'var'))
%     bNoReshape = false;
% end
% if(bNoReshape)
%     % Wvec is already in (NFilters x Class) x D format
%     W = Wvec;
% else
%     % W is F x D x C and X is D x N, y is 1 x N with the correct labels
%     % make W 
%     Wmat = reshape(Wvec, NFilters, DataDim, NClasses);
%     W = reshape(permute(Wmat, [2, 1, 3]), [DataDim, NFilters * NClasses])'; % W is now (F x C) x D
% end
W = reshape(Wvec, NFilters * NClasses, DataDim);

ZeroLossMult = params.ZeroLossMult;
ExpZeroLossMult = params.ExpZeroLossMult;

NTrain = size(X, 2);

%%%% TODO: IMPLEMENT
% consider backprop type (reverse deriv) implementation
% obj = sum_k lambda_k * loss_k(x)
% [c1, g1] = fnLoss1(x, fnEnergy2)
% [c2, g2] = fnLoss2(x, fnEnergy1)
% cost = c1 + c2; grad = g1 + g2;
% ....


% A is C x (F x C)
[Energy, ~, ~, Eres] = L1CostVec(X, W, A); %0.5 * A * (W * X).^2; % C x N  % NOTE: Scores = -Energy
[ind, ExLabelMat] = getLabelIdxMat(size(Energy), y);
[HLossPerFilter, dLdE] = HingeLossPerLabel(Energy, m, corr_scale, ExLabelMat);

corr_energy = Energy(ind);
% margin = bsxfun(@minus, corr_scale * corr_energy + m, Energy);
% margin(ind) = 0;
% loss_per_ex = max(0, margin);
%[max_loss_per_ex, I_max_loss] = max(margin); % max hinge loss
[MHLoss, dMHLdE] = MaxHingeLoss(Energy, m, corr_scale, ind);

ZeroLoss = 1./(1 + (ZeroLossMult * abs(Energy))); % penalty for wrong energies having close to zero energy
ZeroLoss(ind) = 0; % correct labels should've 0 zero-loss penalty

exp_decay_loss = exp(- ExpZeroLossMult * abs(Energy)); % energy is L1 so abs(Energy) = Energy
exp_decay_loss(ind) = 0;


%%% compute non-unit norm penalty
finite_norm = diag(W*W');
finite_norm_deviation = finite_norm - params.FiniteNormConst^2;

loss = (lambdas(1) * sum(corr_energy) ...
        + lambdas(3) * HLossPerFilter ...
        + lambdas(4) * MHLoss ...
        + lambdas(5) * sum(exp_decay_loss(:)) ...
        + lambdas(6) * sum(ZeroLoss(:)) ) / NTrain + lambdas(2) * sum(abs(finite_norm_deviation));


%%%% compute loss derivatives

%%% sum of hinge losses
% dLdS = zeros(size(loss_per_ex)); % C x N
% dLdS(margin > 0) = -1;
% NViolations = sum(-dLdS); % number of constraint violations for each example
% dLdS(ind) = corr_scale * NViolations;

% %%%% Derivative for Max Hinge Loss
% dMHLdS = zeros(NClasses, NTrain);
% IndMHL = sub2ind([NClasses, NTrain], I_max_loss, 1:NTrain);
% dMHLdS(IndMHL) =  -1;
% NViolationsMHL = sum(-dMHLdS);
% %assert(all(max(NViolationsMHL) <= 1));
% dMHLdS(ind) = corr_scale * NViolationsMHL;

%%% derivative for exp decay loss
dExpDecayLoss = -ExpZeroLossMult * sign(Energy) .* exp_decay_loss; % here sign(Energy) because of abs(Energy)
dExpDecayLoss(ind) = 0; % deriv for corr energies is 0

%%% derivative for ZeroLoss only for wrong labels
dZL = - ZeroLossMult * ZeroLoss.^2 .* sign(Energy); % d/dE 1/(1 + a | E | )
dZL(ind) = 0; % only penalize wrong labels. Here ind corresponds to entries corresponding to the right labels

% CorrClassSelector is CxN with 1 at element (i,j) if class label of jth training
% sample is i. This is used for penalizing correct labels for non-zero
% energy
CorrClassSelector = zeros(size(Energy));
CorrClassSelector(ind) = 1;

% C x N loss derivatives
SumOfLosses = lambdas(1) * CorrClassSelector + lambdas(3) * dLdE + ...
              lambdas(4) * dMHLdE + lambdas(5) * dExpDecayLoss + lambdas(6) * dZL;
          
dW = Eres.fnGrad(SumOfLosses);
% Multiplier = SumOfLosses' * A;
% % for derivative computation loop over all example for simplicity
% dW = zeros(size(W));
% for trainIdx = 1:NTrain
% %     Multiplier = lambdas(1) * CorrClassSelector(:,trainIdx) ...
% %                  + lambdas(3) * dLdS(:,trainIdx) ...
% %                  + lambdas(4) * dMHLdS(:,trainIdx) ...
% %                  + lambdas(5) * dExpDecayLoss(:,trainIdx) ...
% %                  + lambdas(6) * dZL(:,trainIdx);
%     L1EnergyDeriv = sign(W * X(:,trainIdx)) * X(:,trainIdx)';
%     dW = dW + diag(Multiplier(trainIdx,:)) * L1EnergyDeriv;
% end

% non-unit penalty gradient
dU = diag(sign(finite_norm_deviation)) * 2 * W;

%
dW = dW / NTrain + lambdas(2) * dU;

% %%%%%%%%%%%%%%%%%%
% if(bNoReshape)
%     grad = dW;
% else
%     % reshape to original form
%     grad = permute(reshape(dW', [DataDim, NFilters, NClasses]), [2, 1, 3]);
% end
