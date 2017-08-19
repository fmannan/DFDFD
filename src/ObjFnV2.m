function [loss, dW] = ObjFnV2(X, Wvec, y, A, params, ...
                                NFilters, DataDim, NClasses) %, bNoReshape)
% implement vectorized objective function with hinge loss and *min* of quadratic
% cost of filters
% NOTE (TODO): Any fnEnergy with min selection should work here. Need to
% replace MinQudraticCostVec with fnMinEnergy function
% NFiters : F
% DataDim : D
% NClasses : C
% l1 : energy
% l2 : sum of loss per example
% l3 : ||W||^2
% l4 : |W^2 - 1|
% l5 : max loss per example
% l6 : loss for wrong labels being close to 0. sum_{j \neq y_i} 1/(1 + (a E_j)^2)
% l7 : loss exp decay for wrong labels
%size(Wvec)
m = params.M0;
corr_scale = params.CorrScale;
lambdas = params.LAMBDAS;

% always assume filters to be in 2D matrix format (FxC) x D
% % if(~exist('bNoReshape', 'var'))
% %     bNoReshape = false;
% % end
% % if(bNoReshape)
% %     % Wvec is already in (NFilters x Class) x D format
% %     W = Wvec;
% % else
% %     % W is F x D x C and X is D x N, y is 1 x N with the correct labels
% %     % make W 
% %     Wmat = reshape(Wvec, NFilters, DataDim, NClasses);
% %     W = reshape(permute(Wmat, [2, 1, 3]), [DataDim, NFilters * NClasses])'; % W is now (F x C) x D
% % end
W = reshape(Wvec, NFilters * NClasses, DataDim);

ZeroLossMult = params.ZeroLossMult;

NTrain = size(X, 2);
% A is C x (F x C)
%Energy = 0.5 * A * (W * X).^2; % C x N  % NOTE: Scores = -Energy
[Energy, ~, ~, MQC] = MinQuadraticCostVec(X, W, A); % IndFilters is CxN with index of filters used in the filter banks
%IndFilters = MQC.I;
ind = sub2ind(size(Energy), y, 1:size(Energy, 2));
corr_energy = Energy(ind);
margin = bsxfun(@minus, corr_scale * corr_energy + m, Energy);
margin(ind) = 0;
loss_per_ex = max(0, margin);
[max_loss_per_ex, I_max_loss] = max(margin); % max hinge loss
ZeroLoss = 1./(1 + (ZeroLossMult * Energy).^2); % penalty for wrong energies having close to zero energy
ZeroLoss(ind) = 0; % correct labels should've 0 zero-loss penalty
loss = (lambdas(1) * sum(corr_energy) + lambdas(2) * sum(loss_per_ex(:)) ...
       + lambdas(5) * sum(max_loss_per_ex(:)) ...
       + lambdas(6) * sum(ZeroLoss(:)) ) / NTrain + lambdas(3) * 0.5 * sum(W(:) .* W(:));

%%% compute non-unit norm penalty
finite_norm = diag(W*W');
finite_norm_deviation = finite_norm - params.FiniteNormConst^2;
loss = loss + lambdas(4) * sum(abs(finite_norm_deviation));

%%%% compute loss derivative
% dSdW = -W * X * X'; % (FxC) x D
dLdS = zeros(size(loss_per_ex)); % C x N
dLdS(margin > 0) = -1;
NViolations = sum(-dLdS); % number of constraint violations for each example
dLdS(ind) = corr_scale * NViolations;

%%%% Derivative for Max Hinge Loss
dMHLdS = zeros(NClasses, NTrain);
IndMHL = sub2ind([NClasses, NTrain], I_max_loss, 1:NTrain);
dMHLdS(IndMHL) =  -1;
NViolationsMHL = sum(-dMHLdS);
assert(all(max(NViolationsMHL) <= 1));
dMHLdS(ind) = corr_scale * NViolationsMHL;

%%% derivative for ZeroLoss
dZL = -2 * ZeroLossMult^2 * ZeroLoss.^2 .* Energy;

% CorrClassSelector is CxN with 1 at element (i,j) if class label of jth training
% sample is i
CorrClassSelector = zeros(size(Energy));
CorrClassSelector(ind) = 1;

% % C x N loss derivatives
SumOfLosses = lambdas(1) * CorrClassSelector + lambdas(2) * dLdS + ...
              lambdas(5) * dMHLdS  + lambdas(6) * dZL; %+ lambdas(7) * dExpDecayLoss
          
dW = MQC.fnGrad(SumOfLosses);
 
% % for derivative computation loop over all example for simplicity
%dW = zeros(size(W));
% FilterCount = cumsum(sum(A, 2));
% Offset = [0; FilterCount(1:end-1)];
% IndFilterOffset = bsxfun(@plus, IndFilters, Offset);
% RR = ndgrid(1:NClasses, 1:NTrain);
% AIdxAll = reshape(sub2ind(size(A), RR(:), IndFilterOffset(:)), [NClasses, NTrain]);
% for trainIdx = 1:NTrain
%     %%% Only compute derivatives for the filters that participated in the
%     %%% min cost and loss
%     %AIdx = sub2ind(size(A), 1:NClasses, IndFilterOffset(:,trainIdx));
%     ASelector = zeros(size(A));
%     ASelector(AIdxAll(:,trainIdx)) = 1;
%     Multiplier = lambdas(2) * diag(sum(diag(dLdS(:,trainIdx)) * ASelector)) + lambdas(1) * diag(sum(diag(CorrClassSelector(:,trainIdx)) * ASelector));
%     Multiplier = Multiplier + lambdas(5) * diag(sum(diag(dMHLdS(:,trainIdx)) * ASelector));
%     Multiplier = Multiplier + lambdas(6) * diag(sum(diag(dZL(:,trainIdx)) * ASelector));
%     dW = dW + Multiplier * (W * XX(:,:,trainIdx));
% end

% non-unit penalty gradient
dU = diag(sign(finite_norm_deviation)) * 2 * W;

%
dW = dW / NTrain + lambdas(3) * W + lambdas(4) * dU;

% if(bNoReshape)
%     grad = dW;
% else
%     % reshape to original form
%     grad = permute(reshape(dW', [DataDim, NFilters, NClasses]), [2, 1, 3]);
% end
