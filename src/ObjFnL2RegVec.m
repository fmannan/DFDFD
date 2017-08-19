function [loss, grad] = ObjFnL2RegVec(X, Wvec, y, A, XX, m, corr_scale, lambdas, fnLoss, ...
                                      fnEnergy, NFilters, DataDim, NClasses, bNoReshape)
% implement vectorized objective function with hinge loss and quadratic
% cost
% NFiters : F
% DataDim : D
% NClasses : C
%size(Wvec)
if(~exist('bNoReshape', 'var'))
    bNoReshape = false;
end
if(bNoReshape)
    % Wvec is already in (NFilters x Class) x D format
    W = Wvec;
else
    % W is F x D x C and X is D x N, y is 1 x N with the correct labels
    % make W 
    Wmat = reshape(Wvec, NFilters, DataDim, NClasses);
    W = reshape(permute(Wmat, [2, 1, 3]), [DataDim, NFilters * NClasses])'; % W is now (F x C) x D
end

NTrain = size(X, 2);

% A is C x (F x C)
%A = -A;
Energy = 0.5 * A * (W * X).^2; % C x N  % NOTE: Scores = -Energy
ind = sub2ind(size(Energy), y, 1:size(Energy, 2));
corr_energy = Energy(ind);
margin = bsxfun(@minus, corr_scale * corr_energy + m, Energy);
margin(ind) = 0;
loss_per_ex = max(0, margin);

loss = (lambdas(1) * sum(corr_energy) + lambdas(2) * sum(loss_per_ex(:))) / NTrain + lambdas(3) * 0.5 * sum(W(:) .* W(:));

%%% compute non-unit norm penalty
unit_norm = diag(W*W');
unit_norm_deviation = unit_norm - 1;
loss = loss + lambdas(4) * sum(abs(unit_norm_deviation));

%%%% compute derivative
% 
% dSdW = -W * X * X'; % (FxC) x D
dLdS = zeros(size(loss_per_ex)); % C x N
dLdS(margin > 0) = -1;
NViolations = sum(-dLdS); % number of constraint violations for each example
dLdS(ind) = corr_scale * NViolations;
% dLdSsum = sum(dLdS, 2);
% dW = diag(sum(diag(dLdSsum) * A, 1)) * dSdW;

% CorrClassSelector is CxN with 1 at element (i,j) if class label of jth training
% sample is i
CorrClassSelector = zeros(size(Energy));
CorrClassSelector(ind) = 1;

% for derivative computation loop over all example for simplicity
dW = zeros(size(W));
for trainIdx = 1:NTrain
    dW = dW + (lambdas(2) * diag(sum(diag(dLdS(:,trainIdx)) * A)) + lambdas(1) * diag(sum(diag(CorrClassSelector(:,trainIdx)) * A)) ) * (W * XX(:,:,trainIdx));
end

% non-unit penalty gradient
dU = diag(sign(unit_norm_deviation)) * 2 * W;

%
dW = dW / NTrain + lambdas(3) * W + lambdas(4) * dU;

if(bNoReshape)
    grad = dW;
else
    % reshape to original form
    grad = permute(reshape(dW', [DataDim, NFilters, NClasses]), [2, 1, 3]);
end
