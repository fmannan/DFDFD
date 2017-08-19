function [cost, grad, hessian, retVal] = L1CostVec(X, W, A)
% Assume that X is all the test/training examples i.e. C x N
% x = x(:);
% if(ndims(W) == 1)
%     W = reshape(W(:), 1, []);
% end
% 
C = size(A, 1);
F = size(W, 1) / C;
N = size(X, 2);
WX = W * X;
E = abs(WX);
cost = A * E; % (FxC) x N

retVal.fnGrad = @(derivIn) L1CostBackprop(X, A, WX, E, derivIn); % return backprop closure

% the following is only valid if X is a single vector
hessian = [];
grad = [];
if(size(X, 2) == 1)
    grad = sign(WX) * X';
end

end

function [dW, dA] = L1CostBackprop(X, A, WX, E, derivIn)
dA = derivIn * E';
Multiplier = derivIn' * A; % N x (F x C)
MWX = Multiplier' .* sign(WX); % *element-wise prod* (F x C) x N
dW = MWX * X';
% NTrain = size(X, 2);
% dW = zeros(size(W));
% parfor trainIdx = 1:NTrain 
%     %L1EnergyDeriv = sign(WX(:,trainIdx)) * X(:,trainIdx)';
%     %dW = dW + diag(Multiplier(trainIdx,:)) * L1EnergyDeriv;
%     dW = dW + MWX(:, trainIdx) * X(:, trainIdx)';
% end
end

