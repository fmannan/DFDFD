function [cost, grad, hessian, retVal] = MinL1CostVec(X, W, A)
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
cost = abs(WX); % (FxC) x N
[cost, MinReducerGradFn] = MinMaxReducer(cost, A, @min);
% [cost, I] = min(reshape(cost, [F, C * N]));
% cost = reshape(cost, [C, N]);
% 
% FilterCount = cumsum(sum(A, 2));
% Offset = [0; FilterCount(1:end-1)];
% IndFilterOffset = bsxfun(@plus, I, Offset);
% [~, CC] = ndgrid(1:(F*C), 1:N);
% AIdxAll = reshape(sub2ind(size(A), IndFilterOffset(:), CC(:)), [F*C, N]);
% ASelector = zeros(F*C, N); % use sparse data-structure? ASelector = sparse(IndFilterOffset(:), CC(:), ones(numel(IndFilterOffset), 1), F*C, N);
% ASelector(AIdxAll) = 1;
% 
% retVal.I = reshape(I, [C, N]);
% retVal.AIdxAll = AIdxAll;
% %retVal.ASelector = ASelector;
% retVal.gradFn = @(derivIn) MinL1CostBackprop(X, WX, A, ASelector, derivIn); % return backprop closure
retVal.fnGrad = @(derivIn) MinL1CostBackpropV2(X, WX, MinReducerGradFn, derivIn);
%retVal.gradFn = @(derivIn) MinL1Cost
% the following is only valid if X is a single vector
hessian = [];
grad = [];
if(size(X, 2) == 1)
    grad = sign(WX) * X';
end

end

% function dW = MinL1CostBackprop(X, WX, A, ASelector, derivIn)
% %eliding call to MinReducerBackprop
% Multiplier = (derivIn' * A)' .* ASelector; % N x (F x C) -> (F x C) x N
% MWX = Multiplier .* sign(WX); % *element-wise prod* (F x C) x N
% dW = MWX * X';
% end

function dW = MinL1CostBackpropV2(X, WX, MinGradFn, derivIn)
Multiplier = MinGradFn(derivIn); 
MWX = Multiplier .* sign(WX); % *element-wise prod* (F x C) x N
dW = MWX * X';
end


