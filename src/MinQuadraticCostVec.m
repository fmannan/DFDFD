function [cost, grad, hessian, retVal] = MinQuadraticCostVec(X, W, A)
% Assume that X is all the test/training examples i.e. C x N
% x = x(:);
% if(ndims(W) == 1)
%     W = reshape(W(:), 1, []);
% end
% 

% C = size(A, 1);
% F = size(W, 1) / C;
% N = size(X, 2);

WX = W * X;
cost = 0.5 * (WX).^2; % (FxC) x N
[cost, MinReducerGradFn, ReducerRetVal] = MinMaxReducer(cost, A, @min);
% [cost, I] = min(reshape(cost, [F, C * N]));
% cost = 0.5 * reshape(cost, [C, N]);
% I = reshape(I, [C, N]);
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
% 
% retVal.gradFn = @(derivIn) QuadraticCostBackprop(X, WX, A, ASelector, derivIn); % return backprop closure
retVal.fnGrad = @(derivIn) QuadraticCostBackpropV2(X, WX, MinReducerGradFn, derivIn); % return backprop closure
retVal.I = ReducerRetVal.I;
% the following is only valid if X is a single vector
hessian = X * X';
grad = W * hessian;

end
% 
% function dW = QuadraticCostBackprop(X, WX, A, ASelector, derivIn)
% Multiplier = (derivIn' * A)' .* ASelector; % N x (F x C) -> (F x C) x N
% MWX = Multiplier .* WX; % *element-wise prod* (F x C) x N
% dW = MWX * X';
% end

function dW = QuadraticCostBackpropV2(X, WX, MinReducerGradFn, derivIn)
Multiplier = MinReducerGradFn(derivIn); % N x (F x C) -> (F x C) x N
MWX = Multiplier .* WX; % *element-wise prod* (F x C) x N
dW = MWX * X';
end


