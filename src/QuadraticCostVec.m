function [cost, grad, hessian, retVal] = QuadraticCostVec(X, W, A)
% Assume that X is all the test/training examples i.e. C x N
% x = x(:);
% if(ndims(W) == 1)
%     W = reshape(W(:), 1, []);
% end
% 
WX = W * X;
E = 0.5 * WX.^2;
cost = A * E; 
hessian = X * X';
grad = W * hessian;
retVal.fnGrad = @(derivIn) QuadraticCostBackprop(X, A, WX, E, derivIn); % return backprop closure
end

function [dW, dA] = QuadraticCostBackprop(X, A, WX, E, derivIn)
dA = derivIn * E'; % C x N * N x (F x C)
Multiplier = derivIn' * A; % N x (F x C)
MWX = Multiplier' .* WX; % *element-wise prod* (F x C) x N
dW = MWX * X';
end
