function [cost, grad, hessian, retVal] = AffineCostVec(X, W, A)
% Assume that X is all the test/training examples i.e. C x N
WX = W * X;
cost = A * WX; % (FxC) x N

retVal.fnGrad = @(derivIn) AffineCostBackprop(X, A, WX, derivIn); % return backprop closure


% the following is only valid if X is a single vector
hessian = [];
grad = [];
if(size(X, 2) == 1)
    grad = sign(WX) * X';
end

end

function [dW, dA] = AffineCostBackprop(X, A, WX, derivIn)
dA = derivIn * WX';
Multiplier = derivIn' * A; % N x (F x C)
dW = Multiplier' * X'; % *element-wise prod* (F x C) x N
end


