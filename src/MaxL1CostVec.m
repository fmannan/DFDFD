function [cost, grad, hessian, retVal] = MaxL1CostVec(X, W, A)
% Assume that X is all the test/training examples i.e. C x N

WX = W * X;
cost = abs(WX); % (FxC) x N
[cost, MaxReducerGradFn] = MinMaxReducer(cost, A, @max);

retVal.gradFn = @(derivIn) MaxL1CostBackpropV2(X, WX, MaxReducerGradFn, derivIn);

% the following is only valid if X is a single vector
hessian = [];
grad = [];
if(size(X, 2) == 1)
    grad = sign(WX) * X';
end

end

function dW = MaxL1CostBackpropV2(X, WX, MaxGradFn, derivIn)
Multiplier = MaxGradFn(derivIn); 
MWX = Multiplier .* sign(WX); % *element-wise prod* (F x C) x N
dW = MWX * X';
end


