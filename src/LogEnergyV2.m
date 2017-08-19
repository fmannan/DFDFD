function [cost, grad, hessian, retVal] = LogEnergyV2(X, W, A, A0, fnEnergy)
% TODO: MAKE A THAT IS PART OF fnEnergy passed as closure

[E, grad, hessian, retValEnergy] = fnEnergy(X, W, A0);
retVal.fnGrad = @(derivIn) LogEnergyBackprop(derivIn, E, A, retValEnergy.fnGrad); % return backprop closure
cost = A * log(E);
end

function [dW, dA] = LogEnergyBackprop(derivIn, Energy, A, fnEnergyGrad)
dA = derivIn * Energy'; % C x N * N x (F x C)
Multiplier = derivIn' * A; % N x (F x C)
MWX = Multiplier' ./ Energy; % *element-wise prod* (F x C) x N
dW = fnEnergyGrad(MWX);
end
