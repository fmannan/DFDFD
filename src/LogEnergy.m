function [cost, grad, hessian, retVal] = LogEnergy(X, W, A, fnEnergy)
% TODO: MAKE A THAT IS PART OF fnEnergy passed as closure

[E, grad, hessian, retValEnergy] = fnEnergy(X, W, A);
retVal.fnGrad = @(derivIn) LogEnergyBackprop(derivIn, E, retValEnergy.fnGrad); % return backprop closure
cost = log(E);
end

function dW = LogEnergyBackprop(derivIn, Energy, fnEnergyGrad)
dW = fnEnergyGrad(derivIn./Energy);
end
