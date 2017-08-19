close all
clear
clc

addpath ../
addpath ../autodiff

D = 3;
NFilters = 5;
NTrain = 4;
NClasses = 3;
X = rand(D, NTrain);
W = rand(NFilters, D, NClasses);
y = round((NClasses - 1) * rand(1, NTrain)) + 1
A = zeros(NClasses, NFilters * NClasses);
for i = 1:size(A, 1)
   A(i,(i - 1) * NFilters + 1:(i * NFilters)) = 1; 
end
A
m = 1;
lambdas = [1, 1];
fnLoss = [];
fnEnergy = [];
[R, C, D] = size(W);

Wmat = W;
Wopt = reshape(permute(Wmat, [2, 1, 3]), [C, R * D])'; % W is now (F x C) x D
bNoReshape = true; %false;
[cost, grad] = ObjFnL2RegVec(X, Wopt, y, A, m, lambdas, fnLoss, ...
                                      fnEnergy, R, C, D, bNoReshape)

%%                                  
[errL1, relerrL1, num_grad, grad] = gradcheck(@(W) ObjFnL2RegVec(X, W, y, A, m, lambdas, fnLoss, ...
                                      fnEnergy, R, C, D, bNoReshape), Wopt, 1e-6)
                                  
%%
W_ad = adiff(W)
[x, dx] = autodiff(W_ad, @(W) ObjFnL2RegVec(X, W, y, A, m, lambdas, fnLoss, ...
                                      fnEnergy, R, C, D, bNoReshape))                                  