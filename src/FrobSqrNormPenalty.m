function [cost, dW] = FrobSqrNormPenalty(W)
% w^2
% NOTE: if W is a 1D vec then it should be a row

finite_norm = 0.5 * W.*W;
cost = sum(abs(finite_norm(:)));

dW = W;
