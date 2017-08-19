function [cost, dW] = AbsL2SqrNormPenalty(W, norm_const)
% |w^2 - const|
% NOTE: if W is a 1D vec then it should be a row

finite_norm = diag(W*W');
finite_norm_deviation = finite_norm - norm_const^2;

cost = sum(abs(finite_norm_deviation));

dW = diag(sign(finite_norm_deviation)) * 2 * W;
