function [cost, grad, hessian] = QuadraticCost(x, W)

%size(W)
%size(x)
%if(size(W, 2) == 1690)
%    display('here')
%end
x = x(:);
if(ndims(W) == 1)
    W = reshape(W(:), 1, []);
end

cost = 0.5 * x' * W' * W * x; % works for both vector and matrix W
hessian = x * x';
grad = W * hessian;

