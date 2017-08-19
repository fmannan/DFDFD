function [cost, gradFn, retVal] = MinReducer(E, A)

C = size(A, 1);
F = size(E, 1) / C;
N = size(E, 2);

cost = E; % (FxC) x N
[cost, I] = min(reshape(cost, [F, C * N]));
cost = reshape(cost, [C, N]);

FilterCount = cumsum(sum(A, 2));
Offset = [0; FilterCount(1:end-1)];
IndFilterOffset = bsxfun(@plus, I, Offset);
[~, CC] = ndgrid(1:(F*C), 1:N);
AIdxAll = reshape(sub2ind(size(A), IndFilterOffset(:), CC(:)), [F*C, N]);
ASelector = zeros(F*C, N); % use sparse data-structure? ASelector = sparse(IndFilterOffset(:), CC(:), ones(numel(IndFilterOffset), 1), F*C, N);
ASelector(AIdxAll) = 1;

retVal.I = reshape(I, [C, N]);
retVal.AIdxAll = AIdxAll;
gradFn = @(derivIn) MinReducerBackprop(A, ASelector, derivIn);
retVal.gradFn = gradFn;
end

function grad = MinReducerBackprop(A, ASelector, derivIn)
grad = (derivIn' * A)' .* ASelector; % N x (F x C) -> (F x C) x N
end


