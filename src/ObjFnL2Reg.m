function [cost, grad, H] = ObjFnL2Reg(X, Wvec, m, corr_scale, lambdas, fnLoss, ...
                                      fnEnergy, R, C, D)

W = reshape(Wvec, R, C, D);
cost = 0;
grad = zeros(size(W));
H = zeros(size(W, 2), size(W, 2), size(W, 3));
%display(['size(W) ' num2str(size(W))])
%if(size(W, 2) == 1690)
%    display('here')
%end
for tIdx = 1:size(X, 2)
    for label = 1:size(W, 3)
        [Li, Gi, Hi] = fnLoss(X(:,tIdx, label), W, m, corr_scale, label, fnEnergy);
        cost = cost + Li;
        grad = grad + Gi;
        H = H + Hi;
    end
end
NTrain = size(X, 2) * size(X, 3);
cost = lambdas(1) * cost / NTrain + lambdas(2) * 0.5 * sum(W(:) .* W(:));
grad = lambdas(1) * grad / NTrain + lambdas(2) * W;

