function Res = EstDefocusFilterQuadraticUnc(ImageDepthSet, params, filterBankSize)
% ImageDepthSet is the set of defocused image patches
% of the form A' * A where A is the convolution matrix for an image.

lambda = 1e5;
if(isfield(params, 'lambda'))
    lambda = params.lambda;
end
NFilters = 1;
if(isfield(params, 'NFilters'))
    NFilters = params.NFilters;
end
if(~exist('filterBankSize', 'var'))
    filterBankSize = 2;
end
a = 1;
b = 1;
if(isfield(params, 'penalty'))
    if(isfield(params.penalty, 'a'))
        a = params.a;
    end
    if(isfield(params.penalty, 'b'))
        b = params.b;
    end
end
NDepth = length(ImageDepthSet);
MatSize = size(ImageDepthSet{1,1}, 1);
KSize = sqrt(MatSize/filterBankSize);
Filters = cell(NDepth, filterBankSize * NFilters);


X0 = rand(NDepth * MatSize, 1);
options = optimoptions(@fminunc,'Algorithm','trust-region', ...
           'Display','off', ...
          'MaxIter', 1e4, 'MaxFunEvals', 1e7, 'DerivativeCheck','off', ...
       'FinDiffType', 'central', 'GradObj','on'); %,...
       %'Hessian','user-supplied','HessFcn',@(x, lambda) hessinterior_v1(x, lambda, Q)); %
[X, FVAL, EXITFLAG, OUTPUT] = fminunc(@(x) fnObjV1(x, lambda, @(y) expLoss(y, a, b), ImageDepthSet) ,X0, options);


for idx = 1:NDepth
    IDX = (idx - 1) * MatSize + (1:MatSize);
    F = X(IDX);
    for idx2 = 1:filterBankSize
        Filters{idx, idx2} = reshape(F(KSize * KSize * (idx2 - 1) + (1:KSize * KSize)), KSize, KSize);
    end
end

Res.Filters = Filters;
Res.X = X;
Res.FVAL = FVAL;
Res.EXITFLAG = EXITFLAG;
Res.OUTPUT = OUTPUT;

end

% objective functions
function [cost, grad] = fnObjV1(x, lambda, fnConstrCost, ImageDepthSet)
    % ImageDepthSet is assumed to be A'*A in this case of Quadratic cost
    cost = 0;
    grad = zeros(length(x), 1);
    NDepth = length(ImageDepthSet);
    BlockSize = size(ImageDepthSet{1}, 2);
    for idx = 1:NDepth
        IDX = (idx - 1) * BlockSize + (1:BlockSize);
        cost = cost + 0.5 * x(IDX)' * ImageDepthSet{idx} * x(IDX);
        grad(IDX) = ImageDepthSet{idx} * x(IDX);
        for idx2 = 1:NDepth
            if(idx ~= idx2)
                IDX2 = (idx2 - 1) * BlockSize + (1:BlockSize);
                dist = 0.5 * (x(IDX2)' * ImageDepthSet{idx2} * x(IDX2) - x(IDX)' * ImageDepthSet{idx} * x(IDX));
                [bCost, bGrad] = fnConstrCost(dist);
                cost = cost + lambda * bCost;
                grad(IDX) = grad(IDX) - lambda * bGrad * ImageDepthSet{idx} * x(IDX);
            end
        end
        
    end
end

function h = hessinterior_v1(~, ~, Q)
    h = Q;
end

function [c, gc] = expLoss(x, a, b)
%     if(x < -100) 
%         disp(x)
%     end
    x = max(x, -100);
    c = exp(-a * (x - b));
    gc = -a * c;
end



