function res = EstDefocusFilterPairsBasicQCQP(ImageDepthSet, params, filterBankSize)

% constrainted optimization to find filter pairs for every depth
% based on the unconstrained version in EstDefocusFilterPairs.m

% ImageDepthSet is the set of defocused image patches
% of the form A' * A where A is the convolution matrix for an image.

% In this basic setting the optimization problem is:
% argmin_x x'Ax - t
% s.t. x'Aix - x'Ajx + t <= 0
% t >= |e|

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
NDepth = length(ImageDepthSet);
MatSize = size(ImageDepthSet{1,1}, 1);
KSize = sqrt(MatSize/filterBankSize);
Filters = cell(NDepth, filterBankSize * NFilters);
% find filter pairs for every depth separately by solving the optimization
% problem
ERR_CONST = 10;
if(isfield(params, 'ERR_CONST'))
    ERR_CONST = params.ERR_CONST;
end

A = [];
B = [];
Aeq = [];
Beq = [];
LB = [];
UB = [];
X0_all = []; %zeros(numel(ImageDepthSet), 1);
if(isfield(params, 'X0'))
    X0_all = params.X0;
end
for idx = 1:NDepth
    Q = lambda * ImageDepthSet{idx};
    X0 = rand(size(Q, 1), 1);
    if(~isempty(X0_all))
        X0 = X0_all((idx -1) * MatSize + (1:MatSize));
    end
    
    NONLCON = @(x) fnNLConV1(x,ERR_CONST,ImageDepthSet,MatSize, idx); 
    options = optimoptions(@fmincon,'Algorithm','interior-point', ...
          'UseParallel', 'Always', 'Display','off', ...
          'MaxIter', 1e6, 'MaxFunEvals', 1e7, 'DerivativeCheck','off', ...
       'FinDiffType', 'central', 'GradObj','on','GradConstr','on'); %,...
       %'Hessian','user-supplied','HessFcn',@(x, lambda) hessinterior_v1(x, lambda, Q)); %
    [X, ~, ~, OUTPUT] = fmincon(@(x) fnObjV1(x, Q) ,X0, A,B,Aeq,Beq,LB,UB,NONLCON, options);
    display(OUTPUT)
    display(OUTPUT.message)
    if(filterBankSize == 1)
        Filters{idx} = reshape(X, KSize, KSize);
    elseif(filterBankSize == 2)
        f1 = reshape(X(1:KSize * KSize), KSize, KSize);
        f2 = reshape(X(KSize * KSize + 1:end), KSize, KSize);

        Filters{idx, 1} = f1;
        Filters{idx, 2} = f2;
    else
        error(['filterBankSize = ' num2str(filterBankSize) ' not Supported'])
    end
end

res.Filters = Filters;

end

% objective functions
function [cost, grad, hessian] = fnObjV1(x, Q)
    cost = 0.5 * x' * Q * x;
    grad = Q * x;
    hessian = Q;
end

function h = hessinterior_v1(~, ~, Q)
    h = Q;
end

function [cost, grad, hessian] = fnObjV2(x, Q)
    cost = x(1:end-1)' * Q * x(1:end-1) - x(end);
    grad = [Q * x(1:end-1); -1];
    hessian = blkdiag(Q, 0);
end


% constraint functions
function [c, ceq, gradc, gradceq] = fnNLConV1(x,const, ImageDepthSet, MatSize, depthIdx)
    ceq = [];
    
    NDepth = length(ImageDepthSet);
    c = nan((NDepth - 1), 1);
    counter = 1;
    

    AA1 = x' * ImageDepthSet{depthIdx} * x;
    for idx2 = 1:NDepth
       if(depthIdx ~= idx2)
          c(counter) = AA1 - x' * ImageDepthSet{idx2} * x + const;
          counter = counter + 1;
       end
    end
   
    if nargout > 2 % gradient of constraints
        gradceq = [];
        
        gradc = zeros(size(x, 1), (NDepth - 1));
        counter = 1;
       
        dAA1 = ImageDepthSet{depthIdx} * x;

        for idx2 = 1:NDepth
           if(depthIdx ~= idx2)
               dAA2 = ImageDepthSet{idx2} * x;
               gradc(:,counter) = 2 * (dAA1 - dAA2);
               counter = counter + 1;
           end
       end
       
    end
end

% % constraint functions v2 (NOT IMPLEMENTED)
% function [c, ceq, gradc, gradceq] = fnNLConV2(x,const, ImageDepthSet, MatSize, depthIdx)
%     ceq = [];
%     
%     NDepth = length(ImageDepthSet);
%     c = nan((NDepth - 1), 1);
%     counter = 1;
%     
% 
%     AA1 = x' * ImageDepthSet{depthIdx} * x;
%     for idx2 = 1:NDepth
%        if(depthIdx ~= idx2)
%           c(counter) = AA1 - x' * ImageDepthSet{idx2} * x + const;
%           counter = counter + 1;
%        end
%     end
%    
%     if nargout > 2 % gradient of constraints
%         gradceq = [];
%         
%         gradc = zeros(size(x, 1), (NDepth - 1));
%         counter = 1;
%        
%         dAA1 = ImageDepthSet{depthIdx} * x;
% 
%         for idx2 = 1:NDepth
%            if(depthIdx ~= idx2)
%                dAA2 = ImageDepthSet{idx2} * x;
%                gradc(:,counter) = 2 * (dAA1 - dAA2);
%                counter = counter + 1;
%            end
%        end
%        
%     end
% end