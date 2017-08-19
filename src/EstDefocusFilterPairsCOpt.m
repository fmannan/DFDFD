function res = EstDefocusFilterPairsCOpt(ImageDepthSet, params, filterBankSize)

% constrainted optimization to find filter pairs for every depth
% based on the unconstrained version in estDefocusFilterPairs.m

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

ERR_CONST = 1;
if(isfield(params, 'ERR_CONST'))
    ERR_CONST = params.ERR_CONST;
end

NDepth = length(ImageDepthSet);
MatSize = size(ImageDepthSet{1,1}, 1);
KSize = sqrt(MatSize/filterBankSize);
Filters = cell(NDepth, filterBankSize * NFilters);
Q = blkdiag(lambda * sparse(ImageDepthSet{1}));
for idx = 2:NDepth
    Q = blkdiag(Q, sparse(lambda * ImageDepthSet{idx}));
end

A = [];
B = [];
Aeq = [];
Beq = [];
LB = [];
UB = [];
useVersion = 2;
if(isfield(params, 'useVersion'))
    useVersion = params.useVersion;
end

if(useVersion == 1) % have at least a specified separation
    NONLCON = @(x) fnNLConV1(x,ERR_CONST,ImageDepthSet,MatSize); 
    OBJFN = @(x) fnObjV1(x, Q);
    X0 = rand(size(Q, 1), 1);
elseif(useVersion == 2) % maximize separation
    NONLCON = @(x) fnNLConV2(x,ERR_CONST,ImageDepthSet,MatSize); 
    OBJFN = @(x) fnObjV2(x, Q);
    X0 = rand(size(Q, 1) + 1, 1);
    X0(end) = ERR_CONST;
else
    error('no version specified');
end

if(isfield(params, 'X0'))
    X0 = params.X0;
end

DerivCheck = 'off';
if(isfield(params, 'DerivCheck'))
    DerivCheck = params.DerivCheck;
end
% 'FinDiffType', 'central',
options = optimoptions(@fmincon,'Algorithm','interior-point', ...
          'UseParallel', 'Always', 'Display','off', ...
          'MaxIter', 1e5, 'MaxFunEvals', 1e7, 'DerivativeCheck', DerivCheck, ...
        'GradObj','on','GradConstr','on'); %,...
       %'Hessian','user-supplied','HessFcn',@(x, lambda) hessinterior_v1(x, lambda, Q)); %
[X, FVAL, EXITFLAG, OUTPUT, LAMBDA] = fmincon(OBJFN ,X0, A,B,Aeq,Beq,LB,UB,NONLCON, options);

for idxFilter = NFilters:-1:1
    for idx = 1:NDepth
        StartIdx = (idx - 1) * MatSize;
        IDX = StartIdx + (1:MatSize);
        F = X(IDX, idxFilter);
        if(filterBankSize == 1)
            Filters{idx, idxFilter} = reshape(F, KSize, KSize);
        elseif(filterBankSize == 2)
            f1 = reshape(F(1:KSize * KSize), KSize, KSize);
            f2 = reshape(F(KSize * KSize + 1:end), KSize, KSize);

            Filters{idx, 1 + 2*(idxFilter - 1)} = f1;
            Filters{idx, 2 + 2*(idxFilter - 1)} = f2;
        else
            error(['filterBankSize = ' num2str(filterBankSize) ' not Supported'])
        end
        
    end
end
res.Filters = Filters;
res.X = X;
res.X0 = X0;
res.FVAL = FVAL;
res.EXITFLAG = EXITFLAG;
res.OUTPUT = OUTPUT;
res.LAMBDA = LAMBDA;
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
%display('fnObjV2')
    %display(num2str(x(end)));
    cost = 0.5 * x(1:end-1)' * Q * x(1:end-1) - 0.5 *  x(end);
    grad = [Q * x(1:end-1);  -0.5];
    hessian = blkdiag(Q, 0);
end


% constraint functions
function [c, ceq, gradc, gradceq] = fnNLConV1(x,const, ImageDepthSet, MatSize)
    ceq = [];
    
    NDepth = length(ImageDepthSet);
    c = nan(NDepth * (NDepth - 1), 1);
    counter = 1;
    for idx1 = 1:NDepth
        X1 = x(MatSize * (idx1 - 1)  + (1:MatSize));
        AA1 = X1' * ImageDepthSet{idx1} * X1;
        for idx2 = 1:NDepth
           X2 = x(MatSize * (idx2 - 1) + (1:MatSize));
           if(idx1 ~= idx2)
              c(counter) = AA1 - X2' * ImageDepthSet{idx2} * X2 + const; %%% THIS LOOKS WRONG!!! SHOULD BE -X1' * ... * X1
              counter = counter + 1;
           end
       end
    end
    
    if nargout > 2 % gradient of constraints
        gradceq = [];
        
        gradc = zeros(size(x, 1), NDepth * (NDepth - 1));
        counter = 1;
        for idx1 = 1:NDepth
            X1 = x(MatSize * (idx1 - 1)  + (1:MatSize));
            dAA1 = ImageDepthSet{idx1} * X1;
            %AA1 = X1' * dAA1;
            
            for idx2 = 1:NDepth
               X2 = x(MatSize * (idx2 - 1) + (1:MatSize));
               if(idx1 ~= idx2)
                   dAA2 = ImageDepthSet{idx2} * X2;
                   gradc(MatSize * (idx1 - 1)  + (1:MatSize),counter) = 2 * dAA1;
                   gradc(MatSize * (idx2 - 1)  + (1:MatSize),counter) = - 2 * dAA2;
                   counter = counter + 1;
               end
           end
        end
    end
end

% constraint functions v2
function [c, ceq, gradc, gradceq] = fnNLConV2(x,const, ImageDepthSet, MatSize)
    ceq = [];
    
    NDepth = length(ImageDepthSet);
    c = nan(NDepth * (NDepth - 1) + 1, 1);
    counter = 1;
    for idx1 = 1:NDepth
        X1 = x(MatSize * (idx1 - 1)  + (1:MatSize));
        AA1 = X1' * ImageDepthSet{idx1} * X1;
        for idx2 = 1:NDepth
           X2 = x(MatSize * (idx2 - 1) + (1:MatSize));
           if(idx1 ~= idx2)
              c(counter) = AA1 - X2' * ImageDepthSet{idx2} * X2 + x(end);
              counter = counter + 1;
           end
       end
    end
    c(counter) = const - x(end);
    
    if nargout > 2 % gradient of constraints
        gradceq = [];
        
        gradc = zeros(size(x, 1), NDepth * (NDepth - 1) + 1);
        gradc(end,1:end-1) = 1; % d/dt = 1
        gradc(end, end) = -1;   % d/dt (const - t) = -1
        counter = 1;
        for idx1 = 1:NDepth
            X1 = x(MatSize * (idx1 - 1)  + (1:MatSize));
            dAA1 = ImageDepthSet{idx1} * X1;
            %AA1 = X1' * dAA1;
            
            for idx2 = 1:NDepth
               X2 = x(MatSize * (idx2 - 1) + (1:MatSize));
               if(idx1 ~= idx2)
                   dAA2 = ImageDepthSet{idx2} * X2;
                   gradc(MatSize * (idx1 - 1)  + (1:MatSize),counter) = 2 * dAA1;
                   gradc(MatSize * (idx2 - 1)  + (1:MatSize),counter) = - 2 * dAA2;
                   
                   counter = counter + 1;
               end
           end
        end
    end
end