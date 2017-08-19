function result = EstDefocusFilterIterUnc(ImageDepthSet, params)
% uses fminunc or custom implementation of grad desc.
% for simplicity and clarity the caller composes the required functions and
% provides it as fnObj

Algorithm = 'trust-region'; %'quasi-newton'; %
if(isfield(params, 'Algorithm'))
    Algorithm = params.Algorithm;
end

DerivCheck = 'off';
if(isfield(params, 'DerivCheck'))
    DerivCheck = params.DerivCheck;
end

Display = 'off';
if(isfield(params, 'Display'))
    Display = params.Display;
end

bSteepestDescent = false;
if(isfield(params, 'bSteepestDescent'))
    bSteepestDescent = params.bSteepestDescent;
end

bOptA = false;
if(isfield(params, 'bOptA'))
   bOptA =  params.bOptA;
end

bUsefminunc = params.bUsefminunc;

if(~bUsefminunc && isfield(params, 'fnOpt'))
    % if custom opt is used then caller has to provide option and
    % bUsefminunc has to be set to false
    fnOpt = params.fnOpt;
    options = params.options;
else
    fnOpt = @fminunc;
end

bBatchGD = true;
if(isfield(params, 'bBatchGD'))
   bBatchGD = params.bBatchGD; 
end

if(bBatchGD) % run steepest descent iteratively for M batch size for NBatchIterations
    NBatchIterations = params.NBatchIterations;
    BatchSize = params.BatchSize;
    bSteepestDescent = true;
    
    bBGDVerbose = true;
    BGDPrintInterval = 10;
    if(isfield(params, 'bBGDVerbose'))
        bBGDVerbose = params.bBGDVerbose;
    end
    
end    
if(bUsefminunc)
    if(bSteepestDescent)
        options = optimoptions(@fminunc, 'Display', Display, 'Algorithm', 'quasi-newton', ...
                           'GradObj', 'on', 'Hessian', 'off', 'HessUpdate', 'steepdesc', 'DerivativeCheck', DerivCheck)    
    else
        options = optimoptions(@fminunc, 'Display', Display, 'Algorithm', Algorithm, ...
                           'GradObj', 'on', 'Hessian', 'off', 'DerivativeCheck', DerivCheck)
    end
end
options.MaxIter = 1000;
if(isfield(params, 'MaxIter'))
    options.MaxIter = params.MaxIter;
end

if(isfield(params, 'NFilters'))
    NFilters = params.NFilters; % number of filters per depth
end

Dataset = genDataMatrix(ImageDepthSet, params.fnLabelLoss);
DataDim = Dataset.DataDim;
NClasses = Dataset.NClasses;
%%%
W0 = rand(NFilters, DataDim, NClasses);

if(isfield(params, 'W0'))
    W0 = params.W0;
    assert(NFilters == size(W0, 1));
end

assert(isfield(params, 'LAMBDAS'));

% bUseVectorized = true;
% if(isfield(params, 'bUseVectorized'))
%     bUseVectorized = params.bUseVectorized;
% end


fnObj = params.fnObj; % make fnObj mandatory


%if(bUseVectorized || bBatchGD)
   
W0_2D = convertFilter3Dto2DFormat(W0);
OptVars = W0_2D(:);

%%%%
Ainit = zeros(NClasses, NFilters * NClasses);
for i = 1:size(Ainit, 1)
    Ainit(i,(i - 1) * NFilters + 1:(i * NFilters)) = 1; 
end

%%%%
if(bOptA)
   A = params.A0;
   A1 = params.A1;
   if(A == 1 && isnan(A1)) % automatic initialization
      A1 = Ainit + randn(size(Ainit)) * 1e-4;
   end
   A1init = A1;
   OptVars = [OptVars; A1(:)];
else
   A = Ainit;  
end

% PREPARE X AND y
Xtrain = Dataset.X_flat; %reshape(X, [size(X, 1), size(X, 2) * size(X, 3)]);
y = Dataset.y;

NTrain = size(Xtrain, 2);

PerLabelLoss = Dataset.PerLabelLoss; %params.fnLabelLoss(bsxfun(@minus, ndgrid(1:NClasses, 1:NTrain), y));
params.PerLabelLoss = PerLabelLoss;
%     XX = nan(DataDim, DataDim, NTrain);
%     for i = 1:NTrain
%         XX(:,:,i) = Xtrain(:,i) * Xtrain(:,i)';
%     end
if(bBatchGD) % run steepest descent iteratively for M batch size for NBatchIterations
    Wprev = OptVars; %W0_2D;
    for outerIter = 1:NBatchIterations
        % choose batch
        if(BatchSize == NTrain)
            batchIdx = 1:NTrain;
        else
            batchIdx = randperm(NTrain);
            batchIdx = batchIdx(1:BatchSize);
        end
        XtrainBatch = Xtrain(:, batchIdx);
        yBatch = y(batchIdx);
        PerLabelLossBatch = PerLabelLoss(:, batchIdx);
        params.PerLabelLoss = PerLabelLossBatch;
        % optimize
        [Wvec, feval, exitflag, output] = fnOpt(@(W) fnObj(XtrainBatch, W, yBatch, A, params, ...
                                            NFilters, DataDim, NClasses), Wprev(:), options);

        % update to current optimal
        Wprev = Wvec;
        
        if(bBGDVerbose)
            
        end
        
        %if(bBGDSaveIntermediate)
            % save to file (TODO)
            % ...
            %end
    end
else
    [Wvec, feval, exitflag, output] = fnOpt(@(W) fnObj(Xtrain, W, y, A, params, ...
                                               NFilters, DataDim, NClasses), OptVars(:), options);
end
OptVarsRes = Wvec;
Wsize = NFilters * NClasses * DataDim;
Wvec = reshape(OptVarsRes(1:Wsize), NFilters * NClasses, DataDim);
W = convertFilter2Dto3DFormat(Wvec, NFilters, NClasses);
if(bOptA)
    A1 =  reshape(OptVarsRes(Wsize+1:end), NClasses, []);
end
result.W = W;
result.W0 = W0;
result.Wvec = Wvec;
if(bOptA)
    result.A0 = A;
    result.A1 = A1;
    result.A1init = A1init;
else
    result.A = A;
end
result.feval = feval;
result.exitflag = exitflag;
result.output = output;

