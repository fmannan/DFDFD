function result = EvalFilterCost(ImageDepthSet, FilterVecs, fnCostEval, fnMinMax, cost_scale, bNeg, fnCostSumNormalization)
% take a set of filters, cost function and data and evaluate the
% performance (??)
% cost per example, average cost, loss

if(~exist('cost_scale', 'var'))
    cost_scale = 1;
end

Dataset = genDataMatrix(ImageDepthSet, @(x)(x));
NClasses = Dataset.NClasses; %length(ImageDepthSet);
NFilters = size(FilterVecs, 1);

%[DataDim, ExamplesPerClass] = size(ImageDepthSet{1});
%DataDim = Dataset.DataDim;

% X = reshape(cell2mat(ImageDepthSet), [DataDim, ExamplesPerClass, NClasses]);
% X_flat = reshape(X, [size(X, 1), size(X, 2) * size(X, 3)]);
%X_flat_noise = X_flat + NoiseStd * randn(size(X_flat));
% [~, y] = meshgrid(1:size(X, 2), 1:size(X, 3)); %nan(1, NTrain);
%                                                %%round((NClasses
%                                                %- 1) * rand(1,
%                                                %NTrain)) + 1
% y = reshape(y', 1, []);
% YmatIdx = sub2ind([length(y), max(y)], 1:length(y), y);
% Ymat = zeros(length(y), max(y));
% Ymat(YmatIdx) = 1;
% YmatUnnorm = Ymat;
% Ymat = Ymat * diag(1./sum(Ymat)); % normalize

X_flat = Dataset.X_flat;
y = Dataset.y;
YmatUnnorm = Dataset.Ymat;
Ymat = Dataset.YmatNorm;

if(~exist('bNeg', 'var'))
    bNeg = false;
end

A = zeros(NClasses, NFilters * NClasses);
if(bNeg)
    A = -1/(NFilters * NClasses) * ones(NClasses, NFilters * NClasses);
end

for i = 1:size(A, 1)
    A(i,(i - 1) * NFilters + 1:(i * NFilters)) = 1; 
end
%%%%%%

W = convertFilter3Dto2DFormat(FilterVecs) * cost_scale;

fnMaxMinNorm = @(x) ( bsxfun(@rdivide, bsxfun(@minus,x, min(x)) , max(x) - min(x)) );
if(exist('fnCostSumNormalization', 'var'))
    fnMaxMinNorm = fnCostSumNormalization;
end

Cost = fnCostEval(X_flat, W, A);
CostSum = Cost * Ymat;
hCost = figure; imagesc(fnMaxMinNorm(CostSum))
hCostSurf = figure; surf(fnMaxMinNorm(CostSum)); shading interp

[~, EstLabel] = fnMinMax(Cost);
Error = EstLabel - y;

ECount = hist(abs(Error), NClasses);
CError = fliplr(cumsum(fliplr(ECount)));
CError = CError / max(CError(:));

result.Error = Error;
result.TotalError = sum(Error(:) ~= 0);
result.TEPercent = result.TotalError / numel(result.Error) * 100;
result.CError = CError;
result.EstLabel = EstLabel;
result.MeanEstLabel = EstLabel * Ymat;
PerExMeanEstLabel = bsxfun(@times, result.MeanEstLabel, YmatUnnorm);
PerExMeanEstLabel = sum(PerExMeanEstLabel, 2);
Diff = bsxfun(@minus, EstLabel, PerExMeanEstLabel');
result.StdEstLabel = sqrt(Diff.^2 * YmatUnnorm * diag(1./(sum(YmatUnnorm) - 1)));
result.RMSE = sqrt(Error.^2 * Ymat);
result.Cost = Cost;
result.CostSum = CostSum;
result.A = A;
result.y = y;
result.Ymat = Ymat;
result.hFigCost = hCost;
result.hFigCostSurf = hCostSurf;
result.fnMaxMinNorm = fnMaxMinNorm;