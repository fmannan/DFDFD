function res = genDataMatrix(ImageDepthSet, fnLabelLoss)

DataDim = size(ImageDepthSet{1}, 1);
NClasses = size(ImageDepthSet, 2);

%X = reshape(cell2mat(ImageDepthSet), [DataDim, ExamplesPerClass, NClasses]);
X_flat = cell2mat(ImageDepthSet);
NTrain = size(X_flat, 2);
y = []; %nan(1, size(X_flat, 2));
for idx = 1:NClasses
    y = [y, ones(1, size(ImageDepthSet{idx}, 2)) * idx];
end

YmatIdx = sub2ind([length(y), max(y)], 1:length(y), y);
Ymat = zeros(length(y), max(y));
Ymat(YmatIdx) = 1;
YmatNorm = Ymat * diag(1./sum(Ymat)); % normalize

res.PerLabelLoss = fnLabelLoss(bsxfun(@minus, ndgrid(1:NClasses, 1:NTrain), y));

res.X_flat = X_flat;
res.DataDim = DataDim;
res.NClasses = NClasses;
res.NTrain = NTrain;
res.y = y;
res.Ymat = Ymat;
res.YmatNorm = YmatNorm;
