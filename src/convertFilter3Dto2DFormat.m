function W2D = convertFilter3Dto2DFormat(W3D)
% 3D format is NFilters x DataDim x NClasses
% 2D format is (NFilters x NClasses) x DataDim

%Wmat = reshape(Wvec, NFilters, DataDim, NClasses);
[NFilters, DataDim, NClasses] = size(W3D);
W2D = reshape(permute(W3D, [2, 1, 3]), [DataDim, NFilters * NClasses])'; % W is now (F x C) x D
