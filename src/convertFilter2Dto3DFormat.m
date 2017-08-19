function W3D = convertFilter2Dto3DFormat(W2D, NFilters, NClasses)
% 3D format is NFilters x DataDim x NClasses
% 2D format is (NFilters x NClasses) x DataDim
size(W2D)
NFilters
NClasses
[FC, DataDim] = size(W2D);
assert(FC == NFilters * NClasses)

W3D = permute(reshape(W2D', [DataDim, NFilters, NClasses]), [2, 1, 3]);
