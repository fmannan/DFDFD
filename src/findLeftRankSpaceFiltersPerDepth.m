function [Vecs, retVal] = findLeftRankSpaceFiltersPerDepth(A, NFilters, TruncatePercentage)

if(~exist('NFilters', 'var'))
    NFilters = 1;
end

if(~exist('TruncatePercentage', 'var'))
    TruncatePercentage = 1;
end

[U, S, V] = svd(A);

ktrunc = ceil(min(size(S)) * TruncatePercentage);
Utruncated = U(:,1:ktrunc);
Vecs = Utruncated(:,1:NFilters);

Whitening = U(:,1:ktrunc) * diag(1./diag(S(1:ktrunc, 1:ktrunc))) * U(:,1:ktrunc)';

retVal.Vecs = Vecs;
retVal.U = U;
retVal.S = S;
retVal.V = V;
retVal.Whitening = Whitening;
retVal.Utruncated = Utruncated;
retVal.ktrunc = ktrunc;

