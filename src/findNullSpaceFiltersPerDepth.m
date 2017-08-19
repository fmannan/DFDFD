function [FilterVecs, retVal] = findNullSpaceFiltersPerDepth(A, NFilters, TruncatePercentage)

[FilterVecs, retVal] = findLeftNullSpaceFiltersPerDepth(A * A', NFilters, TruncatePercentage);

