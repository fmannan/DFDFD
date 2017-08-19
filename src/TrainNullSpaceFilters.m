function result = TrainNullSpaceFilters(ImageDepthSet, params) %(Im_seq, KernelSet, params)
% find a bank of null space filters. Bank size speficied by the rank of the
% matrix. This is essentially Favaro and Soatto's orthogonal operator
% approach.
% code is based on testNullSpaceFilters.m
% uses findNullSpaceFilterPairsPerDepth for finding filters per depth
% this is different from our formulation in that the filters found at each
% depth do not know about other depths and so there may be overlap and they
% may respond with lower energy for other depths.


%res = BuildTrainingSet(Im_seq, KernelSet, params);
%ImageDepthSet = res.ImageDepthSet;

TruncatePercentage = 1; % no truncation
if(isfield(params, 'TruncatePercentage'))
    TruncatePercentage = params.TruncatePercentage;
end
fnNullSpace = @findLeftNullSpaceFiltersPerDepth;
if(isfield(params, 'fnNullSpace'))
    fnNullSpace = params.fnNullSpace;
end
display(['Truncate Percentage: ' num2str(TruncatePercentage)]);

NFilters = 5;
if(isfield(params, 'NFilters'))
    NFilters = params.NFilters;
end
NDepth = length(ImageDepthSet);
%NPairs = params.NPairs; %size(ImageDepthSet{1}, 1) / (params.KR * params.KC);
DataDim = size(ImageDepthSet{1}, 1);
%Filters = cell(NFilters, NPairs, NDepth);
FilterVecs = nan(NFilters, DataDim, NDepth);
retVals = cell(1, NDepth);
for i = 1:NDepth
   [~, retVals{i}] = fnNullSpace(ImageDepthSet{i}, NFilters, TruncatePercentage);
   FilterVecs(:,:,i) = retVals{i}.Vecs';
   %for idx = 1:NFilters
   %    for idxP = 1:NPairs
   %        Filters{idx,idxP,i}  = F{idx, idxP};
   %    end
   %end
end

result.FilterVecs = FilterVecs;
%result.Filters = Filters;
% by default do not store return values
if(isfield(params, 'bStoreRetVals') && params.bStoreRetVals) 
    result.retVals = retVals;
end
