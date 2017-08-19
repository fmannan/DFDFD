function params_out = BuildInputParams(Filters, params)

% build parameters for evaluating filters

[NFilters, FilterGroupSize, NClasses] = size(Filters);

A = zeros(NClasses, NFilters * NClasses);
for i = 1:size(A, 1)
    A(i,(i - 1) * NFilters + 1:(i * NFilters)) = 1; 
end
%%%%%%
if(iscell(Filters))
[KR, KC] = size(Filters{1, 1, 1});
FilterSize = KR * KC;
FilterVecs = nan(NFilters, FilterGroupSize * FilterSize, NClasses);
for i = 1:NClasses
   for j = 1:NFilters
      for k = 1:FilterGroupSize
          StartIdx = (k - 1) * FilterSize + 1;
          EndIdx = k * FilterSize;
          FilterVecs(j, StartIdx:EndIdx, i) =  reshape(Filters{j, k, i}, 1, []);
      end
   end
end
else
    FilterVecs = Filters;
end
W = convertFilter3Dto2DFormat(FilterVecs);

params_out.Filters = Filters;
params_out.FilterVecs = FilterVecs;
params_out.NClasses = NClasses;
params_out.NFilters = NFilters;
params_out.FilterGroupSize = FilterGroupSize;
params_out.A = A;
params_out.W = W;
%params_out.KR = KR;
%params_out.KC = KC;

params_out.fnCost = params.fnCost;
params_out.fnCombine = params.fnCombine;
params_out.fnObj = params.fnObj;
if(isfield(params, 'fnPatchNormalization'))
    params_out.fnPatchNormalization = params.fnPatchNormalization;
end
