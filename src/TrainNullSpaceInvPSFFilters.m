function result = TrainNullSpaceInvPSFFilters(Im_seq, KernelSet, params)
% code based on TrainNullSpaceFilters and BuildTrainingSet
% inverts the PSFs and premults with null space of sharp images.
% code is based on testNullSpaceFilters.m
% uses findNullSpaceFilterPairsPerDepth for finding filters per depth
% this is different from our formulation in that the filters found at each
% depth do not know about other depths and so there may be overlap and they
% may respond with lower energy for other depths.

% build training set of sharp images
DeltaKernel = {1};
ImageDepthSet = BuildTrainingSet(Im_seq, DeltaKernel, params);

% get the null space of sharp images
resNullSpace = TrainNullSpaceFilters(ImageDepthSet, params);

% construct inv psf filters
resInvK = InvPSF(KernelSet, params);

% construct final filters by premultiplying with the nullspace of sharp
% images
NDepth = size(KernelSet, 1);
NPairs = size(KernelSet, 2);

FilterVecsNS = resNullSpace.FilterVecs;
Dim = size(FilterVecsNS, 2);
NFilters = size(FilterVecsNS, 1);
FilterVecs = zeros(NFilters  * NPairs, Dim  * NPairs, NDepth); % NOTE: In the case of inverse we stack one on top of the other
KDim = size(resInvK.InvKernel{1,1}, 1); % assuming square dim
FilterVecsInvPSF = zeros(KDim * NPairs, KDim * NPairs, NDepth); % Deconv recon error 
for idx1 = 1:NDepth
    for idx2 = 1:NPairs
        Start = (idx2 - 1) * NFilters + 1;
        End = idx2 * NFilters;
        
        Start2 = (idx2 - 1) * Dim + 1;
        End2 = idx2 * Dim;
        FilterVecs(Start:End,Start2:End2,idx1) = FilterVecsNS * resInvK.InvKernel{idx1, idx2};
        
        Start3 = (idx2 - 1) * KDim + 1;
        End3 = idx2 * KDim;
        FilterVecsInvPSF(Start3:End3, Start3:End3,idx1) =  resInvK.Kernel{idx1, idx2} * resInvK.InvKernel{idx1, idx2} - eye(size(resInvK.Kernel{idx1, idx2}));
    end
end
result.FilterVecs = FilterVecs;
result.FilterVecsNS = FilterVecsNS;
result.FilterVecsInvPSF = FilterVecsInvPSF;
result.InvKernel = resInvK.InvKernel;
