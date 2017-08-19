function [MeanDepth, StdDepth, DepthRMSE, MeanInvDepth, StdInvDepth, ...
         InvDepthRMSE] = DepthErrorStatAll(ErrorStat)
NDepths = length(ErrorStat)
  
    MeanDepth = nan(1, NDepths);
    StdDepth = nan(1, NDepths);
    DepthRMSE = nan(1, NDepths);
    MeanInvDepth = nan(1, NDepths);
    StdInvDepth = nan(1, NDepths);
    InvDepthRMSE = nan(1, NDepths);
    for idx = 1:NDepths
        DistStat = ErrorStat{idx};
        if(isfield(ErrorStat{idx}, 'DistStat'))
            DistStat = ErrorStat{idx}.DistStat;
        end
        if(isfield(DistStat, 'MeanDepth'))
            MeanDepth(idx) = DistStat.MeanDepth;
            StdDepth(idx) = DistStat.StdDepth;
            DepthRMSE(idx) = DistStat.DepthRMSE;
            MeanInvDepth(idx) = DistStat.MeanInvDepth;
            StdInvDepth(idx) = DistStat.StdInvDepth;
            InvDepthRMSE(idx) = DistStat.InvDepthRMSE;
        elseif(isfield(DistStat, 'DepthMean'))
            MeanDepth(idx) = DistStat.DepthMean;
            StdDepth(idx) = DistStat.DepthStd;
            DepthRMSE(idx) = DistStat.DepthRMSE;
            MeanInvDepth(idx) = DistStat.InvDepthMean;
            StdInvDepth(idx) = DistStat.InvDepthStd;
            InvDepthRMSE(idx) = DistStat.InvDepthRMSE;
        end
    end
