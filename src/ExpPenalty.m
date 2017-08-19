function [loss, dLdE, loss_per_ex, Diff] = ExpPenalty(Energy, alpha, E0, corr_label_ind, bApplyToCorrLabels)

Diff = bsxfun(@minus, Energy, E0);
loss_per_ex = exp(-alpha * Diff);
if(bApplyToCorrLabels)
    tmp = zeros(size(loss_per_ex));
    tmp(corr_label_ind) = loss_per_ex(corr_label_ind); % only consider entries corresponding to the correct label
    loss_per_ex = tmp;
    
    loss = sum(loss_per_ex(:));
    
    dLdE = -alpha * loss_per_ex;
else
    loss = sum(loss_per_ex(:));

    dLdE = -alpha * loss_per_ex; % here sign(Energy) because of abs(Energy)
    dLdE(corr_label_ind) = 0; % deriv for corr energies is 0
end