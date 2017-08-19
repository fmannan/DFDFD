function [loss, dMHLdE, max_loss_per_ex, margin] = MaxHingeLoss(Energy, m, corr_scale, corr_label_ind, PerLabelLoss, margin)
% WARNING: IF ENERGY IS ALLOWED TO BE NEGATIVE THEN corr_scale has to be 1.
% W is assumed to be NLabels x D and X is DxN (NTrainingExamples: To keep
% things composable this function computes loss for one example)
% corr_label corresponding to x
% MaxHingeLoss per example: \max_{j \neq corr_labe} max(0, scale * E(corr_label) -
% E(j) + m)

if(abs(corr_scale - 1) > 1e-16)
    assert(all(Energy(:) >= 0))
end

[NClasses, NTrain] = size(Energy);
if(~exist('margin', 'var'))
    corr_energy = Energy(corr_label_ind);
    margin = bsxfun(@minus, corr_scale * corr_energy + m, Energy);
    margin(corr_label_ind) = 0;
    margin = max(0, margin) .* PerLabelLoss;
end
[max_loss_per_ex, I_max_loss] = max(margin); % max hinge loss. NOTE: if an entire column is 0 then I_max_loss may have the wrong index
loss = sum(max_loss_per_ex(:));

% max hinge loss derivate wrt energy
dMHLdE = zeros(NClasses, NTrain);
IndMHL = sub2ind([NClasses, NTrain], I_max_loss, 1:NTrain);
dMHLdE(IndMHL) =  -1;
dMHLdE(margin == 0) = 0; % to correct for the case when max loss in a column is 0
dMHLdE = dMHLdE .* PerLabelLoss;
NViolationsMHL = sum(-dMHLdE);
%assert(all(max(NViolationsMHL) <= 1));
dMHLdE(corr_label_ind) = corr_scale * NViolationsMHL;
