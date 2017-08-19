function [loss, dLdE, loss_per_ex] = HingeLoss(Energy, m, corr_scale, corr_label_ind, PerLabelLoss)
% WARNING: IF ENERGY IS ALLOWED TO BE NEGATIVE THEN corr_scale has to be 1.
% W is assumed to be NLabels x D and X is DxN (NTrainingExamples: To keep
% things composable this function computes loss for one example)
% corr_label corresponding to x
% HingeLoss per example: \sum_{j \neq corr_labe} max(0, scale * E(corr_label) -
% E(j) + m)

if(abs(corr_scale - 1) > 1e-16)
    assert(all(Energy(:) >= 0))
end

corr_energy = Energy(corr_label_ind);
margin = bsxfun(@minus, corr_scale * corr_energy + m, Energy);
margin(corr_label_ind) = 0;
loss_per_ex = max(0, margin) .* PerLabelLoss;
loss = sum(loss_per_ex(:));

% loss derivate wrt energy
dLdE = zeros(size(loss_per_ex)); % C x N
dLdE(margin > 0) = -1;
dLdE = dLdE .* PerLabelLoss;
NViolations = sum(-dLdE); % number of constraint violations for each example
dLdE(corr_label_ind) = corr_scale * NViolations;