function [loss, dLdE, dLdm, loss_per_ex, MarginLoss, dLdm_1, dLdm_tmp] = HingeLossMarginOpt(Energy, m, alpha, m0, corr_scale, corr_label_ind, label_per_ex, PerLabelLoss)
% WARNING: IF ENERGY IS ALLOWED TO BE NEGATIVE THEN corr_scale has to be 1.
% W is assumed to be NLabels x D and X is DxN (NTrainingExamples: To keep
% things composable this function computes loss for one example)
% corr_label corresponding to x
% HingeLoss per example: \sum_{j \neq corr_labe} max(0, scale * E(corr_label) -
% E(j) + m)
%%%%%%
if(abs(corr_scale - 1) > 1e-16)
    assert(all(Energy(:) >= 0))
end

corr_energy = Energy(corr_label_ind);
margin = bsxfun(@minus, corr_scale * corr_energy + reshape(m(label_per_ex), 1, []), Energy);
margin(corr_label_ind) = 0;
loss_per_ex = max(0, margin) .* PerLabelLoss;
loss = sum(loss_per_ex(:));

% loss derivate wrt energy
dLdE = zeros(size(loss_per_ex)); % C x N
dLdE(margin > 0) = -1;
dLdE = dLdE .* PerLabelLoss;
NViolations = sum(-dLdE); % number of constraint violations for each example
dLdE(corr_label_ind) = corr_scale * NViolations;
%%%%%%
MarginLoss = max(0, alpha - log(m - abs(m0))); % to ensure that margin is always greater than a certain value
loss = loss + sum(MarginLoss);
dLdm_1 = - 1./(m - abs(m0));
dLdm_1(MarginLoss <= 0) = 0;
dLdm_tmp = zeros(size(Energy));
dLdm_tmp(corr_label_ind) = dLdE(corr_label_ind) / corr_scale;
dLdm_2 = sum(dLdm_tmp, 2);
dLdm = dLdm_1(:) + dLdm_2(:);
