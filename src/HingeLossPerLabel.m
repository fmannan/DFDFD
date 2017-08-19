function [loss, dLdE, loss_per_label, dLdE_C, EnergyAll] = HingeLossPerLabel(Energy, m, corr_scale, ExLabelMat)
% WARNING: IF ENERGY IS ALLOWED TO BE NEGATIVE THEN corr_scale has to be 1.
% W is assumed to be NLabels x D and X is DxN (NTrainingExamples: To keep
% things composable this function computes loss for one example)
% corr_label corresponding to x
% HingeLoss per label: \sum_{j \neq corr_ex_class} max(0, scale * E(corr_example) -
% E(j) + m)
% y : label for each example (i.e. each column of Energy)

if(abs(corr_scale - 1) > 1e-16)
    assert(all(Energy(:) >= 0))
end

NClasses = size(ExLabelMat, 2);
ExLabelMat = ExLabelMat * diag(1./sum(ExLabelMat)); % normalize
EnergyAll = Energy * ExLabelMat;
corr_energy = diag(EnergyAll);
DiagInd = sub2ind([NClasses, NClasses], 1:NClasses, 1:NClasses);
margin = bsxfun(@minus, corr_scale * corr_energy + m, EnergyAll);
margin(DiagInd) = 0; % margin = margin - diag(diag(margin)); %set margin of correct entries to 0
loss_per_label = max(0, margin);
loss = sum(loss_per_label(:));

% loss derivate wrt energy
dLdE_C = zeros(size(loss_per_label)); % C x N
dLdE_C(margin > 0) = -1;
NViolations = sum(-dLdE_C, 2); % number of constraint violations for each label
dLdE_C(DiagInd) = corr_scale * NViolations;

% make dLdE the size of energy
dLdE = (ExLabelMat * diag(sum(dLdE_C)))';
