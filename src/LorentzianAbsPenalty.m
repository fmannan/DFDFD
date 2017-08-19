function [loss, dLdE, loss_per_ex, Diff] = LorentzianAbsPenalty(Energy, alpha, E0, corr_label_ind)

Diff = Energy - E0;
loss_per_ex = 1./(1 + alpha * abs(Diff));
loss = sum(loss_per_ex(:));

dLdE = -alpha * sign(Diff) .* loss_per_ex.^2; % here sign(Energy) because of abs(Energy)
dLdE(corr_label_ind) = 0; % deriv for corr energies is 0
