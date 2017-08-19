function [loss, dLdE, corr_energy] = LowEnergyLoss(Energy, corr_label_ind)

corr_energy = Energy(corr_label_ind);
loss = sum(corr_energy);

dLdE = zeros(size(Energy));
dLdE(corr_label_ind) = 1;
