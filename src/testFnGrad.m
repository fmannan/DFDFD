E = [9.5717    1.4189    7.9221    0.3571    6.7874;
    4.8538    4.2176    9.5949    8.4913    7.4313;
    8.0028    9.1574    6.5574    9.3399    7.5774;
    9.1338    5.4688    9.7059    1.4189    9.5949];
y = [2, 4, 1, 2, 3]
[corr_label_ind, ExLabelMat] = getLabelIdxMat(size(E), y)
m = 1;
corr_scale = 1;
[Hloss, dHLdE, loss_per_ex] = HingeLoss(E, m, corr_scale, corr_label_ind)
assert(sum(dHLdE(:)) == 0)
[MHloss, dMHLdE, max_loss_per_ex, margin] = MaxHingeLoss(E, m, corr_scale, corr_label_ind)
assert(sum(dMHLdE(:)) == 0)

%%
[HlossPerLabel, dLdE, loss_per_label, dLdE_C, EAll] = HingeLossPerLabel(E, m, corr_scale, ExLabelMat)

%%
[l, dlde, dldm, lex, ml, dldm1, dldm_tmp] = HingeLossMarginOpt(E, [1, 10, 1e4, .9], 10, 1e-3, 1, corr_label_ind, y)