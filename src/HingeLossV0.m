function [cost, G, H] = HingeLossV0(x, W, m, corr_scale, corr_label, fnEnergy)

% W is assumed to be MxDxNLabels and X is Dx1 (NTrainingExamples: To keep
% things composable this function computes loss for one example)
% corr_label corresponding to x
% HingeLoss per example: \sum_{j \neq corr_labe} max(0, E(corr_label) -
% E(j) + m)
%size(W(:,:,corr_label))
%corr_label
[Ecorr, dEcorr, d2Ecorr] = fnEnergy(x, W(:,:,corr_label));
cost = 0;
G = zeros(size(W));
H = zeros(size(W,2), size(W,2), size(W,3));
% TODO: VECTORIZE (HOW TO DO WITH fnEnergy being a function!!?)
for i = 1:size(W, 3)
   if(i ~= corr_label)
      [E, dE, d2E] = fnEnergy(x, W(:,:,i));
      A = corr_scale * Ecorr - E + m;
      if(A > 0)
         cost = cost + A;
         G(:,:,corr_label) = G(:,:,corr_label) + corr_scale * dEcorr;
         G(:,:,i) = G(:,:,i) - dE;
         H(:,:,corr_label) = H(:,:,corr_label) + d2Ecorr;
         H(:,:,i) = H(:,:,i) - d2E;
      end
   end
end
