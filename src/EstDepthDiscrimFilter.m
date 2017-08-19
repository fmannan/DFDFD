function res = EstDepthDiscrimFilter(im1, im2, params)
% estimate depth at every pixel from discriminative filters

K = params.KernelSize;

im1seq = PartVecNormImage(im1, K, params.fnNormalize);
im2seq = PartVecNormImage(im2, K, params.fnNormalize);

imseq = params.fnCombine(im1seq, im2seq);

% preprocess the filters so that the energy function can work with it
W = params.W;
A = params.A;
Cost = params.fnEnergy(imseq, W, A);

[~, LabelIdx] = min(Cost);

LabelIdx = reshape(LabelIdx, size(im1));
res.LabelIdx = LabelIdx;
if(isfield(params, 'Z'))
    res.Z = params.Z(LabelIdx);
end

end

function im_seq_norm = PartVecNormImage(im, K, fnNormalize)

padsize = (K - 1)/2;
impad = padarray(im, padsize * ones(1, 2), 'symmetric');
im_seq = im2col(impad, [K, K], 'sliding');

im_seq_norm = fnNormalize(im_seq);

end