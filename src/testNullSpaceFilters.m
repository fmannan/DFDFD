addpath ../
addpath ../DFD

close all
clear
clc

TruncatePercentage = 1; %.25;

K = 21; %13;
NFilters = 10;
Stride = 10;

im0 = im2double(imread('../DFD/brown_noise_pattern_512x512.png'));
s1 = 2; %1; %2; %5; %.5;
s2 = 8; %5; %6; %5; %8; %1.5;
fnPSF = @(x) PillboxKernel(x, [3, 3], 0); %@(x) GaussianKernel(x); % 
PSF1 = fnPSF(s1); %PillboxKernel(0, [3, 3], 0); % 2
PSF2 = fnPSF(s2); %PillboxKernel(5, [3, 3], 0);

figure;
subplot(1, 2, 1); imagesc(PSF1); axis image
subplot(1, 2, 2); imagesc(PSF2); axis image

im1 = blurImg(im0, PSF1, 'PSF', 'symmetric');
im2 = blurImg(im0, PSF2, 'PSF', 'symmetric');
figure
imshow([im1, zeros(size(im1, 1), 3), im2])

im1seq = PartitionImage(im1, [K, K], Stride);
im2seq = PartitionImage(im2, [K, K], Stride);

A = [reshape(im1seq, [], size(im1seq, 3)); 
     reshape(im2seq, [], size(im2seq, 3))];
A1 = A; 
W = findNullSpaceFiltersPerDepth(A, NFilters, TruncatePercentage);
R1 = W' * A1;
sqrt(sum(R1.^2, 2))

Filters = VecToFilterGroup(W, K);
% for idx = 1:size(Filters, 1)
%    figure
%    subplot(1, 2, 1); imagesc(Filters{idx, 1}); axis image
%    subplot(1, 2, 2); imagesc(Filters{idx, 2}); axis image
% end

im1seqN = bsxfun(@minus, im1seq, mean(mean(im1seq, 1), 2));
im2seqN = bsxfun(@minus, im2seq, mean(mean(im2seq, 1), 2));

A = [reshape(im1seqN, [], size(im1seqN, 3)); 
     reshape(im2seqN, [], size(im2seqN, 3))];

FiltersN = VecToFilterGroup(findNullSpaceFiltersPerDepth(A, NFilters, TruncatePercentage), K);

%% var normalization
im1seqNv = bsxfun(@rdivide, im1seqN, std(reshape(im1seqN, 1, [], size(im1seqN, 3))));
im2seqNv = bsxfun(@rdivide, im2seqN, std(reshape(im2seqN, 1, [], size(im2seqN, 3))));

A = [reshape(im1seqNv, [], size(im1seqNv, 3)); 
     reshape(im2seqNv, [], size(im2seqNv, 3))];
 
FiltersNv = VecToFilterGroup(findNullSpaceFiltersPerDepth(A, NFilters, TruncatePercentage), K);
%%
[K1, K2] = PadEqualSize(PSF1, PSF2, 0);
AA1 = im2convmtx(K1, K, K);
AA2 = im2convmtx(K2, K, K);
A = [AA1'; AA2'];
WPSF = findNullSpaceFiltersPerDepth(A, NFilters, TruncatePercentage);
FiltersPSF = VecToFilterGroup(WPSF, K);

A4 = A; 
R4 = WPSF' * A4;
sqrt(sum(R4.^2, 2))

R4_1 = WPSF' * A1;
sqrt(sum(R4_1.^2, 2))

R1_4 = W' * A4;
sqrt(sum(R1_4.^2, 2))
%%
A = [A1, A4];
W5 = findNullSpaceFiltersPerDepth(A, NFilters, TruncatePercentage);
Filters5 = VecToFilterGroup(W5, K);

A5 = A; 
R5 = W5' * A5;
sqrt(sum(R5.^2, 2))
%%
R5_4 = W5' * A4;
sqrt(sum(R5_4.^2, 2))

R5_1 = W5' * A1;
sqrt(sum(R5_1.^2, 2))
%%
S = 2 * size(K1) - 1;
A1 = psf2convmtx(K1, S(1), S(2), 'valid');
A2 = psf2convmtx(K2, S(1), S(2), 'valid');

A = [A1;A2];
[U1, S1, V1] = svd(A1, 0);
WPSF1 = findNullSpaceFiltersPerDepth(A, NFilters, TruncatePercentage);
FiltersPSF1 = VecToFilterGroup(WPSF1, size(K1, 1));
%%
[U2, S2, V2] = svd(A2, 0);
im2seq1 = PartitionImage(im2, size(K1), Stride);
im2seq1_v = reshape(im2seq1, [], size(im2seq1, 3));
%%
for idx = 1:size(FiltersN, 1)
   figure
   colormap gray
   subplot(6, 2, 1); imagesc(Filters{idx, 1}); axis image
   subplot(6, 2, 2); imagesc(Filters{idx, 2}); axis image
   subplot(6, 2, 3); imagesc(FiltersN{idx, 1}); axis image
   subplot(6, 2, 4); imagesc(FiltersN{idx, 2}); axis image
   subplot(6, 2, 5); imagesc(FiltersNv{idx, 1}); axis image
   subplot(6, 2, 6); imagesc(FiltersNv{idx, 2}); axis image
   subplot(6, 2, 7); imagesc(FiltersPSF{idx, 1}); axis image
   subplot(6, 2, 8); imagesc(FiltersPSF{idx, 2}); axis image
   subplot(6, 2, 9); imagesc(Filters5{idx, 1}); axis image
   subplot(6, 2, 10); imagesc(Filters5{idx, 2}); axis image
   subplot(6, 2, 11); imagesc(FiltersPSF1{idx, 1}); axis image
   subplot(6, 2, 12); imagesc(FiltersPSF1{idx, 2}); axis image
end


