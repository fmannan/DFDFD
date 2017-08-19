% evaluate real coded aperture result
close all
clear
clc

addpath ../Levin_filts/images/

ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth1_9_9_PSFSet_Levin_N9_fmingdadam.mat';
ResultFilename = 'FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K15_depth1_9_9_PSFSet_Levin_N9_fmingdadam_run2.mat';
ResData = load(ResultFilename);
if(~isfield(ResData.params, 'NPairs'))
    ResData.params.NPairs = 1;
end
if(isfield(ResData.resultD, 'Filters'))
    Filters = ResData.resultD.Filters;
else
    Filters = ResData.resultD.W;
end
params = ResData.params;
ResParam = BuildInputParams(Filters, ResData.params)

if(isfield(ResData, 'KernelSetDisk'))
    KernelSet = ResData.KernelSetDisk;
else
    KernelSet = ResData.KernelSet;
end

Im_seq = ResData.Im_seq;
%%
%im1 = imread('people_inp.bmp');
%im1g = im2double(im1(:,:,2));
K = params.KR;
H = fspecial('gaussian', 5, .8);
for gtIdx = 1:9
    im1g = conv2(Im_seq{2}, KernelSet{gtIdx});
    NDepth = size(Filters, 3);

    Cost = zeros(size(im1g, 1), size(im1g, 2), NDepth);
    
    for idx = 1:9
        for filterIdx = 1:size(Filters,1)
            F = rot90(reshape(Filters(filterIdx, :, idx), K, K), 2);
            f = conv2(im1g, F, 'same');
            Cost(:,:,idx) = Cost(:,:,idx) + f.^2;
        end
        Cost(:,:,idx) = conv2(log(Cost(:,:,idx)), H, 'same');
    end
    %Cost = log(Cost);
    %
    [Y, I] = min(Cost, [], 3);
    display(['GT : ' num2str(gtIdx) ', mean : ' num2str(mean(I(:))) ' , median : ' num2str(median(I(:)))])
end

%%
im1 = imread('beer_coke_inp.bmp'); %imread('people_inp.bmp');
im1g = im2double(im1(:,:,2));
NDepth = size(Filters, 3);

Cost = zeros(size(im1g, 1), size(im1g, 2), NDepth);
K = params.KR;
H = fspecial('gaussian', 5, .8);
for idx = 1:9
    for filterIdx = 1:size(Filters,1)
        F = rot90(reshape(Filters(filterIdx, :, idx), K, K), 2);
        f = conv2(im1g, F, 'same');
        Cost(:,:,idx) = Cost(:,:,idx) + f.^2;
    end
    Cost(:,:,idx) = conv2(log(Cost(:,:,idx)), H, 'same');
end
%Cost = log(Cost);
%
[Y, I] = min(Cost, [], 3);
figure
imagesc(I)
