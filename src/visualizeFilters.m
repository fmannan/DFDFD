clear
close all
clc

addpath ../DFD/

PSFType = 'Disk' %'QPPSFGauss_2A_zFocus609.6_N1_22_N2_11'; %'QPPSFGauss'; %'Disk'; %'Zhou'; %
res = load('FullRes_n1_m1_cs1__0_1_0_1000000_0_0_0_0_0_1_genLogQuad_unnorm_im1_tp1_sn1_g1_K13_depth0_6_21_PSFSet_Disk_single1_s0_0_s1_6_fmingdadam.mat'); 
W = res.resultD.W;
NDepths = size(W, 3);
NFilters = size(W, 1);
NPairs = 1;
%%
K = 13;
ImSet = 1:NDepths; %[2, 9, 16, 25] %[2, 8, 22, 25];
if(NPairs == 1)
    Nplots = 1 + NFilters; 
    for idx = ImSet %1:size(res.PSFSet, 1) % 
        figure
        subplot(1, Nplots, 1); imagesc(res.KernelSet{idx}); axis image; 
        for idx2 = 1:NFilters
            subplot(1, Nplots, idx2 + 1); imagesc(reshape(W(idx2, :, idx), K, K)); axis image; 
        end
    end
end
%%
h = figure;
PSFSet = res.PSFSet;
NIter = size(res.PSFSet, 1)
ReconErr = nan(1, NIter);
bComputeReconErr = true;
if(isfield(res, 'ReconErr'))
    ReconErr = res.ReconErr;
    bComputeReconErr = false;
end

AnimFilename = [PSFType '_relblur'];
Frames(NIter) = struct('cdata',[],'colormap',[]);
for idx = 1:NIter
    [H1, H2] = PadEqualSize(PSFSet{idx, 1}, PSFSet{idx, 2} , 0);
    figure(h)
    colormap gray
    subplot(2, 4, 1); imagesc(H1); axis image;  title('h_1')
    subplot(2, 4, 2); imagesc(H2); axis image; title('h_2');
    if(res.PSFSGN(idx) == 1)
        subplot(2, 4, 3); imagesc(res.PSFSet{idx, 3}); axis image
        imsharp = H1;
        imblur = H2;
        HRel = res.PSFSet{idx, 3};
        imest = conv2(H1, HRel, 'same');
    else
        subplot(2, 4, 3); imagesc(res.PSFSet{idx, 4}); axis image
        imsharp = H2;
        imblur = H1;
        HRel = res.PSFSet{idx, 4};
        imest = conv2(H2, HRel, 'same');
    end
    title('rel blur')
    
    subplot(2, 4, 4); imagesc(imest); axis image; title('reconstruction');
    
    if(bComputeReconErr)
        ReconErr(idx) = ComputeRelBlurReconError(imsharp, imblur, HRel, 1, 0, 1);
    end
    
    subplot(2, 4, 5:8); plot(1000./res.Z(1:idx), ReconErr(1:idx), 'r-+')
    xlabel('Diopters')
    title(['Z = ' num2str(res.Z(idx)/1000) ' m']);
    drawnow
    Frames(idx) = getframe(h);
    im = frame2im(Frames(idx));
    [imind,cm] = rgb2ind(im,256);
    
    % On the first loop, create the file. In subsequent loops, append.
    if idx==1
        imwrite(imind,cm,[AnimFilename '.gif'],'gif','DelayTime',1,'loopcount',inf);
    else
        imwrite(imind,cm,[AnimFilename '.gif'],'gif','DelayTime',1,'writemode','append');
    end
 
    pause(1)
end
movie(Frames)
movie2avi(Frames, [AnimFilename '.avi'], 'fps', 1)
% figure(h)
% subplot(2, 4, 5:8); plot(res.Z, res.ReconErr, 'r-+')
%%
figure
plot(ReconErr)

%%
outDir = ['./relblur_' PSFType '/']; %'/usr/local/data/fmannan/thesis/thesisdoc/DFD_cvpr_2016/suppl_figs/'
if(~exist(outDir, 'dir'))
    mkdir(outDir)
end

for idx = ImSet
    if(res.PSFSGN(idx) < 0)
        sIdx = 2;
        bIdx = 1;
        rIdx = 4;
    else
        sIdx = 1;
        bIdx = 2;
        rIdx = 3;
    end
    imsharp = res.PSFSet{idx, sIdx};
    imblur = res.PSFSet{idx, bIdx};
    Hrel = res.PSFSet{idx, rIdx};
        
    imest = conv2(imsharp, Hrel, 'same');
    suffix_str = num2str(idx);
    
    imwrite(uint8(255 * normalizeImage(imsharp)), [outDir PSFType '_sharp_' suffix_str '.png'])
    imwrite(uint8(255 * normalizeImage(imblur)), [outDir PSFType '_blur_' suffix_str '.png'])
    imwrite(uint8(255 * normalizeImage(Hrel)), [outDir PSFType '_relblur_' suffix_str '.png'])
    imwrite(uint8(255 * normalizeImage(imest)), [outDir PSFType '_estblur_' suffix_str '.png'])
end