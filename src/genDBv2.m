% Based on genDB.m in code/cvpr16/
% a simpler version in that generate PSFs only based on the scale
% Generate DB of generated PSFs (Disk, Gaussian, Zhou, and corresponding rel blur and Gaussian approx)

% always use 2A configuration since 2F would be similar but symmetric (not
% entirely because the relative blur is non linear in 2F)
% But for now for simplicity use 2A config. This also lets us use disk
% along with other shapes.
% Maybe use 2F with focus inside the scene which is the most general and
% should work in all cases (need to flip asymmetric aperture)

close all
clear
clc

addpath ../
addpath ../DFD
addpath ../Zhou


bGenZhou = 1;
bGenDisk = 0;
NDepth = 9; %10; %5; %7; %11; %21; %19; %20; 
s0 = 1; %0.5; %-5; %-3; %1; %0.5; %0; %5; %0; %-3;
s1 = 5; %3; %10; %6; %15; %6; %3; 
sigR = 1; %2; %1.7;
r_px1 = linspace(s0, s1, NDepth); %[linspace(s0, -1, (NDepth-1)/2), 0, linspace(1, s1, (NDepth-1)/2)]; %
r_px2 = r_px1 * sigR;

%%
if(bGenZhou) % Zhou
%% Create DB using Zhou's aperture
PSFSetZhou = cell(NDepth, 2); % 
PSFSetZhouSingle = cell(NDepth, 1); % 
%%% Load Zhou's aperture
sz = 33;
K = im2double(imread('../Zhou/input/zhou.bmp'));
K = imresize(K, [sz, sz*2]);
inKernel1 = K(:, 1:end/2); 
inKernel2 = K(:, end/2+1:end);
inKernel1 = inKernel1 / sum(inKernel1(:));
inKernel2 = inKernel2 / sum(inKernel2(:));

Ksingle = im2double(imread('../Zhou/input/zhou_single.gif'));
Ksingle = imresize(Ksingle, [sz, sz]);
Ksingle = Ksingle / sum(Ksingle(:));

hei = 256;
wid = 256;
maxr = max([abs(r_px1(:)); abs(r_px2(:))])
maxksizeby2 = ceil(maxr * 1.1); % 10% bigger

for idx = 1:NDepth
   [~, psf1] = eScaleKernelOne(hei, wid, inKernel1, 2 * r_px1(idx), sum(inKernel1(:)));
   [~, psf2] = eScaleKernelOne(hei, wid, inKernel2, 2 * r_px2(idx), sum(inKernel2(:)));
   [~, psfsingle] = eScaleKernelOne(hei, wid, Ksingle, 2 * r_px1(idx), sum(Ksingle(:)));
   
   ksize = size(psf1);
   cidx = (ksize(1) + 1)/2 + [-maxksizeby2:maxksizeby2];
   
   PSFSetZhou{idx, 1} = psf1(cidx, cidx);
   PSFSetZhou{idx, 2} = psf2(cidx, cidx);
   PSFSetZhouSingle{idx} = psfsingle(cidx, cidx);
   
   figure;
   subplot(1,3,1); imagesc(PSFSetZhou{idx, 1}); axis image;
   subplot(1,3,2); imagesc(PSFSetZhou{idx, 2}); axis image
   subplot(1,3,3); imagesc(PSFSetZhouSingle{idx}); axis image
end
%%
KernelSet = PSFSetZhou;
scale = r_px1;
scale1 = r_px1;
scale2 = r_px2;
save(['PSFSet_Zhou_pair_s0_' num2str(s0) '_s1_' num2str(s1) '_sR' num2str(sigR) '_N' num2str(NDepth) '.mat'], ...
     'KernelSet', 's0', 's1', 'scale', 'scale1', 'scale2', 'NDepth');
KernelSet = PSFSetZhou(:,1);
scale = r_px1;
save(['PSFSet_Zhou_single1_s0_' num2str(s0) '_s1_' num2str(s1)  '_N' num2str(NDepth) '.mat'], 'KernelSet', 's0', 's1', 'scale', 'NDepth');
KernelSet = PSFSetZhou(:,2);
scale = r_px2;
save(['PSFSet_Zhou_single2_s0_' num2str(s0) '_s1_' num2str(s1)  '_N' num2str(NDepth) '.mat'], 'KernelSet', 's0', 's1', 'scale', 'NDepth');

KernelSet = PSFSetZhouSingle;
scale = r_px1;
save(['PSFSet_ZhouSingle_s0_' num2str(s0) '_s1_' num2str(s1)  '_N' num2str(NDepth) '.mat'], 'KernelSet', 's0', 's1', 'scale', 'NDepth');
end

%%
if(bGenDisk) % Disk
PSFSetDisk = cell(NDepth, 2); % 

pad_size = [4, 4];
pad_val = 0;

for idx = 1:NDepth
    
    PSFSetDisk{idx, 1} = PillboxKernel(r_px1(idx), pad_size, pad_val);
    PSFSetDisk{idx, 2} = PillboxKernel(r_px2(idx), pad_size, pad_val);
   
    figure;
    subplot(1,2,1); imagesc(PSFSetDisk{idx, 1}); axis image;
    subplot(1,2,2); imagesc(PSFSetDisk{idx, 2}); axis image
end
%%
KernelSet = PSFSetDisk;
scale = r_px1;
scale1 = r_px1;
scale2 = r_px2;
save(['PSFSet_Disk_pair_s0_' num2str(s0) '_s1_' num2str(s1) '_sR' num2str(sigR) '_N' num2str(NDepth) '.mat'], ...
     'KernelSet', 's0', 's1', 'scale', 'scale1', 'scale2', 'NDepth');
KernelSet = PSFSetDisk(:,1);
scale = r_px1;
save(['PSFSet_Disk_single1_s0_' num2str(s0) '_s1_' num2str(s1)  '_N' num2str(NDepth) '.mat'], 'KernelSet', 's0', 's1', 'scale', 'NDepth');
KernelSet = PSFSetDisk(:,2);
scale = r_px2;
s0 = r_px2(1);
s1 = r_px2(end);
save(['PSFSet_Disk_single1_s0_' num2str(r_px2(1)) '_s1_' num2str(r_px2(end))  '_N' num2str(NDepth) '.mat'], 'KernelSet', 's0', 's1', 'scale', 'NDepth');

end
