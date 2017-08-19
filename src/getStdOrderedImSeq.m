function [im_seq_out, std_v_sorted, std_v] = getStdOrderedImSeq(im_seq, nTop)

R = size(im_seq, 1);
C = size(im_seq, 2);

if(ndims(im_seq) == 3)
    im_seq_v = reshape(im_seq, R * C, []);
else
    im_seq_v = im_seq;
end

std_v = std(im_seq_v);

[std_v_sorted, I] = sort(std_v, 2, 'descend');

im_seq_out_v = im_seq_v(:,I);
im_seq_out_v = im_seq_out_v(:,1:nTop);

if(ndims(im_seq) == 3)
    im_seq_out = reshape(im_seq_out_v, [R, C, nTop]);
else
    im_seq_out = im_seq_out_v;
end
