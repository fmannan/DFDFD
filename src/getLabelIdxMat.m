function [Indices, ExampleLabelMat] = getLabelIdxMat(sz, label)

% returns index for correct label per column and a matrix with columns
% indicating groups that belong to the same class
% ExampleGroupMat is size NTrain x NClasses

Indices = sub2ind(sz, label, 1:sz(2));

YmatIdx = sub2ind([length(label), max(label)], 1:length(label), label);
Ymat = zeros(length(label), max(label));
Ymat(YmatIdx) = 1;

ExampleLabelMat = Ymat;
