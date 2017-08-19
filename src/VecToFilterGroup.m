function Filters = VecToFilterGroup(W, K)

[R, C] = size(W);
NPairs = R / (K * K);

Filters = cell(C, NPairs);

for i = 1:C
    for j = 1:NPairs
        Start = (j - 1) * K * K + 1;
        End = Start - 1 + K * K;
        Filters{i, j} = reshape(W(Start:End, i), [K, K]);
    end
end
      