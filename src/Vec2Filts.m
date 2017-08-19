function Filters = Vec2Filts(Vecs, K, NFilters, NPairs)
Ksqr = K * K;
Filters = cell(NFilters, NPairs);
for fIdx = 1:NFilters
    for pairIdx = 1:NPairs
        StartIdx = (pairIdx - 1) * Ksqr + 1;
        EndIdx = pairIdx * Ksqr;
        Filters{fIdx, pairIdx} = reshape(Vecs(StartIdx:EndIdx,fIdx), K, K);
    end
end
