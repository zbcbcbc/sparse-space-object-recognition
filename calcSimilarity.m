function  sim  = calcSimilarity( A, B, method)
%SIMILARITY Summary of this function goes here
%   Detailed explanation goes here

    A = double(A);
    B = double(B);

    if strcmp(method,'normalized_correlation')
        %TODO: calculate similarity between clusters
        sim = mean(mean(A.*B))/sqrt(mean(mean(A.^2))*mean(mean(B.^2)));
        %fprintf('meanAB:%.2f\n',mean(mean(A.*B)));
        %fprintf('meanAA:%.2f\n',mean(mean(A.^2)));
        %fprintf('meanBB:%.2f\n',mean(mean(B.^2)));
        %fprintf('sqrt:%.2f\n',sqrt(mean(mean(A.^2))*mean(mean(B.^2))))
	elseif strcmp(method, 'normalized_euclidean')

    end
end

