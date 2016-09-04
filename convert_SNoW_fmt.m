% Part-based Drone detector 
% PartE: Convert to SNoW readable format

% Author: Bicheng Zhang
% University of Illinois Urbana Champaign
% email: zhang368@illinois.edu


% Arguments
FileName = 'drone-training.txt';

% Initialization
load image-representation;

numPositiveSamples = size(posImgFeatureCell,1);
numNegativeSamples = size(negImgFeatureCell,1);

if exist(FileName, 'file')
    delete(FileName);
end

f = fopen(FileName, 'wt');



% Start converting positive training samples
for i=1:numPositiveSamples
    featureArray = posImgFeatureCell{i};
    featureArray = featureArray{1};
    output_line = '1';

    %disp(vocabArray);
    %disp(length(vocabArray));
    for fi = 1:length(featureArray)
        output_line = sprintf('%s,%d',output_line,featureArray(fi));  
    end
    
    output_line = sprintf('%s:\n',output_line);
    
    count = fwrite(f,output_line);
    assert(count == length(output_line));
    %dlmwrite(OutPutFileName, vocabArray, '-append','delimiter', ',', 'newline', 'unix');
    
end


% Start converting negative training samples
for i=1:numNegativeSamples
    featureArray = negImgFeatureCell{i};
    featureArray = featureArray{1};
    output_line = '0';

    %disp(vocabArray);
    
    for fi = 1:length(featureArray)
        output_line = sprintf('%s,%d',output_line,featureArray(fi));  
    end
    
    output_line = sprintf('%s:\n',output_line);
    
    count = fwrite(f,output_line);
    assert(count == length(output_line));
    %dlmwrite(OutPutFileName, vocabArray, '-append','delimiter', ',', 'newline', 'unix');
    
end

fclose(f);




