% Part-based Drone detector 
% PartA: Construct Vocabulary

% Author: Bicheng Zhang
% University of Illinois Urbana Champaign
% email: zhang368@illinois.edu


%% Arguments
ImgDir = 'dataset/construct/';
ImgHeight = 50;
ImgWidth = 160;
ImgDim = 3;
InterestPointOperator = 'harris'; % harris, foerstner
HarrisCoe.threshold = 5000;
HarrisCoe.sigma = 2.1;
HarrisCoe.radius = 5;

FoerCoe.sigma_deriv = 0.7;
FoerCoe.sigma_n = 1;
FoerCoe.sigma_int = 2;
FoerCoe.precision_thresh = 0.97; % default: 0.5
FoerCoe.round_thresh = 0.97; % default: 0.3
FoerCoe.significance_level = 0.999; % default: 0.999
FoerCoe.visualize = 'on';

SimilarityMethod = 'normalized_correlation'; % normalized_correlation, normalized_euclidean
SimilarityThreshold = 0.88; % increase threshold, less clusters will be merged
PatchSize = 15; % 15*15 pixels
TotalNumPatches = 600;
ClusteringMethod = 'bottom-up';

%% Options for Speed & Optimization
UseDP = true;
HistEqual = true;
DispResults = true;
ExpressMode = true;



%% Initialize Parameters
imgs = dir([ImgDir, '*.jpg']);
numImgs = size(imgs,1);
total_feature_points = 0;
imgPool = cell(numImgs,1);
patch_r = floor(PatchSize/2);
patchPool = cell(TotalNumPatches,1);
clusterPool = cell(TotalNumPatches,1);
curNumPatches = 0;


%% A1 Read 100 representative images of Drone, and convert to gray scale
tic;
for i=1:numImgs
    img = imread([ImgDir imgs(i).name]);
    [h,w,d] = size(img);
    if h~= ImgHeight || w~= ImgWidth || d~= ImgDim
        error('%s has wrong dimension w:%d,h:%d,d:%d\n', imgs(i).name, w,h,d);
    else
        if HistEqual == true
            imgPool{i} = histeq(rgb2gray(img));
        else
            imgPool{i} = rgb2gray(img);
        end
    end
end
fprintf('A1 complete[%f s]-- Read %d traning images[%d %d %d].\n', toc, numImgs, ImgWidth,ImgHeight,ImgDim);

%% A2 Apply Interest Point Operator
tic;
for cur_img_idx=1:numImgs
   img = imgPool{cur_img_idx};
   if strcmp(InterestPointOperator, 'harris')
       %TODO: add timing mechanism
       [fcoors,cim] = harris(img, HarrisCoe, DispResults);
       %waitforbuttonpress;
   elseif strcmp(InterestPointOperator, 'foerstner')

        [fcoors] = foerstner(img, FoerCoe);
        waitforbuttonpress;
   else
       error('%s is not a recognized interest point operator.\n', InterestPointOperator);
   end
   
   
   %% A3 Extract patches of size 13*13 pixels around each interest point
   numFeatures = size(fcoors, 1);
   num_cropped = 0;
   total_feature_points = total_feature_points + numFeatures;
   
   for j=1:numFeatures
       fx = fcoors(j,1); fy = fcoors(j,2);
       if fx<=patch_r || fx>=ImgWidth-patch_r || fy<=patch_r || fy>=ImgHeight-patch_r
           %warning('feature [%d,%d] too close to edge, disregarded\n', fx, fy);
           num_cropped = num_cropped + 1;
           continue;
       end
       patch = img(fy-patch_r:fy+patch_r,fx-patch_r:fx+patch_r);
       assert(size(patch,1) == PatchSize && size(patch,2) == PatchSize);
       curNumPatches = curNumPatches + 1;
       patchPool{curNumPatches} = patch;
       clusterPool{curNumPatches} = curNumPatches;
       
       if curNumPatches >= TotalNumPatches
           break;
       end
   end
   
    if curNumPatches >= TotalNumPatches
        break;
    end
   
end

if curNumPatches < TotalNumPatches
    error('%d extracted from %d images, %d away from exptected %d patches\n', curNumPatches, ...
                i, TotalNumPatches-curNumPatches, TotalNumPatches);
else
    fprintf('A2-3 complete[%f s]-- %f avg feature points, %3.1f%% cropped, %d of patches extracted within %d of images.\n', ...
                toc, total_feature_points/cur_img_idx, num_cropped/total_feature_points*100, TotalNumPatches,cur_img_idx);
end



%% A4 Patch clustering with bottom up procedure
%TODO: use dynamic programming to optimize performance
tic;
if UseDP == true
   dp_patch_sim_matrix = zeros(TotalNumPatches);
end
if strcmp(ClusteringMethod, 'bottom-up')
    combos = combntns(1:TotalNumPatches,2);
    for k = 1:size(combos,1);
        left_cluster_idx = combos(k,1);
        right_cluster_idx = combos(k,2);
        assert(left_cluster_idx ~= right_cluster_idx);
        
        left_patch_idxs = clusterPool{left_cluster_idx};
        right_patch_idxs = clusterPool{right_cluster_idx};

        similarity_sum = 0;
        % Calculate similarity average
        [p,q] = meshgrid(left_patch_idxs, right_patch_idxs);
        patch_pairs = [p(:),q(:)]; %[left, right]
        %disp(patch_pairs);
        num_patch_pairs = size(patch_pairs,1);
        for n = 1:num_patch_pairs;
            pair = patch_pairs(n,:);
            if UseDP == true
                if dp_patch_sim_matrix(pair(1),pair(2)) ~= 0
                    pair_similarity = dp_patch_sim_matrix(pair(1),pair(2));
                else
                    pair_similarity = calcSimilarity(patchPool{pair(1)}, ...
                                    patchPool{pair(2)}, SimilarityMethod);
                    dp_patch_sim_matrix(pair(1),pair(2)) = pair_similarity;
                    dp_patch_sim_matrix(pair(2),pair(1)) = pair_similarity;
                end
                similarity_sum = similarity_sum + pair_similarity;
            else
                similarity_sum = similarity_sum + ...
                   calcSimilarity(patchPool{pair(1)}, patchPool{pair(2)}, SimilarityMethod);
            end
        end
        
        avgSimilarity = similarity_sum/num_patch_pairs;
        %fprintf('similarity sum:%f, number:%d, average similarity:%f\n', similarity_sum, num_patch_pairs, avgSimilarity);
        if avgSimilarity > SimilarityThreshold
            % Merge
            clusterPool{left_cluster_idx} = [clusterPool{left_cluster_idx} clusterPool{right_cluster_idx}];
            clusterPool{right_cluster_idx} = [];
        end
            
    end
    
    % Purge empty cells from clusterPool
    emptyClusters = cellfun('isempty', clusterPool); 
    clusterPool(all(emptyClusters,2),:) = [];
        
else
    error('Clustering Method: %s is not recognized.\n', ClusteringMethod);
end

fprintf('A4 complete[%f s]-- %d clusters remains; clustering method:%s; similarity method:%s; use dp? %d\n', ...
                        toc, size(clusterPool, 1), ClusteringMethod, SimilarityMethod, UseDP);
                    
                    

save('vocabulary', 'HistEqual', 'patchPool', 'clusterPool', 'SimilarityMethod', ...
    'InterestPointOperator', 'HarrisCoe','FoerCoe', 'PatchSize', 'ImgWidth', 'ImgHeight');

fprintf('Vocabulary construction complete, saved data to vocabulary.mat\n');







