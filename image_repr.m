% Part-based Drone detector 
% PartB: Image Representation

% Author: Bicheng Zhang
% University of Illinois Urbana Champaign
% email: zhang368@illinois.edu

%% Arguments
PositiveImgDir = 'dataset/training-positive/';
NegativeImgDir = 'dataset/training-negative/';




%% Initialize Parameteres
load vocabulary;

%% Options for Speed & Optimization & Analysis
DebugMode = true;
Profiling = true;
UseMemorization = 1;
LazyMatch = true;
    
    %% Default Argument
ImgReprCoe.show_progress = false;
ImgReprCoe.subset_ratio = 0.5;
ImgReprCoe.dist_range = 4;
ImgReprCoe.angle_range = 4;
ImgReprCoe.max_feature_occur = 3;
ImgReprCoe.max_feature_rel_occur = 2;
ImgReprCoe.sim_threshold = 0.85;
ImgReprCoe.subimg_size = [ImgHeight, ImgWidth];
ImgReprCoe.step_size = 1;
ImgReprCoe.vocab_pool = clusterPool;
ImgReprCoe.patch_pool = patchPool;
ImgReprCoe.patch_size = PatchSize;
ImgReprCoe.interest_point_op = InterestPointOperator;
ImgReprCoe.harris_coe = HarrisCoe;
ImgReprCoe.foerstner_coe = FoerCoe;
ImgReprCoe.foerstner_coe.visualize = 'off';
ImgReprCoe.sim_mthd = SimilarityMethod;
ImgReprCoe.debug_mode = DebugMode;
ImgReprCoe.profiling = Profiling;
ImgReprCoe.use_memorization = UseMemorization;
ImgReprCoe.lazy_match = LazyMatch;
ImgReprCoe.depth_assisted = false;
ImgReprCoe.depth_assisted_edge_mthd = 'canny'; % canny, prewitt
ImgReprCoe.depth_assisted_mthd = 'centroid'; % boundingbox, centroid

save('imgrepr-coe', 'ImgReprCoe');

disp(ImgReprCoe);

positiveImgs = dir([PositiveImgDir, '*.jpg']);
numPositiveImgs = size(positiveImgs,1);
negativeImgs = dir([NegativeImgDir, '*.jpg']);
numNegativeImgs = size(negativeImgs,1);


posImgNameCell = cell(numPositiveImgs,1);
posImgFeatureCell = cell(numPositiveImgs,1);

negImgNameCell = cell(numNegativeImgs,1);
negImgFeatureCell = cell(numNegativeImgs,1);

pos_num_matching_vocab = 0;
pos_num_interest_points = 0;
pos_total_t = 0;
pos_num_exceed_occur_limit = 0;
neg_num_matching_vocab = 0;
neg_num_interest_points = 0;
neg_total_t = 0;
neg_num_exceed_occur_limit = 0;

total_t = tic;
hbar = waitbar(0,'start processing training images...');
positive_phase = true;
for curImgIdx=1:(numPositiveImgs+numNegativeImgs)
    if positive_phase==true && curImgIdx > numPositiveImgs
        positive_phase = false;
        fprintf('%d positive training images processing finished...\n', numPositiveImgs);
    end
    
    if positive_phase == true
        imgName = positiveImgs(curImgIdx).name;
        img = imread([PositiveImgDir imgName]);
        if HistEqual == true
            imgGray = histeq(rgb2gray(img));
        else
            imgGray = rgb2gray(img);
        end
        posImgNameCell{curImgIdx} = imgName;
    else
        imgName = negativeImgs(curImgIdx-numPositiveImgs).name;
        img = imread([NegativeImgDir imgName]);
        if HistEqual == true
            imgGray = histeq(rgb2gray(img));
        else
            imgGray = rgb2gray(img);
        end
        negImgNameCell{curImgIdx} = imgName;
    end
    
	[imgH, imgW, imgD] = size(imgGray);
    
	assert(imgH == ImgHeight && imgW == ImgWidth);
    
    [repr,coor_array,profile] = imgRepr(imgGray, ImgReprCoe);
    
    if Profiling == true
       disp(profile); 
    end
    
	if positive_phase == true
        pos_num_matching_vocab = pos_num_matching_vocab + profile.total_num_matching_vocab;
        pos_num_interest_points = pos_num_interest_points + profile.num_interest_points;
        pos_total_t = pos_total_t + profile.total_t;
        pos_num_exceed_occur_limit = pos_num_exceed_occur_limit + profile.total_num_exceed_occur_limit;
        posImgFeatureCell{curImgIdx} = repr;
    else
        neg_num_matching_vocab = neg_num_matching_vocab + profile.total_num_matching_vocab;
        neg_num_interest_points = neg_num_interest_points + profile.num_interest_points;
        neg_total_t = neg_total_t + profile.total_t;
        neg_num_exceed_occur_limit = neg_num_exceed_occur_limit + profile.total_num_exceed_occur_limit;        
        negImgFeatureCell{curImgIdx-numPositiveImgs} = repr;
    end
    
    waitbar(curImgIdx/(numPositiveImgs+numNegativeImgs), hbar, ...
            sprintf('%d%% of training image processed...', ...
                    int8(curImgIdx/(numPositiveImgs+numNegativeImgs)*100)));      
    
  
    
end
close(hbar);
fprintf('%d positive imgs -- avg interest points:%f,avg match vocab:%f, avg exceed occur limit:%f, avg t:%f\n', ...
    numPositiveImgs, pos_num_interest_points/numPositiveImgs,  ...
    pos_num_matching_vocab/numPositiveImgs, ...
    pos_num_exceed_occur_limit/numPositiveImgs,  pos_total_t/numPositiveImgs);
    
fprintf('%d negative imgs -- avg interest points:%f,avg match vocab:%f, avg exceed occur limit:%f, avg t:%f\n', ...
    numNegativeImgs, neg_num_interest_points/numNegativeImgs,  ...
    neg_num_matching_vocab/numNegativeImgs, ...
    neg_num_exceed_occur_limit/numNegativeImgs, neg_total_t/numNegativeImgs );
                
save('image-representation', 'posImgFeatureCell','negImgFeatureCell');





