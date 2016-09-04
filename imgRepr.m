function [reprCell, coorArray, profile] = imgRepr(img, coe, img_depth)
% FUNCTION: image representation 
% ------------------------------------
% ARGUMENTS:
%   - img: gray scale image with any size
%   - coe:
%       - subimg_size: subimage size
%       - subset_ratio: subset of vocab library to be compared with
%       - dist_range: distance bin range
%       - angle_range: angle bin range
%       - vocab_pool: vocab library indexing patches
%       - patch_pool: patch pool
%       - patch_size: patch size
%
%       - interest_point_op: interest point operator
%       - harris_coe: harris coe
%       - sim_threshold: similarity threshold between patch and library
%       - sim_mthd: similarity comparsion method
%
%       - max_feature_occur: feature occurnace limit 
%       - max_feature_rel_occur: relationship occurance limit
%       - step_size: step size for iterating subimg over image
%
%       - debug_mode:
% ------------------------------------
% RETURNS:
%   - reprCell: cell of representations
%   - coorArray: array of corresponding coordinates
%   - repr_t: total time taken
% ------------------------------------
% REMARKS:
% ------------------------------------
%

% Written by:
% Author: Bicheng Zhang <viczhang1990@gmail.com>
% Company: University of Illinois Urbana Champaign
% Date: July, 2014


    
    
    %% Read parameteres
    subsetRatio = coe.subset_ratio; % subset of vocab library to be compared
    distRange = coe.dist_range; % discretized distance bin range
    angleRange = coe.angle_range; % discretized angle bin range
    simThresh = coe.sim_threshold; % similarity threshold
    maxFeatureOccur = coe.max_feature_occur; 
    maxFeatureRelOccur = coe.max_feature_rel_occur;
    stepSize = coe.step_size;
    vocabPool = coe.vocab_pool;
    patchPool = coe.patch_pool;
    patchSize = coe.patch_size;
    subimgSize = coe.subimg_size;
    interest_point_op = coe.interest_point_op;
    harris_coe = coe.harris_coe;
    foerstner_coe = coe.foerstner_coe;
    sim_mthd = coe.sim_mthd;
    debug_mode = coe.debug_mode;
    verbose_mode = coe.verbose_mode;
    profiling = coe.profiling;
    show_progress = coe.show_progress;
    depth_assisted = coe.depth_assisted;
    depth_assisted_mthd = coe.depth_assisted_mthd;
    depth_assisted_edge_mthd = coe.depth_assisted_edge_mthd;
    lazy_match = coe.lazy_match; % Selling point
    useMemorization = coe.use_memorization;
    
    patch_r = floor(patchSize/2);
    vocabSize = size(vocabPool,1);    
    subimgW = subimgSize(2);
    subimgH = subimgSize(1);   
    

    maxDist = (subimgH-2*patch_r) + (subimgW-2*patch_r);

    maxAng = 2*pi;
    relationMap = containers.Map({'1,1','1,2','1,3','1,4' ...
                              '2,1','2,2','2,3','2,4', ...
                              '3,1','3,2','3,3','3,4', ...
                              '4,1','4,2','4,3','4,4', ...
                              %'5,1','5,2','5,3','5,4', ...
                              } , ...
                {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16});
          
    assigner = UniqueFeatureIdAssigner(vocabSize, distRange, angleRange, maxFeatureOccur, maxFeatureRelOccur);
    if debug_mode
        fprintf('Assigner-- Num unique feature ids:%d, max feature occur:%d, max relation occur:%d\n', ...
            assigner.num_unique_ids, assigner.max_f_occur, assigner.max_rel_occur);
    end

    

    %% 1.Initialize image specific parameters
    total_t_start = tic;
    

    [imgH, imgW, imgD] = size(img);
    assert(imgD == 1);

	cur_idx = 0;
    

    if (imgH<subimgH || imgW<subimgW)
        error('Current image size [w:%d,h:%d] is smaller than minimum image size [w:%d,h:%d].', ...
        imgW,imgH,subimgW,subimgH); 
    elseif (subimgH<patchSize || subimgW<patchSize)
        error('Subimage size [w:%d,h:%d] is smaller than patch size [w:%d,h:%d].', ...
        subimgW,subimgH,patchSize(2),patchSize(1));       
    end
    
    total_subimgs =  floor((imgW-subimgW+1)/stepSize*(imgH-subimgH+1)/stepSize); %TODO: not accurate
    coorArray = zeros(total_subimgs,2); % store each feature points coordinate
    reprCell = cell(total_subimgs,1); % store each feature points' representation

    simMemArray = zeros(imgH,imgW,2); % 1st dim: similarity, 2nd dim: best match cluster idx
    
	sum_distbin = 0; % profiling variable
	sum_angbin = 0; % profiling variable
	count_bin = 0; % profiling variable
	count_exceed_occur_limit = 0;     % profiling variable
    total_num_matching_vocab = 0; % profiling variable
    
    total_match_vocab_t = 0; % profiling variable
    total_calc_rel_t = 0; % profiling variable
    total_construct_feature_space_t = 0; % profiling variable
    
    %% 2.Apply interest operator
    tic;
	if strcmp(interest_point_op, 'harris')
        [fcoors,~] = harris(img, harris_coe, false);
    elseif strcmp(interest_point_op, 'foerstner')
        [fcoors] = foerstner(img, foerstner_coe);
    else
        error('%s is not a recognized interest point operator.\n', interest_point_op);
    end
    
    interest_op_t = toc;
    num_interest_points = size(fcoors,1);
    if debug_mode && verbose_mode
       fprintf('Extract interest point[%f s]. %d points extracted with operation:%s\n', ...
            interest_op_t, num_interest_points, interest_point_op);
    end

    if show_progress == true
        imgrepr_hbar = waitbar(0,'start processing image...');
    end
    
    %% 3. Depth Assistance Global Gist 
    
    if depth_assisted == true
       stepSize = 1;
       assert(nargin==3);
       [depthH,depthW,depthD] = size(img_depth);
       assert(depthH == imgH && depthW == imgW && depthD == 1);
       
       %% 3.1 Find contour
       if strcmp(depth_assisted_edge_mthd, 'canny')
           contour = edge(img_depth,'canny', [0.2 0.8], 1);
         
       elseif strcmp(depth_assisted_edge_mthd, 'prewitt')
           contour = edge(img_depth,'prewitt');
       else
           error('%s edge detection method not recognized.\n', depth_assisted_edge_mthd); 
       end
       %% 3.2 locate global position
       if strcmp(depth_assisted_mthd,'centroid')
           stats = regionprops(contour, 'centroid');
           global_locations = floor(cat(1, stats.Centroid));
       elseif strcmp(depth_assisted_mthd,'boundingbox')
           %g_filter = fspecial('gaussian', [3, 3], 1);
           %contour = imfilter(contour, g_filter, 'same'); 
           stats = regionprops(contour, 'boundingbox');
           boundingboxes = cat(1, stats.BoundingBox);
           global_locations = [];
           for box_idx = 1:size(boundingboxes,1)
              bounding_box = boundingboxes(box_idx,:);
              if bounding_box(3) < 1/5*subimgW || bounding_box(4) < 1/5*subimgH
                  %fprintf('bounding bix width:%d, height:%d too small, ignore\n', ...
                   %     bounding_box(3), bounding_box(4));
                  continue;
              end
              global_locations = [global_locations; ...
                    floor(bounding_box(1)+bounding_box(3)/2), ...
                    floor(bounding_box(2)+bounding_box(4)/2)];
           end
           %error('%s has not finished implementation.', depth_assisted_mthd);
           %TODO: locate the central location of the bounding box
       elseif strcmp(depth_assisted_mthd, '')
           
       else
           error('%s edge method is unrecognized.', depth_assisted_mthd);
       end
       if debug_mode
           if verbose_mode
                fprintf('%d global locations found by %s method.\n', size(global_locations,1), depth_assisted_mthd);
           end
           imshow(contour);
           hold on
           if strcmp(depth_assisted_mthd,'centroid')
               plot(global_locations(:,1), global_locations(:,2), 'b*');
           elseif strcmp(depth_assisted_mthd,'boundingbox')
               for i = 1:numel(stats)
                   rectangle('Position', stats(i).BoundingBox, ...
                    'Linewidth', 3, 'EdgeColor', 'r', 'LineStyle', '--');
               end 
           end
           hold off
           %waitforbuttonpress;
       end
       
       
       
       
    end
    
    if depth_assisted == true
        num_global_locations = size(global_locations, 1);
        break_flag = 0;
    end
    
    
    
    %% 4. Iterate over image
	for x = 1:stepSize:imgW-subimgW+1
        for y = 1:stepSize:imgH-subimgH+1
            
            loc_x = x;
            loc_y = y;
            cur_idx = cur_idx + 1;
            if depth_assisted == true
                if cur_idx > num_global_locations
                   break_flag = 1;
                   cur_idx = cur_idx - 1;
                   break; 
                end
                global_loc = global_locations(cur_idx, :);
                loc_x = global_loc(1);
                loc_y = global_loc(2);
                if show_progress == true
                    waitbar(cur_idx/num_global_locations, imgrepr_hbar, ...
                        sprintf('%d%% of image processed...', int8(cur_idx/num_global_locations*100)));             
                end
                if loc_x+subimgW/2 >= imgW || loc_y+subimgH/2 >= imgH || ...
                           loc_x-subimgW/2 <= 0 || loc_y-subimgH/2 <= 0
                    warning('[imgRepr WARNING]: x:%d,y:%d is considered out of image boundary\n', loc_x,loc_y);
                    continue;
                end  
                


            else   
                if show_progress == true
                    waitbar(cur_idx/total_subimgs, imgrepr_hbar, ...
                        sprintf('%d%% of image processed...', int8(cur_idx/total_subimgs*100)));             
                end
                
                if loc_x+subimgW-1 > imgW || loc_y+subimgH-1 > imgH
                    warning('x:%d,y:%d is out of image boundary\n', loc_x,loc_y);
                    continue;
                end   
            end

            
            %% B0. Accessing the pre-computed subimage features
            if (depth_assisted)
                subimg_features = fcoors(fcoors(:,1)>=loc_x-subimgW/2 & fcoors(:,1)<= loc_x+subimgW/2 & ...
                    fcoors(:,2)>=loc_y-subimgH/2 & fcoors(:,2)<=loc_y+subimgH/2,:);                
            else
                subimg_features = fcoors(fcoors(:,1)>=loc_x & fcoors(:,1)<= loc_x+subimgW-1 & ...
                    fcoors(:,2)>=loc_y & fcoors(:,2)<=loc_y+subimgH-1,:);
            end 
            %fprintf('Number of subimg features:%d\n', size(subimg_features,1));
                
            occurVector = [];
            vocabVector = [];
            coorVector = [];
            occurMap = containers.Map();
                
            %% B1. Compare features to vocab
            %% 4.2 
            tic;
            for j=1:size(subimg_features,1)
              %% 4.2.1 Extract patch from img
                fx = subimg_features(j,1); fy = subimg_features(j,2);
                if fx<=patch_r || fx>=imgW-patch_r || fy<=patch_r || fy>=imgH-patch_r
                    %warning('feature [%d,%d] too close to edge, disregarded\n', fx, fy);
                    continue;
                end
                patch = img(fy-patch_r:fy+patch_r,fx-patch_r:fx+patch_r);
                assert(size(patch,1) == patchSize && size(patch,2) == patchSize);
        
                max_similarity = 0;
                max_similar_cluster_idx = 0; 
                
                if useMemorization == true
                    max_similarity = simMemArray(fy,fx,1);
                    max_similar_cluster_idx = simMemArray(fy,fx,2);
                end
 
                    
        
                %% 4.2.2 Compare the patch to the vocabulary
                if max_similar_cluster_idx == 0
                    % Find the best match in vocab library
                    for k = 1:vocabSize
                        cur_cluster = vocabPool{k};
                        numPatches = size(cur_cluster, 2);
                        similarity_sum = 0;
                        subsetSize = ceil(numPatches*subsetRatio);
                        subsetIdxs = randperm(numPatches, subsetSize);
                        for subsetIdx = subsetIdxs
                            similarity_sum = similarity_sum + calcSimilarity(patch, ...
                                   patchPool{cur_cluster(subsetIdx)}, sim_mthd);
                        end
                        similarity = similarity_sum/subsetSize;
                        if similarity > max_similarity
                            max_similarity = similarity;
                            max_similar_cluster_idx = k;
                        end
                        
                        %fprintf('use_mem:%d, lazy_match:%d, max_sim:%f, simThresh:%f\n', useMemorization, lazy_match, max_similarity, simThresh);
                        if lazy_match && max_similarity > simThresh
                            %fprintf('lazy match at work!\n');
                            break;
                        end
                    end
                end
                
                %% 4.2.3 Record the patch if determined similar
                if max_similarity > simThresh
                    simMemArray(fy,fx,1) = max_similarity;
                    simMemArray(fy,fx,2) = max_similar_cluster_idx;
                    to_ignore = false;
                    % Find a match in library. TODO: turn on simThresh
                    occur_key = sprintf('%d',max_similar_cluster_idx);
                    if isKey(occurMap, occur_key)
                        num_occurances = occurMap(occur_key) + 1;
                        if num_occurances <= maxFeatureOccur
                            occurMap(occur_key) = num_occurances;
                        else 
                            to_ignore = true;
                        end
                    else
                        % Assign new occur key
                        occurMap(occur_key) = 1;
                    end
                    if ~to_ignore
                        occurVector = [occurVector occurMap(occur_key)];
                        vocabVector = [vocabVector max_similar_cluster_idx];
                        coorVector = [coorVector; fx, fy]; 
                    end
                else
                    simMemArray(fy,fx,1) = -1;
                    simMemArray(fy,fx,2) = -1;                    
                end
            end 
            
            match_vocab_t = toc;
            total_match_vocab_t = total_match_vocab_t + match_vocab_t;
            numMatchingVocab = size(vocabVector,2);
           	if debug_mode && verbose_mode
                fprintf('Matching vocab[%f s]: %d matching vocab found.\n',match_vocab_t, numMatchingVocab);
            end
            
            total_num_matching_vocab = total_num_matching_vocab + numMatchingVocab;
            
            %% B2. Calculate Relations over Detected Parts
            %% 4.3
            tic;
            
            if numMatchingVocab < 2
                if debug_mode && verbose_mode
                        warning('[imgRepr WARNING]: Not enough matching vocab found, pass...\n');
                end
            else
                patchRelMatrix = zeros(numMatchingVocab);
                relOccurMatrix = zeros(numMatchingVocab);
                relOccurMap = containers.Map(); % Store relation occurances count
                
                %%TODO: case when only 1 matching vocab exists
                for vi = 1:numMatchingVocab
                    for vj = 1:numMatchingVocab
                        coor_i = coorVector(vi,:);
                        coor_j = coorVector(vj,:);

                        % Calculate patch dist, angles (coor_j wrt coor_i)
                        dist = abs(coor_i(1)-coor_j(1)) + abs(coor_i(2)-coor_j(2));

                        
                        dist_bin = ceil(dist/maxDist * distRange);   
                        if dist_bin == 0
                            dist_bin = 1;
                        elseif dist_bin > distRange
                            dist_bin = distRange;
                            warning('dist:%f,maxDist:%f,distRange:%d,dist_bin:%d\n', ...
                                dist,maxDist,distRange,dist_bin);
                        end
                        
                        ang = atan2(coor_j(2)-coor_i(2),coor_j(1)-coor_i(1));
                        if ang < 0
                            ang = ang + 2*pi;
                        end
                        ang_bin = ceil(ang/maxAng * angleRange);
                        if ang_bin == 0
                            ang_bin = 1;
                        elseif ang_bin > angleRange
                            ang_bin = angleRange;
                            warning('ang:%f,maxAng:%f,angleRange:%d,ang_bin:%d\n', ...
                                ang,maxAng,angleRange,ang_bin);
                        end
                
                        patchRelMatrix(vi,vj) =  relationMap(sprintf('%d,%d',dist_bin,ang_bin));
                
                        rel_key = sprintf('%d,%d,%d',vocabVector(vi),vocabVector(vj),patchRelMatrix(vi,vj));
                        if isKey(relOccurMap, rel_key)
                            num_occurances = relOccurMap(rel_key) + 1;
                            if num_occurances > maxFeatureRelOccur
                            %% RelOccur clean to 0 bug 
                                %relOccurMatrix(vi,vj) = 0; 
                                if debug_mode == true
                                    count_exceed_occur_limit = count_exceed_occur_limit + 1;
                                end
                                continue;
                            else
                                relOccurMap(rel_key) = num_occurances;
                            end
                        else
                            relOccurMap(rel_key) = 1;
                        end
                
                        relOccurMatrix(vi,vj) = relOccurMap(rel_key);

                        sum_distbin = sum_distbin + dist_bin;
                        sum_angbin = sum_angbin + ang_bin;
                        count_bin = count_bin + 1;

                    end
                end
                calc_rel_t = toc;
                total_calc_rel_t = total_calc_rel_t + calc_rel_t;
                if debug_mode && verbose_mode 
                    fprintf('Calculate relationship[%f s]: avg_distbin:%f ,avg_angbin:%f, %d exceed occurance limit.\n', ...
                        calc_rel_t,sum_distbin/count_bin,sum_angbin/count_bin,count_exceed_occur_limit);
                end
    
        
              %% B3. Construct Feature Vector
              %% 4.4
                tic;
            
                featureVector = zeros(1,numMatchingVocab);
                relFeatureVector = zeros(1,numMatchingVocab^2-numMatchingVocab); %TODO;
    
                assert(isempty(find(occurVector==0,1)));
                for fi = 1:numMatchingVocab
                    [featureVector(fi),assigner] =  assigner.assignFeatureId(vocabVector(fi), ...
                                    occurVector(fi));
                end
    
                cur_relFeature_idx = 0;
                for fi = 1:numMatchingVocab
                    for fj = 1:numMatchingVocab
                        if fi == fj || relOccurMatrix(fi,fj) == 0
                            continue; % Ignore self-relations
                        else
                            cur_relFeature_idx = cur_relFeature_idx + 1;
                            [relFeatureVector(cur_relFeature_idx),assigner] = assigner.assignRelationId( ...
                            patchRelMatrix(fi,fj), fi, fj, relOccurMatrix(fi,fj)); 
                        end
                    end
                end
            
                relFeatureVector = relFeatureVector(1:cur_relFeature_idx);
                assert(cur_relFeature_idx == size(relFeatureVector,2));
    
                repr = [featureVector, relFeatureVector];
                assert(isempty(find(repr==0,1)));
                assert(length(unique(repr)) == length(repr));    
                coorArray(cur_idx,:) = [loc_x,loc_y];
                %fprintf('added x:%d, y:%d into coorArray\n', loc_x, loc_y);
                reprCell{cur_idx} = repr;
            
                construct_feature_space_t = toc;
                total_construct_feature_space_t = total_construct_feature_space_t + construct_feature_space_t;
                if debug_mode && verbose_mode
                    fprintf('Construct feature space[%f s]: %d feature ids, %d relation ids, %d total ids.\n', ...
                        construct_feature_space_t, size(featureVector,2), size(relFeatureVector,2), size(repr,2)); 
                end
               
            end
            %%B4. Sanity check
            assigner = assigner.refreshIds();
        end
        
        if (depth_assisted && break_flag == 1)
            break;
        end
    end
    
    good_repr_indxes = find(coorArray(:,1)~=0 | coorArray(:,2)~=0);
    coorArray = coorArray(good_repr_indxes,:);
    reprCell = reprCell(good_repr_indxes);

    %assert(size(coorArray,1) == cur_idx);
    
    total_t = toc(total_t_start);

    
    if show_progress == true
        close(imgrepr_hbar);
    end
        
    if debug_mode && verbose_mode
        fprintf('Image repr success[%f s]:%d subimgs, %d interest points, %3.1f%% matching vocab.\n', ...
                total_t, total_subimgs, num_interest_points, total_num_matching_vocab/num_interest_points*100);
    end
    
    if profiling == true
        profile.num_subimgs = total_subimgs;
        profile.num_interest_points = num_interest_points;
        if depth_assisted
            profile.num_global_locations = num_global_locations;
        end
        profile.total_iter = cur_idx;
        %profile.num_img_representations = length(good_repr_indxes);
        profile.total_num_matching_vocab = total_num_matching_vocab;
        if cur_idx == 0
            profile.avg_num_matching_vocab = 0;
        else
            profile.avg_num_matching_vocab = total_num_matching_vocab/cur_idx;
        end
        %profile.matching_vocab_percent = total_num_matching_vocab/num_interest_points;
        profile.total_num_exceed_occur_limit = count_exceed_occur_limit;
        %profile.exceed_occur_limit_percent = count_exceed_occur_limit/total_num_matching_vocab;
        if count_bin == 0
            profile.avg_distbin = 0;
            profile.avg_angbin = 0;
        else
            profile.avg_distbin = sum_distbin/count_bin;
            profile.avg_angbin = sum_angbin/count_bin;
        end

        %profile.vocab_match_percent = total_num_matching_vocab/num_interest_points;
        profile.interest_op_t = interest_op_t;
        if cur_idx == 0
            profile.avg_matching_vocab_t = 0;
        else
            profile.avg_matching_vocab_t = total_match_vocab_t/cur_idx;
        end
        
        if cur_idx == 0
            profile.avg_calc_rel_t = 0;
        else
            profile.avg_calc_rel_t = total_calc_rel_t/cur_idx;
        end
        
        if cur_idx == 0
            profile.avg_construct_feature_space_t = 0;
        else
            profile.avg_construct_feature_space_t = total_construct_feature_space_t/cur_idx;
        end
            profile.total_t = total_t;
        
    else
        profile = [];
    end
    
    
    
end

