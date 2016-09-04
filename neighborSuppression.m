function [detections,iter] = neighborSuppression(actiMap, actiRecord, coe)
% FUNCTION: neighborhood suppression
% ------------------------------------
% ARGUMENTS:
%   - actiMap: activation map
%   - neighbor_size: neighborhood size
%   - 
% ------------------------------------
% RETURNS:
%   - detections: x,y coordinates of detected objs
%   - iter: total iterations used
% ------------------------------------
% REMARKS:
% ------------------------------------
%

% Written by:
% Author: Bicheng Zhang <viczhang1990@gmail.com>
% Company: University of Illinois Urbana Champaign
% Date: July, 2014
    
    %% Initialization
    neighbor_size = coe.neighbor_size;
    acti_threshold = coe.acti_threshold;
    use_linear_optimize = coe.linear_optimize; % Selling point
    debug_mode = coe.debug_mode;
    verbose_mode = coe.verbose_mode;
    
	[imgH, imgW] = size(actiMap);
    assert(imgH>neighbor_size(1) && imgW>neighbor_size(2));
	neighbor_height_r = floor(neighbor_size(1)/2);
	neighbor_width_r = floor(neighbor_size(2)/2); 
    
    iter = 0;
    detections = [];   
    

    if use_linear_optimize == true
        %fprintf('use_linaer_optimize:%d\n', use_linear_optimize);
        actiRecord(actiRecord(:,1)<acti_threshold, :) = [];
        
        
        num_activations = size(actiRecord,1);
        
        for i = 1:num_activations
            iter = iter + 1;
            acti = actiRecord(i,1);
            coor = actiRecord(i, 2:3);
            
            if acti == 0
                continue;
            end
            
            for j = i+1:num_activations
                neighbor_acti = actiRecord(j,1);
                neighbor_coor = actiRecord(j, 2:3);
                
                if neighbor_acti == 0
                   continue; 
                end
                
                if abs(coor(1)-neighbor_coor(1)) <= neighbor_size(2) && ...
                    abs(coor(2)-neighbor_coor(2)) <= neighbor_size(1)
                   if acti > neighbor_acti
                       actiRecord(j,:) = 0;
                   else
                       actiRecord(i,:) = 0;
                   end
                end
            end
            
            if actiRecord(i,1) ~= 0
               if debug_mode && verbose_mode
                    fprintf('Adding detection with acti:%f, loc_x:%d, loc_y:%d\n', actiRecord(i,1), actiRecord(i, 2), actiRecord(i, 3));
               end
               detections = [detections; actiRecord(i, 2:3)]; 
            end
        end
        
    else
        num_activations = length(nnz(actiMap));
        suppMap = padarray(ones(imgH-neighbor_size(1)+1, imgW-neighbor_size(2)+1), ...
                [neighbor_height_r, neighbor_width_r]); % 1: unsuppressed, 0: suppressed
        %disp(size(actiMap));
        %disp(size(suppMap));
        assert(size(suppMap,1) == size(actiMap,1) && size(suppMap,2)==size(actiMap,2));
    
        % Suppress activations lower than threshold
        actiMap(actiMap(:)<acti_threshold) = 0;
    
        %% Starting loop
        while ( num_activations && ~isempty(nonzeros(suppMap)))
            iter = iter + 1;
            unsuppedMap = suppMap.*actiMap;
            suppedMap = ~suppMap.*actiMap;
        
            if isempty(nonzeros(unsuppedMap))
                break;
            end
        
            % find highest unsuppressed activation innnz the map
            [val,loc] = max(unsuppedMap(:)); 
            [row,col] = ind2sub(size(unsuppedMap), loc);
            if row-neighbor_height_r <= 0 || row+neighbor_height_r > imgH || ...
                col-neighbor_width_r <= 0 || col+neighbor_width_r > imgW
                suppMap(row,col) = 0;
                continue;
            end
        
            actiMap_neighbor = actiMap((row-neighbor_height_r):(row+neighbor_height_r), ...
                col-neighbor_width_r:col+neighbor_width_r);
            all_neighbor_max = max(actiMap_neighbor(:));
            suppedMap_neighbor = suppedMap(row-neighbor_height_r:row+neighbor_height_r, ...
                                           col-neighbor_width_r:col+neighbor_width_r);
            supped_neighbor_max = max(suppedMap_neighbor(:));
        
            if val >= all_neighbor_max
                % Output as a detection
                detections = [detections; col,row];
                %fprintf('val:%f, all_neighbor_max:%f\n', val, all_neighbor_max);
                suppMap(row-neighbor_height_r:row+neighbor_height_r, ...
                    col-neighbor_width_r:col+neighbor_width_r) = 0;
            elseif val < supped_neighbor_max
                % Marked as suppressed
                suppMap(row,col) = 0;
            else
                continue;
            end     
        end        
    end

    



end

