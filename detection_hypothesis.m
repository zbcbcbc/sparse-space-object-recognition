% DETECTION HYPOTHESIS
% ------------------------------------
% DESCRIPTIONS:
% ------------------------------------
%

% Credit:
% Author: Bicheng Zhang <viczhang1990@gmail.com>
% Company: University of Illinois Urbana Champaign
% Date: July, 2014
import java.net.Socket
import java.io.*

%% Arguments
Endianess = 'big'; %litte, big
OptMsg = '-o allactivations -l -';
ShowResult = true;
DebugMode = true;
VerboseMode = true;
UseRemoteClassifier = true;
ClassifierHost = '192.168.0.114';
ClassifierPort = 8991;
Profiling = true;
TrueLocation = false;
DepthAssisted = true;

%% File directories
ImgDir = './../dataset/drone-testing-depth-cplusplus/';
if TrueLocation
    TrueFileName = './../dataset/drone-testing-depth/drone-locations.txt';
end
LogFileName = './../results/profileData.xls';



%% Key Parameteres
load vocabulary;
load imgrepr-coe;

DetectScale = 'single'; % single, multi
Analyzer = 'neighbor_suppression'; % neighbor_suppression, repeated_part_elimination


ImgReprCoe.subset_ratio = 0.5; % increase will enlarge chance of finding matching vocab
ImgReprCoe.step_size = 24; % increase will decrease number of subimages, recommended
ImgReprCoe.debug_mode = DebugMode;
ImgReprCoe.verbose_mode = VerboseMode;
ImgReprCoe.foerstner_coe.visualize = 'on';
ImgReprCoe.show_progress = true;
ImgReprCoe.depth_assisted = DepthAssisted;
ImgReprCoe.depth_assisted_edge_mthd = 'canny'; % canny, prewitt
ImgReprCoe.depth_assisted_mthd = 'centroid'; % boundingbox, centroid
ImgReprCoe.lazy_match = true;
ImgReprCoe.use_memorization = true;

NeighborSuppressionCoe.debug_mode = DebugMode;
NeighborSuppressionCoe.verbose_mode = VerboseMode;
NeighborSuppressionCoe.neighbor_size = [ImgHeight+1,ImgWidth+1]; % [height, width]
NeighborSuppressionCoe.acti_threshold = 0.5;
NeighborSuppressionCoe.linear_optimize = true; % Selling point use when number of activation is small


%% Initialization
imgs = dir([ImgDir, 'rgb_*.jpg']);
numImgs = size(imgs,1);
windowSize = [ImgHeight, ImgWidth];

alpha_height= ImgHeight*0.30; % Number of pixels
alpha_width = ImgWidth*0.30;



% Initiate client and send line of options
if UseRemoteClassifier == true
    conn = Socket(ClassifierHost, ClassifierPort);
    output_stream = conn.getOutputStream;
    data_output_stream = DataOutputStream(output_stream);
    input_stream = conn.getInputStream;
    data_input_stream = DataInputStream(input_stream);
    fprintf('Connect to Classifier Server: %s:%d success. Options: %s\n', ClassifierHost, ClassifierPort, OptMsg);
    data_output_stream.writeInt(length(OptMsg)); %WARNING: should be big endian
    data_output_stream.writeBytes(char(OptMsg));
    data_output_stream.flush;
    %bytes_to_read = data_input_stream.readInt();
    bytes_available = 0;
    while (bytes_available<4)
        bytes_available = input_stream.available; 
    end
    message_len = zeros(1, 4, 'uint8');
    for i = 1:4
        % Big Endian
        message_len(i) = data_input_stream.readByte;
    end

    message_len = message_len(end);
    message = zeros(1, message_len, 'uint8');
    for i = 1:message_len
        message(i) = data_input_stream.readByte;
    end  
    message = char(message);  

    fprintf('%d bytes received: %s\n', message_len, message);
    onCleanup(@()conn.close);
end

% Initiate data analysis
if Profiling
   if TrueLocation
        truedata = importdata(TrueFileName);
   end
   
   logdata_menu = {'Imgage Name','# Objects','# Correct','# Error','Time(s)-ImgRepr', ...
            'Time(s)-Classificaiton','Time(s)-NeighborSuppr', 'Time(s)-Total', ...
            'Num interest points', 'Num global', 'Avg matching vocab',  'Num exceeding occur limit', 'Avg distbin', 'Avg angbin', ...
            'Highest activation', 'DetectScale','Step Size','Activation Threshold', ...
              'Suppression Neighbor Size'};
    %{      
    if exist(LogFileName, 'file') == 2
        delete(LogFileName);
    end
    %}      
    xlswrite(LogFileName, logdata_menu, 1, sprintf('A%d',1));
    
    Evaluation.tot_interest_point = 0;
    Evaluation.tot_match_vocab = 0;
    Evaluation.tot_interest_loc = 0;
    Evaluation.tot_t = 0;
end


total_iter = 1;

%% Start reading image and detect objects
for curImgIdx = 1:numImgs
    total_detection_t = tic;
    img_name = imgs(curImgIdx).name;
	img = imread([ImgDir img_name]);  
    [imgH,imgW,~] = size(img);
	if HistEqual == true
        imgGray = histeq(rgb2gray(img));
    else
        imgGray = rgb2gray(img);    
    end
    
    if DepthAssisted == true
       splitted = strsplit(img_name, '_');
       img_depth_name = sprintf('depth_%s_%s', splitted{2},splitted{3});
       img_depth = imread([ImgDir img_depth_name]);
       [depthH, depthW, ~] = size(img_depth);
       assert(depthH == imgH && depthW == imgW);
    else
               splitted = strsplit(img_name, '_');
       img_depth_name = sprintf('depth_%s_%s', splitted{2},splitted{3});
       img_depth = imread([ImgDir img_depth_name]);
    end
    
    if (strcmp(DetectScale, 'single')) % Single scale object detection
        [reprCell, coorArray,imgrepr_profile] = imgRepr(imgGray, ImgReprCoe, img_depth);
        if Profiling == true
           disp(imgrepr_profile); 
           Evaluation.tot_interest_point = Evaluation.tot_interest_point + imgrepr_profile.num_interest_points;
           Evaluation.tot_match_vocab =  Evaluation.tot_match_vocab + imgrepr_profile.total_num_matching_vocab;
           Evaluation.tot_interest_loc = Evaluation.tot_interest_loc + imgrepr_profile.total_iter;
        end
       %% Send examples to classifier server
        tic;
        classify_hbar = waitbar(0,sprintf('start classifing features, total amount:%d...', size(reprCell,1)));
        if UseRemoteClassifier == true
            actiMap = zeros(imgH, imgW, 'double');
            actiRecord = zeros(0,3);
            for k = 1:size(reprCell,1)
                snd_msg = '';
                repr = reprCell{k};
                coor = coorArray(k,:);
                if length(repr) < 2
                    warning('repr has zero features in it, disregard...\n');
                    continue;
                end
                for fi = 1:length(repr)
                    if isempty(snd_msg)
                        snd_msg = sprintf('%d',snd_msg,repr(fi)); 
                    else
                        snd_msg = sprintf('%s,%d',snd_msg,repr(fi));  
                    end
                end
                snd_msg = sprintf('%s:',snd_msg);  
                data_output_stream.writeInt(length(snd_msg));
                data_output_stream.writeBytes(char(snd_msg));
            	data_output_stream.flush;
            
                if DebugMode && VerboseMode
                    fprintf('Message:[%s] sent successful.\n', snd_msg);
                end
            
                bytes_available = 0;
                while (bytes_available<4)
                    bytes_available = input_stream.available; 
                end
                message_len = zeros(1, 4, 'uint8');
                for i = 1:4
                    % Big Endian
                    message_len(i) = data_input_stream.readByte;
                end

                message_len = message_len(end);
                message = zeros(1, message_len, 'uint8');
                for i = 1:message_len
                    message(i) = data_input_stream.readByte;
                end  
                message = char(message);   
                if DebugMode && VerboseMode
                    fprintf('%d bytes received: %s',message_len, message);
                end
            
                [token, remain] = strtok(message, char(10));
                classes = strsplit(remain, char(10));
            
           
                [class_1_id, ~] = strtok(classes(2), ':');
                [class_2_id, ~] = strtok(classes(3), ':');
                class_1_id = str2double(class_1_id);
                class_2_id = str2double(class_2_id);
            
                if class_1_id == 1
                    out = strsplit(classes{2});
                    %disp(out);
                    actiMap(coor(2), coor(1)) = str2double(out{2});
                    actiRecord = [actiRecord; str2double(out{2}), coor(1), coor(2)];
                elseif class_2_id == 1
                    out = strsplit(classes{3});
                    %disp(out);
                    actiMap(coor(2), coor(1)) = str2double(out{2});
                    actiRecord = [actiRecord; str2double(out{2}), coor(1), coor(2)];
                end
                waitbar(k/size(reprCell,1), classify_hbar, ...
                    sprintf('%d%% of features classified...', ...
                        int8(k/size(reprCell,1)*100)));             
                
            end
        else
            actiMap = rand(imgH, imgW, 'double');
        end
        close(classify_hbar);
        feature_classification_t = toc;
        if DebugMode && VerboseMode
            fprintf('%d feature classification finished taking time:%f seconds.\n', size(reprCell,1), feature_classification_t);
        end

       %% Analyze the classifier activation map
        tic;
        if UseRemoteClassifier == true
            if strcmp(Analyzer, 'neighbor_suppression')
                [detections,iter] = neighborSuppression(actiMap, actiRecord, NeighborSuppressionCoe);
            elseif strcmp(Analyzer, 'repeated_part_elimination')
                % Requires to repeatedly apply classifer, avoid
            end
        else
            detections = [];
            iter = 0;
        end
        
        activation_analyze_t = toc;
        if VerboseMode
            fprintf('%d objects detected taking %f seconds, %d iterations.\n', size(detections,1), activation_analyze_t, iter);
        end
    elseif strcmp(DetectScale, 'multi')
        error('%s has not been implemented yet.', DetectScale);
    else
       error('%s is not recognized detect scale.', DetectScale);
    end
    
    total_t = toc(total_detection_t);

	h = figure(5); 
    imshow(img);
    %subplot(1,2,1), imshow(img);
    %subplot(1,2,2), imshow(uint8(actiMap.*255));
    hold on;
    for detect_idx = 1:size(detections,1)
        obj_loc = detections(detect_idx,:);
        plot(obj_loc(1), obj_loc(2), 'ys'); 
        rectangle('Position',[obj_loc(1)-windowSize(2)/2, obj_loc(2)-windowSize(1)/2, windowSize(2), windowSize(1)],'LineWidth',2,'EdgeColor','b');
    end
	hold off

    
    
    if Profiling == true
        if TrueLocation
            ind = find(ismember(truedata.textdata, img_name));
            if isempty(ind)
            	error('Image name:%s does not have corresponding true object location data.\n', img_name);
            else
                true_loc = truedata.data(ind,:);
                if (true_loc(1) == -1)
                    num_object = 0; %TODO: store multiple objects locations
                else
                    num_object = 1;
                end
                num_correct = 0;
                num_error = 0;
                for detect_idx = 1:size(detections,1)
                    obj_loc = detections(detect_idx,:);
                    if abs(obj_loc(1)-true_loc(1))^2/alpha_height^2+abs(obj_loc(2)-true_loc(2))^2/alpha_width^2 <= 1
                        num_correct = num_correct + 1;
                    else
                        num_error = num_error + 1;
                    end
            
                end
            end
        else
            num_object = 1;
            num_correct = min(1, size(detections,1));
            num_error = size(detections,1) - num_correct;
        end
                
        
        log_data = {img_name,int16(num_object),int16(num_correct),int16(num_error), ...
                double(imgrepr_profile.total_t), double(feature_classification_t), ...
                double(activation_analyze_t),double(total_t), ...
                int16(imgrepr_profile.num_interest_points), ...
                int16(imgrepr_profile.num_global_locations), ...
                int16(imgrepr_profile.avg_num_matching_vocab), ...
                int16(imgrepr_profile.total_num_exceed_occur_limit), ...
                double(imgrepr_profile.avg_distbin), ...
                double(imgrepr_profile.avg_angbin), ...
                double(max(actiRecord(:,1))), ...
                DetectScale, ImgReprCoe.step_size, ...
                double(NeighborSuppressionCoe.acti_threshold), ...
                 mat2str(NeighborSuppressionCoe.neighbor_size)};
            
            %{
            logdata_menu = {'Imgage Name','# Objects','# Correct','# Error','Time(s)-ImgRepr', ...
            'Time(s)-Classificaiton','Time(s)-NeighborSuppr', 'Time(s)-Total', ...
            'Num interest points', 'Avg matching vocab',  'Num exceeding occur limit', ...
            'DetectScale','Analyzer','Step Size','Activation Threshold', ...
              'Suppression Neighbor Size'};
                 %}
         
        xlswrite(LogFileName, log_data, 1, sprintf('A%d',total_iter+1));
        save_name = sprintf('./../results/%s_marked.jpg',img_name);
        saveas(h,save_name);
            
        fprintf('%d objects detected. %d in range, %d out of range, img_repr_t:%f, class_t:%f, total_t:%f\n',... 
                num_correct+num_error, num_correct, num_error, imgrepr_profile.total_t, feature_classification_t, double(total_t));
            
        Evaluation.tot_t = Evaluation.tot_t + double(total_t);
    end
        
    
    
    
    waitforbuttonpress;
    
    total_iter = total_iter + 1;
    close(h);
    
end

if Profiling
    Evaluation.avg_interest_point = double(Evaluation.tot_interest_point/numImgs);
    Evaluation.avg_match_vocab = double(Evaluation.tot_match_vocab/numImgs);
    Evaluation.avg_interest_loc = double(Evaluation.tot_interest_loc/numImgs);
    Evaluation.avg_time = double(Evaluation.tot_t/numImgs);
    
    disp(Evaluation);
    
end







