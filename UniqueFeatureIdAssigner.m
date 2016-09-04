classdef UniqueFeatureIdAssigner
    %UNIQUEFEATUREIDASSIGNER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        num_clusters;
        num_relations;
        max_f_occur;
        max_rel_occur;
        num_unique_ids;
        min_feature_id;
        max_feature_id;
        min_feature_relation_id;
        max_feature_relation_id;
        assigned_ids;
        
    end
    
    methods
        
        function obj = UniqueFeatureIdAssigner(numClusters, distRange, angRange, max_feature_occur, max_rel_occur)
            obj.num_clusters = numClusters;
            obj.num_relations = distRange * angRange;
            obj.max_f_occur = max_feature_occur;
            obj.max_rel_occur = max_rel_occur;
            obj.min_feature_id = 1;
            obj.max_feature_id = numClusters*max_feature_occur;
            obj.min_feature_relation_id = obj.max_feature_id + 1;
            obj.max_feature_relation_id = obj.max_feature_id + numClusters^2*distRange*angRange*max_rel_occur;
            obj.num_unique_ids = obj.max_feature_relation_id;
            obj.assigned_ids = zeros(1,obj.num_unique_ids);
            disp(obj.num_unique_ids);
        end

        function [id,obj] = assignFeatureId(obj,n,i)
           %fprintf('assignFeatureId:n:%d,i:%d\n',n,i);
           assert(n>=1 && i>=1);
           if i > obj.max_f_occur
               error('cluster[%d] has %d of occurances which exceeds maximum occurance limit:%d\n', ...
                   n,i,obj.max_f_occur);
           end
           
           id = obj.max_f_occur*(n-1)+i;
           if id > obj.max_feature_id || id < obj.min_feature_id
               error('Id:%d is out of boundary [%d,%d]-> n:%d,i:%d', ...
                    id, obj.min_feature_id, obj.max_feature_id, n, i);
           elseif obj.assigned_ids(1,id) == 1
               error('Id:%d has already been assigned. n:%d,i:%d', ...
                   id, n, i);
           else
               obj.assigned_ids(1,id) = 1;
               %fprintf('id:%d,n:%d,i:%d\n', id, n, i);
           end
        end
        
        function [id,obj] = assignRelationId(obj,m,n1,n2,j)
            %fprintf('assignRelationId:m:%d,n1:%d,n2:%d,j:%d\n',m,n1,n2,j);
            assert(m>=1 && n1>=1 && n2>=1 && j>=1);
            if j > obj.max_rel_occur
                error('%d occurances exceed max occurance limit:%d. ->m:%d,n1:%d,n2:%d,j:%d', ...
                   j,obj.max_rel_f_occur,m,n1,n2,j);
            end      
            
            id = obj.max_feature_id + (n1-1)*obj.num_clusters*obj.num_relations*obj.max_rel_occur+ ...
                    (n2-1)*obj.num_relations*obj.max_rel_occur+(m-1)*obj.max_rel_occur + j;
            if id > obj.max_feature_relation_id || id < obj.min_feature_relation_id
                error('Id:%d is out of boundary [%d,%d]-> m:%d,n1:%d,n2:%d.j:%d', ...
                    id, obj.min_feature_relation_id, obj.max_feature_relation_id, m, n1, n2, j);               
            elseif obj.assigned_ids(1,id) == 1
               error('Id:%d has already been assigned-> m:%d,n1:%d,n2:%d.j:%d', ...
                   id, m, n1, n2, j); 
            else
                obj.assigned_ids(1,id) = 1;
                %fprintf('id:%d,m:%d,n1:%d,n2:%d,j:%d\n',id,m,n1,n2,j);
            end

        end
        
        function obj = refreshIds(obj)
            obj.assigned_ids(1,:) = 0;
            %fprintf('refreshed....\n');
        end
    end
    
end

