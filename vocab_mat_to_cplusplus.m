%% Parameters
PatchPoolFileName = 'patchPool.txt';
ClusterPoolFileName = 'clusterPool.txt';
VocabConfigFileName = 'vocabConfig.txt';
Version = 1;

load vocabulary;

if exist(PatchPoolFileName, 'file')
    delete(PatchPoolFileName);
end

if exist(ClusterPoolFileName, 'file')
    delete(ClusterPoolFileName);
end

if exist(VocabConfigFileName, 'file')
    delete(VocabConfigFileName);
end

numClusters = size(clusterPool,1);
numPatches = size(patchPool,1);

fprintf('Number of patches:%d; number of clusters:%d; Patch Size:[%d,%d]\n', ...
    numPatches, numClusters, PatchSize, PatchSize);


%% Construct config file
config_f = fopen(VocabConfigFileName, 'wt');
fprintf(config_f, 'Version:%d\n', Version);
fprintf(config_f, 'Patch Size:%d\n', PatchSize);
fprintf(config_f, 'Number of Patches:%d\n',numPatches);
fprintf(config_f, 'Number of Clusters:%d\n', numClusters);
fprintf(config_f, 'Image Width:%d\n', ImgWidth);
fprintf(config_f, 'Image Height:%d\n', ImgHeight);
fclose(config_f);

%% Construct patchPool
dlmwrite(PatchPoolFileName, Version);
for n = 1:numPatches;
    patch = patchPool{n};
    dlmwrite(PatchPoolFileName, patch, '-append');
end


%% Construct clusterPool
dlmwrite(ClusterPoolFileName, Version);
for n = 1:numClusters
    cluster = clusterPool{n};
    dlmwrite(ClusterPoolFileName, cluster, '-append');
end