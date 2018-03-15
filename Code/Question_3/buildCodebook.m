% build Codebook for question 3-3 using the random forest
function [leaves nodes] = buildCodebook(desc_sel, param);

%% Setting the parameter of the tree
param.s = size(desc_sel,1)*(1 - 1/exp(1)); %size of bags s
param.replacement = 1; % 0 for no replacement and 1 for replacement

%% Training Tree
param.dimensions = size(desc_sel,2)-1;

[bags] = bagging(param, desc_sel);

tic

[leaves, nodes] = trainForest3(bags, param);
t = toc;
param.trainingtime = t;

clear bags
end