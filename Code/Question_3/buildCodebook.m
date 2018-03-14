function [leaves nodes] = buildCodebook(desc_sel, param);

%% Setting the parameter of the tree
param.s = size(desc_sel,1)*(1 - 1/exp(1)); %size of bags s
param.replacement = 1; % 0 for no replacement and 1 for replacement

%% Training Tree
param.dimensions = size(desc_sel,2)-1;

[bags] = bagging(param, desc_sel);

disp('Your Lord and Saviour is training the tree...')
tic

[leaves, nodes] = trainForest3(bags, param);
t = toc;
param.trainingtime = t;
formatSpec = '... and on the %2.2f second, the Lord said "Let there be a Randomised Forest Tree"';
fprintf(formatSpec,t)

clear bags
end