clear all
close all

[data_train, data_test] = getData('Toy_Spiral');

%% Bagging

param.n = 4; %number of bags, n
param.s = size(data_train,1)*(1 - 1/exp(1)); %size of bags s
param.replacement = 1; % 0 for no replacement and 1 for replacement

% bagging and visualise bags, Choose a bag for root node.
[bags] = bagging(param, data_train);
visBags(bags, param.replacement);

%% Training Tree

disp('You Lord and Saviour is training the tree...')
tic

param.numfunct = 3;
param.numlevels = 6;
param.rho = 0.8;

[leaves, nodes] = trainTree(bags, param);
t = toc;
formatSpec = '... and on the %2.2f second, the Lord said "Let there be a Randomised Forest Tree"';
fprintf(formatSpec,t)

%% Test Tree