clear all
close all
clc

% Load training and querying data saved from the getData.m or getData_RF.m function
%% FOR Q31/32
load data_train_320.mat
load data_test_320.mat

%% FOR Q33
% load data_train_5_10_5.mat
% load data_test_5_10_5.mat

%% Setting the parameter of the tree
param.s = size(data_train,1)*(1 - 1/exp(1)); %size of bags s
param.replacement = 1; % 0 for no replacement and 1 for replacement

%% Training Tree
AccTot = [];

%Find the size of the training data (i.e. size of vocabulary)
param.dimensions = size(data_train,2)-1;

% Optional parameter sweep loops
for n = [10 15 30 50]
    param.n = n;
    [bags] = bagging(param, data_train);
    for numlevels =[7 8 9]
        param.numlevels = numlevels;
        for numfunct = [5 10]
            param.numfunct= numfunct;
            
            %Train the forest using the bags and parameters defined
            tic
            [leaves, nodes] = trainForest3(bags, param);
            param.trainingtime = toc;
            
            % query the trees and evaluate the accuracy of the classification of
            % the query images.
            tic
            [classPred] = testForest3(param, data_test, leaves, nodes, 0, 0);
            Acc = [param.n, param.numlevels, param.numfunct,accuracy(param, data_test, classPred)];
            AccTot = [AccTot; Acc];
            param.testingtime = toc;

            clear Acc
            
            % Calculate the confussion matrix for the image categories
            [Conf, order] = confusionmat(data_test(:,param.dimensions+1),classPred(:,1));
            Conf = 100/15.*Conf;
            
            clear leaves
            clear nodes
        end
    end
    clear bags
end