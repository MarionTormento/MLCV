clear all
close all
clc

load data_train_256.mat
load data_test_256.mat
% [data_train, data_test] = getData('Caltech');

%% Setting the parameter of the tree
param.s = size(data_train,1)*(1 - 1/exp(1)); %size of bags s
param.replacement = 1; % 0 for no replacement and 1 for replacement

%% Training Tree
AccTot = [];
param.dimensions = size(data_train,2)-1;

for n = [10 15 20 25]
    param.n = n; %nb of bags
    [bags] = bagging(param, data_train);
    for numlevels = 7:9
        param.numlevels = numlevels;
        for numfunct = [5 10 15]
            param.numfunct = numfunct;
            disp('Your Lord and Saviour is training the tree...')
            tic
            
            [leaves, nodes] = CopytrainForest(bags, param);
            t = toc;
            param.trainingtime = t;
            formatSpec = '... and on the %2.2f second, the Lord said "Let there be a Randomised Forest Tree"';
            fprintf(formatSpec,t)
            
            % Test Tree
            
            [classPred] = testForest(param, data_test, leaves, nodes, 0, 0);
            Acc(1,1) = param.n;
            Acc(1,2) = param.numlevels;
            Acc(1,3) = param.numfunct;
            Acc(1,4) = accuracy(param, data_test, classPred);
            AccTot = [AccTot; Acc];
            clear Acc
            
            [Conf, order] = confusionmat(data_test(:,param.dimensions+1),classPred(:,1));
            Conf = 100/15.*Conf;
            clear leaves
            clear nodes
        end
    end
    clear bags
end