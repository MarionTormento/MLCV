clear all
close all

% Get training and test data
[data_train, data_test] = getData('Toy_Spiral');

%% Setting the parameter of the tree
param.s = size(data_train,1)*(1 - 1/exp(1)); %size of bags s
param.replacement = 1; % 0 for no replacement and 1 for replacement

%% Training Tree

AccTot = [];

% Optional loop for a parameter sweep of the number of bags (i.e. number of
% trees)
for n = 8
    param.n = n;
    % perform bagging with replacement using the training data
    [bags] = bagging(param, data_train);
    % Optional loop for a parameter sweep of the number of levels in each tree
    for numlevels = 8
        param.numlevels = numlevels;
        % Optional loop for a parameter sweep of rho
        for numfunct = 10
            param.numfunct = numfunct;
            
            tic
            %Train forest using bags and defined parameters
            [leaves, nodes] = trainForest(bags, param);
            param.trainingtime = toc;
            
            % calculate the accuracy of the
            Acc = [param.n, param.numlevels, param.numfunct, accuracy(param, data_train, leaves, nodes)];
            AccTot = [AccTot; Acc];
            clear Acc
            
            % Test Tree for the 2D grid or for the sample points.
            [classPred] = testForest(param, data_test, leaves, nodes, 0, 0);
            clear leaves
            clear nodes
        end
    end
    clear bags
end