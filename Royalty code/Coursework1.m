clear all
%close all

[data_train, data_test] = getData('Toy_Spiral');

%% Bagging

param.n = 16; %number of bags, n
param.s = size(data_train,1)*(1 - 1/exp(1)); %size of bags s
param.replacement = 1; % 0 for no replacement and 1 for replacement
param.numfunct = 6;
param.numlevels = 5;

% bagging and visualise bags, Choose a bag for root node.
[bags] = bagging(param, data_train);
%visBags(bags, param.replacement);

%% Training Tree

for n = 4:8
    param.n = n;
    for numlevels = 4:8
        param.numlevels = numlevels;
        for numfunct = 4:9
            param.numfunct = numfunct;
            disp('Your Lord and Saviour is training the tree...')
            tic

            [leaves, nodes] = trainForest(bags, param);
            t = toc;
            param.trainingtime = t;
            formatSpec = '... and on the %2.2f second, the Lord said "Let there be a Randomised Forest Tree"';
            fprintf(formatSpec,t)

            %% Test Tree

            % points = [-.5 -.7; .4 .3; -.7 .4; .5 -.5];
            % [classPred] = testForest(param, points, leaves, nodes, 1, 0);
            
            [classPred] = testForest(param, data_test, leaves, nodes, 0, 1);
            pause(0.25)
        end
    end
end