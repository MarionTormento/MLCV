clear all
%close all

[data_train, data_test] = getData('Toy_Spiral');

%% Setting the parameter of the tree
param.s = size(data_train,1)*(1 - 1/exp(1)); %size of bags s
param.replacement = 1; % 0 for no replacement and 1 for replacement

%% Training Tree
AccTot = [];
for n = 4:8
     param.n = n; %nb of bags
     [bags] = bagging(param, data_train);
<<<<<<< HEAD
    for numlevels = 4:8
         param.numlevels = numlevels;
        for numfunct = 3:10
=======
    for numlevels = 7
         param.numlevels = numlevels;
        for numfunct = 100
>>>>>>> bf8f911fff1042420fc72bcd59f4a585c204cf3e
            param.numfunct = numfunct;
            disp('Your Lord and Saviour is training the tree...')
            tic

            [leaves, nodes] = CopytrainForest(bags, param);
            t = toc;
            param.trainingtime = t;
            formatSpec = '... and on the %2.2f second, the Lord said "Let there be a Randomised Forest Tree"';
            fprintf(formatSpec,t)
            Acc(1,1) = param.n;
            Acc(1,2) = param.numlevels;
            Acc(1,3) = param.numfunct;
            Acc(1,4) = accuracy(param, data_train, leaves, nodes);
            AccTot = [AccTot; Acc];
            clear Acc
            
            % Test Tree
             points = [-.5 -.7; .4 .3; -.7 .4; .5 -.5];
             [classPred] = testForest(param, points, leaves, nodes, 0, 0);
            
            [classPred] = testForest(param, data_test, leaves, nodes, 0, 0);
            %pause(0.25)
        clear leaves
        clear nodes
        end
    end
    clear bags
end