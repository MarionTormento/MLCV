clear all
close all

[data_train, data_test] = getData('Toy_Spiral');

%% Bagging

n = 4; %number of bags, n
s = size(data_train,1)*(1 - 1/exp(1)); %size of bags s
replacement = 1; % 0 for no replacement and 1 for replacement
leafCount = 1;

infoGain = []; %initialise infoGain

% bagging and visualise bags, Choose a bag for root node.
[bags] = bagging(n, s, data_train, replacement);
visBags(bags, replacement, infoGain);

%% Split Function
param.numfunct = 3;
param.numlevels = 6;
param.rho = 0.8;

%% Recursive test

for k = 1:n
    
    %initialize
    tree{1,k} = cell(param.numlevels, 2^(param.numlevels-1));
    
    %Split the root node into the initial children
    rootNode = bags{k};
    tree{1,k}{1,1} = rootNode;
    [children, infoGain] = optimalNodeSplit(param, rootNode);
    clear rootNode
    visNodes(children, replacement, infoGain, k, 1);
    clear infoGain
    tree{1,k}{2,1} = children{1};
    tree{1,k}{2,2} = children{2};
    parent = children;
    clear children
    
    %number of levels in the tree
    for j = 3:param.numlevels 
        %for each child we decide on an optimum split function
        idx = 2^(j-2);
        for i = 1:idx
            if isempty(parent{i})
                children{2*i-1} = cell(0);
                children{2*i} = cell(0);
                continue
            end
            rootNode = parent{i};
            [childrenNew, infoGain] = optimalNodeSplit(param, rootNode);
            visNodes(childrenNew, replacement, infoGain, k, j);
            for ii = 1:length(childrenNew)
                isLeaf = leafTest(childrenNew{ii}); 
                if isLeaf
                    prob1 = sum(childrenNew{ii}(:,3) == 1)/(sum(childrenNew{ii}(:,3) == 1)+sum(childrenNew{ii}(:,3) == 2)+sum(childrenNew{ii}(:,3) == 3));
                    prob2 = sum(childrenNew{ii}(:,3) == 2)/(sum(childrenNew{ii}(:,3) == 1)+sum(childrenNew{ii}(:,3) == 2)+sum(childrenNew{ii}(:,3) == 3));
                    prob3 = sum(childrenNew{ii}(:,3) == 3)/(sum(childrenNew{ii}(:,3) == 1)+sum(childrenNew{ii}(:,3) == 2)+sum(childrenNew{ii}(:,3) == 3));
                    leaf{leafCount} = [prob1, prob2, prob3];
                    leafCount = leafCount + 1;
                    if ii == 1
                        children{2*i-1} = cell(0);
                    elseif ii == 2
                        children{2*i} = cell(0); 
                    end
                end
                if ~(isLeaf)
                    tree{1,k}{j,2*i-1} = childrenNew{1};  
                    tree{1,k}{j,2*i} = childrenNew{2};
                    % Collect a next children array for next branch
                    if ii == 1
                        children{2*i-1} = childrenNew{1};
                    elseif ii == 2
                        children{2*i} = childrenNew{2};
                    end
                end
            end
            % Complete the tree

            clear rootNode
            clear childrenNew
            clear infoGain
           % pause               
        end
        
        clear parent
        %redefine our new generation of children as our current children for
        %the next layer of the tree
        parent = children;
        clear children
    end
    clear parent
end