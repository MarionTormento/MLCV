clear all
close all

[data_train, data_test] = getData('Toy_Spiral');

%% Bagging

param.n = 4; %number of bags, n
param.s = size(data_train,1)*(1 - 1/exp(1)); %size of bags s
param.replacement = 1; % 0 for no replacement and 1 for replacement
leafCount = 1;

infoGain = []; %initialise infoGain

% bagging and visualise bags, Choose a bag for root node.
[bags] = bagging(param, data_train);
visBags(bags, param.replacement, infoGain);

%% Split Function
param.numfunct = 3;
param.numlevels = 6;
param.rho = 0.8;

%% Recursive test

for k = 1:param.n
    
    %initialize
    tree{1,k} = cell(param.numlevels, 2^(param.numlevels-1));
    
    %Split the root node into the initial children
    rootNode = bags{k};
    tree{1,k}{1,1} = rootNode;
    [children, infoGain] = optimalNodeSplit(param, rootNode);
    clear rootNode
    visNodes(children, infoGain, k, 1);
    clear infoGain
    tree{1,k}{2,1} = children{1};
    tree{1,k}{2,2} = children{2};
    parent = children;
    clear children;

    %number of levels in the tree
    for j = 3:param.numlevels
        % For each child
        for i = 1:(2^(j-2))
            rootNode = parent{i};
            % If the parent is empty set its children to empty and
            % continue
            if isempty(rootNode)
                children{2*i-1} = cell(0);
                children{2*i} = cell(0);
                continue
            end
            %If the root is already a leaf make its children empty and continue 
            isRootLeaf = leafTest(rootNode);
            if isRootLeaf
                children{2*i-1} = cell(0);
                children{2*i} = cell(0);
                continue
            end
            %Elseif the root is not a leaf, perform split function
            [childrenNew, infoGain] = optimalNodeSplit(param, rootNode);
            visNodes(childrenNew, infoGain, k, j);
            for m = 1:length(childrenNew)
                % For each childNew, check if it is a leaf
                isChildLeaf = leafTest(childrenNew{m}); 
                % If the child is a leaf or it's the last layer of the tree,
                % create a leaf node
                if isChildLeaf
                    %Calculate the probabilites for each classification
                    prob1 = sum(childrenNew{m}(:,3) == 1)/size(childrenNew{m},1);
                    prob2 = sum(childrenNew{m}(:,3) == 2)/size(childrenNew{m},1);
                    prob3 = sum(childrenNew{m}(:,3) == 3)/size(childrenNew{m},1);
                    % Create the leaf and tag it with its tree num,
                    % layer num etc.
                    leaf{leafCount} = [k, j, 2*i-(2-m), prob1, prob2, prob3];
                    leafCount = leafCount + 1;
                    tree{1,k}{j,2*i-(2-m)} = childrenNew{m};
                    children{2*i-(2-m)} = childrenNew{m}; 
                %Elseif child is not leaf, create its new children
                elseif ~(isChildLeaf)
                   tree{1,k}{j,2*i-(2-m)} = childrenNew{m};
                   children{2*i-(2-m)} = childrenNew{m};
                end
                if j == param.numlevels
                    %Calculate the probabilites for each classification
                    prob1 = sum(childrenNew{m}(:,3) == 1)/(sum(childrenNew{m}(:,3) == 1)+sum(childrenNew{m}(:,3) == 2)+sum(childrenNew{m}(:,3) == 3));
                    prob2 = sum(childrenNew{m}(:,3) == 2)/(sum(childrenNew{m}(:,3) == 1)+sum(childrenNew{m}(:,3) == 2)+sum(childrenNew{m}(:,3) == 3));
                    prob3 = sum(childrenNew{m}(:,3) == 3)/(sum(childrenNew{m}(:,3) == 1)+sum(childrenNew{m}(:,3) == 2)+sum(childrenNew{m}(:,3) == 3));
                    %Create the leaf and tag it with its tree num,
                    leaf{leafCount} = [k, j, 2*i-(2-m), prob1, prob2, prob3];
                    leafCount = leafCount + 1;
                end
            end
            clear rootNode
            clear childrenNew
            clear infoGain
        end
        
        clear parent
        %redefine our new generation of children as our current children for
        %the next layer of the tree
        parent = children;
        clear Children
    end
    clear parent
end
